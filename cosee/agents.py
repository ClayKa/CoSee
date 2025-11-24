from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from PIL import Image

from cosee.board import Board, View

try:
    # Optional import to mirror API; not used by DummyAgent.
    from cosee.models.qwen_vl_wrapper import QwenVLClient  # type: ignore
except Exception:  # pragma: no cover - defensive import guard
    QwenVLClient = None  # type: ignore

ActionType = Literal["WRITE_NOTE", "FINAL_ANSWER"]


@dataclass
class Action:
    """
    A single agent action in the CoSee loop.

    - WRITE_NOTE: append a new note (cell) to the Board.
    - FINAL_ANSWER: propose a final answer, potentially ending the loop.
    """

    type: ActionType

    # For WRITE_NOTE
    view: Optional[View] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None

    # For FINAL_ANSWER
    answer: Optional[str] = None


class Agent(ABC):
    """
    Abstract base class for a CoSee agent.

    Each agent:
    - has a name and a high-level role description,
    - may (or may not) hold a reference to a QwenVLClient,
    - produces an Action given the current Board, visual inputs, and question.
    """

    def __init__(
        self,
        name: str,
        role_prompt: str,
        qwen_client: Optional["QwenVLClient"] = None,
    ) -> None:
        self.name = name
        self.role_prompt = role_prompt
        self.qwen_client = qwen_client

    @abstractmethod
    def act(
        self,
        board: Board,
        images: List[Image.Image],
        question: str,
        step: int,
        **kwargs: Any,
    ) -> Action:
        """
        Produce an Action for this agent at the given step.

        - `board` is the current shared Board state.
        - `images` are the document pages (or empty list for dummy runs).
        - `question` is the user question for this CoSee episode.
        - `step` is the global step index (0, 1, 2, ...).
        """
        raise NotImplementedError


class DummyAgent(Agent):
    """
    A simple agent used for debugging the CoSee loop.

    Behavior:
    - Writes notes for the first `max_note_steps` steps.
    - Emits a FINAL_ANSWER afterwards.
    """

    def __init__(
        self,
        name: str,
        role_prompt: str,
        max_note_steps: int = 2,
    ) -> None:
        super().__init__(name=name, role_prompt=role_prompt, qwen_client=None)
        self.max_note_steps = max_note_steps

    def act(
        self,
        board: Board,
        images: List[Image.Image],
        question: str,
        step: int,
        **kwargs: Any,
    ) -> Action:
        if step < self.max_note_steps:
            page = 1 if images else 1
            view = View(page=page, description=f"dummy step {step}")
            note_content = f"{self.name} (step {step}) is thinking about the question: {question}"
            tags = ["dummy"]
            return Action(
                type="WRITE_NOTE",
                view=view,
                content=note_content,
                tags=tags,
            )

        board_summary = board.to_text(max_cells_per_page=8, max_total_chars=500)
        answer = (
            f"[DUMMY ANSWER by {self.name}] Based on the current board, "
            f"this is a placeholder answer.\nBoard summary:\n{board_summary}"
        )
        return Action(
            type="FINAL_ANSWER",
            answer=answer,
        )


class QwenAgent(Agent):
    """
    An agent that uses QwenVLClient to read images + board context and write new notes.

    Default behavior:
    - At all steps, write a new note derived from images + board + question (WRITE_NOTE).
    - Optionally, if allow_final_answer is True and step >= final_answer_step,
      it can instead produce a FINAL_ANSWER Action.
    """

    def __init__(
        self,
        name: str,
        role_prompt: str,
        qwen_client: "QwenVLClient",
        board_max_cells_per_page: int = 8,
        board_max_chars: int = 1000,
        default_gen_kwargs: Optional[Dict[str, Any]] = None,
        allow_final_answer: bool = False,
        final_answer_step: int = 2,
    ) -> None:
        super().__init__(name=name, role_prompt=role_prompt, qwen_client=qwen_client)
        self.board_max_cells_per_page = board_max_cells_per_page
        self.board_max_chars = board_max_chars
        self.default_gen_kwargs = default_gen_kwargs or {}
        self.allow_final_answer = allow_final_answer
        self.final_answer_step = final_answer_step

    def _select_target_page(
        self,
        images: List[Image.Image],
        step: int,
    ) -> int:
        """
        Choose a 1-based page index for the note.

        Strategy:
        - If there are images, cycle deterministically through them based on step.
        - If no images, default to page 1.
        """
        if images:
            return (step % len(images)) + 1
        return 1

    def _build_note_prompt(
        self,
        question: str,
        board_text: str,
        step: int,
    ) -> str:
        """
        Build a prompt instructing Qwen to write ONE new observation/note,
        NOT the final answer.
        """
        board_section = board_text.strip() if board_text.strip() else "(The shared board is currently empty.)"

        return (
            f"You are an assistant agent named {self.name}.\n"
            f"Your role: {self.role_prompt}\n\n"
            f"User question:\n{question}\n\n"
            "You are collaborating with other agents using a shared text board.\n"
            "Current board notes:\n"
            f"{board_section}\n\n"
            f"At step {step}, write ONE short new observation that could help "
            "answer the question. Do NOT give the final answer. "
            "Do NOT mention that you are writing a note. "
            "Just state the observation directly."
        )

    def _build_final_answer_prompt(
        self,
        question: str,
        board_text: str,
        step: int,
    ) -> str:
        """
        Build a prompt instructing Qwen to provide a FINAL answer,
        using the board as supporting context.
        """
        board_section = board_text.strip() if board_text.strip() else "(The shared board is currently empty.)"

        return (
            f"You are an assistant agent named {self.name}.\n"
            f"Your role: {self.role_prompt}\n\n"
            f"User question:\n{question}\n\n"
            "You are collaborating with other agents using a shared text board.\n"
            "Current board notes:\n"
            f"{board_section}\n\n"
            "Now provide a single, concise final answer to the question, "
            "using the board notes and visual information as evidence. "
            "Do NOT describe your reasoning process explicitly; "
            "just give the final answer."
        )

    def act(
        self,
        board: Board,
        images: List[Image.Image],
        question: str,
        step: int,
        **kwargs: Any,
    ) -> Action:
        """
        Use QwenVLClient to either:
        - write a new note (WRITE_NOTE), or
        - optionally provide a final answer (FINAL_ANSWER),
          depending on configuration and step.

        Extra **kwargs are treated as generation overrides and merged with
        self.default_gen_kwargs.
        """
        board_text = board.to_text(
            max_cells_per_page=self.board_max_cells_per_page,
            max_total_chars=self.board_max_chars,
        )

        gen_kwargs: Dict[str, Any] = dict(self.default_gen_kwargs)
        gen_kwargs.update(kwargs)

        use_final_answer = self.allow_final_answer and step >= self.final_answer_step

        if use_final_answer:
            prompt = self._build_final_answer_prompt(
                question=question,
                board_text=board_text,
                step=step,
            )
            answer_text = self.qwen_client.generate(
                images=images,
                question=prompt,
                board_text=board_text,
                role_prompt=self.role_prompt,
                **gen_kwargs,
            )
            answer_text = (answer_text or "").strip()
            if not answer_text:
                answer_text = "No answer produced."

            return Action(
                type="FINAL_ANSWER",
                answer=answer_text,
            )

        prompt = self._build_note_prompt(
            question=question,
            board_text=board_text,
            step=step,
        )
        note_text = self.qwen_client.generate(
            images=images,
            question=prompt,
            board_text=board_text,
            role_prompt=self.role_prompt,
            **gen_kwargs,
        )
        note_text = (note_text or "").strip()
        if not note_text:
            note_text = "[EMPTY NOTE]"

        page = self._select_target_page(images=images, step=step)
        view = View(page=page, description=f"qwen step {step}")

        return Action(
            type="WRITE_NOTE",
            view=view,
            content=note_text,
            tags=["qwen-note"],
        )
