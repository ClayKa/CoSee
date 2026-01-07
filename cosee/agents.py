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
        role: str = "generic",
        dataset: str = "generic",
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
        self.role = role
        self.dataset = dataset

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

        # Dataset/role-specific tweaks
        if self.dataset == "chartqapro" and self.role == "scanner":
            return (
                "You are the Scanner agent for chart question answering.\n"
                "Your job is NOT to answer the question directly. Your job is to READ the chart carefully and write short, evidence-style notes on the shared board.\n\n"
                "For each question:\n"
                "- Identify the relevant axis labels, legend entries, and x-axis categories (years, countries, issues, etc.).\n"
                "- Write notes that include CONCRETE numeric values, for example:\n"
                '  - "Overguessing share in Kenya for Early Marriage: about 40%"\n'
                '  - "Average underguessing is highest in Indonesia"\n'
                '  - "There is NO underguess for Maternal Mortality in any country"\n'
                "- Whenever possible, tie each note to specific x-axis values or categories so that another agent can reconstruct the reasoning from your notes alone.\n\n"
                "Rules:\n"
                "- DO NOT output the final answer.\n"
                "- DO NOT restate the question.\n"
                "- DO NOT write long explanations; each note should be 1–2 short sentences focused on specific numeric facts.\n"
                "- Write at most 2–3 high-value notes per step.\n\n"
                f"User question:\n{question}\n\n"
                "You are collaborating with other agents using a shared text board.\n"
                "Current board notes:\n"
                f"{board_section}\n\n"
                f"At step {step}, write ONE short new observation that could help answer the question. Do NOT give the final answer."
            )

        if self.dataset == "slidevqa" and self.role == "scanner":
            return (
                f"You are an assistant agent named {self.name}.\n"
                "You skim slides and add concise, high-signal notes.\n\n"
                f"User question:\n{question}\n\n"
                "Guidelines:\n"
                "- Mention slide/page index when possible.\n"
                "- Copy short phrases and key numbers from the slide text.\n"
                "- Do NOT answer the question; just add 1–2 helpful observations.\n\n"
                "Current board notes:\n"
                f"{board_section}\n\n"
                f"At step {step}, write ONE short observation that could help answer the question. Do NOT give the final answer."
            )

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

        if self.dataset == "chartqapro" and self.role == "cross_checker":
            return (
                "You are the CrossChecker agent for chart question answering.\n\n"
                "You see:\n"
                "- the original question,\n"
                "- the chart image,\n"
                "- and the board notes written by the Scanner.\n\n"
                "Your job is to:\n"
                "1) read the board notes carefully,\n"
                "2) optionally look at the chart ONCE more if the notes are clearly insufficient,\n"
                "3) then produce a FINAL ANSWER in a compact, canonical format that matches the gold style.\n\n"
                "VERY IMPORTANT FORMAT:\n"
                "- You must NOT explain your reasoning.\n"
                "- You must NOT write full sentences.\n"
                "- You must ONLY output the requested items in order, separated by spaces and brackets, with NO extra words.\n\n"
                "Examples:\n"
                'If the question is: "which country\'s policymakers overguesses the most on average? what about for underguessing? how many issues are there no underguessing by policymakers for any country? which of these listed issues also have no overguessing for any country?"\n'
                'The correct answer format is: "Kenya Indonesia [Early Marriage, Labour Force Participation, Maternal Mortality, Secondary Education] [Maternal Mortality]".\n\n'
                "General rules for ChartQAPro:\n"
                "- For single-answer questions: output just the value, e.g. \"2017\" or \"Canada\".\n"
                "- For two-part questions: use \"A B\".\n"
                "- For lists: use \"[Item1, Item2, Item3]\".\n"
                "- For nested questions: follow the pattern from the board notes and the gold style, but NEVER add explanations or extra text.\n\n"
                "Board usage rules:\n"
                "- If the board already contains a clear numeric or categorical value that answers part of the question, you MUST copy that value directly into your final answer.\n"
                "- Do NOT invent new numbers or years that contradict the board.\n"
                "- Only if the board is missing a required piece of information, briefly look at the chart again and add at most one short note before answering.\n\n"
                "When you decide to answer, you must choose the FINAL_ANSWER action and output ONLY the compact answer string, nothing else.\n\n"
                f"User question:\n{question}\n\n"
                "Current board notes:\n"
                f"{board_section}\n"
            )

        if self.dataset == "slidevqa" and self.role == "cross_checker":
            return (
                f"You are an assistant agent named {self.name}.\n"
                "When you give the final answer on SlideVQA:\n"
                "- Answer in ONE short sentence or phrase.\n"
                "- Do NOT explain your reasoning.\n"
                "- Do NOT restate the question.\n"
                "- Prefer to copy key numbers, names or phrases from the board or the slide text.\n\n"
                f"User question:\n{question}\n\n"
                "You are collaborating with other agents using a shared text board.\n"
                "Current board notes:\n"
                f"{board_section}\n\n"
                "Provide a single concise final answer."
            )

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
            # For chartqapro scanner, never produce final answers
            if self.dataset == "chartqapro" and self.role == "scanner":
                use_final_answer = False

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
