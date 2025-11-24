from __future__ import annotations

from typing import List, Tuple

from PIL import Image

from cosee.agents import Action, Agent
from cosee.board import Board


class CoSeeController:
    """
    Orchestrates a multi-agent CoSee episode:
    - maintains a shared Board,
    - iteratively calls agents to act,
    - applies their Actions,
    - decides when to stop and produces a final answer.
    """

    def __init__(
        self,
        agents: List[Agent],
        max_steps: int = 4,
    ) -> None:
        self.agents = agents
        self.max_steps = max_steps

    def validate_action(self, action: Action) -> bool:
        """
        Validate that the given Action is structurally well-formed.

        - For WRITE_NOTE: view and content must be present.
        - For FINAL_ANSWER: answer must be present.
        """
        if action.type == "WRITE_NOTE":
            return action.view is not None and bool(action.content)
        if action.type == "FINAL_ANSWER":
            return bool(action.answer)
        return False

    def apply_action(
        self,
        board: Board,
        action: Action,
        step: int,
        author: str,
    ) -> None:
        """
        Apply a single Action to the Board (for WRITE_NOTE),
        or record the action in other internal structures if needed.
        """
        if action.type == "WRITE_NOTE":
            assert action.view is not None
            assert action.content is not None
            tags = action.tags or []
            board.add_cell(
                view=action.view,
                content=action.content,
                tags=tags,
                author=author,
                step=step,
            )
        elif action.type == "FINAL_ANSWER":
            # No Board mutation needed for FINAL_ANSWER.
            pass

    def should_stop(
        self,
        step: int,
        max_steps: int,
        has_final_answer: bool,
    ) -> bool:
        """
        Decide whether to stop the CoSee loop.

        Default policy:
        - Stop if we already have a final answer.
        - Or if we reached the maximum number of steps.
        """
        if has_final_answer:
            return True
        return step >= max_steps

    def decide_final_answer(
        self,
        candidate_answers: List[str],
        board: Board,
    ) -> str:
        """
        Decide the final answer given a list of candidate answers and the final Board.

        For Dummy runs:
        - If any candidate answers exist, return the last one.
        - Otherwise, fallback to a generic dummy message including the board summary.
        """
        if candidate_answers:
            return candidate_answers[-1]

        summary = board.to_text(max_cells_per_page=8, max_total_chars=500)
        return (
            "[DUMMY CONTROLLER ANSWER] No agent gave a FINAL_ANSWER; "
            "here is the final board summary:\n" + summary
        )

    def run(
        self,
        images: List[Image.Image],
        question: str,
    ) -> Tuple[str, Board]:
        """
        Run a single CoSee episode.

        - Initializes an empty Board.
        - Iteratively calls agents to act in round-robin over `max_steps`.
        - Applies WRITE_NOTE actions to the Board.
        - Collects FINAL_ANSWER candidates.
        - Stops when a stopping condition is met.
        - Returns (final_answer, final_board).
        """
        board = Board()
        candidate_answers: List[str] = []

        step = 0
        num_agents = len(self.agents)

        while True:
            agent_idx = step % num_agents
            agent = self.agents[agent_idx]

            action = agent.act(
                board=board,
                images=images,
                question=question,
                step=step,
            )

            if not self.validate_action(action):
                raise ValueError(f"Invalid action from agent {agent.name}: {action}")

            self.apply_action(
                board=board,
                action=action,
                step=step,
                author=agent.name,
            )

            if action.type == "FINAL_ANSWER" and action.answer is not None:
                candidate_answers.append(action.answer)

            if self.should_stop(
                step=step,
                max_steps=self.max_steps,
                has_final_answer=bool(candidate_answers),
            ):
                break

            step += 1

        final_answer = self.decide_final_answer(candidate_answers, board)
        return final_answer, board
