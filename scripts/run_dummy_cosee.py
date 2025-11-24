from typing import List

from PIL import Image

from cosee.agents import DummyAgent
from cosee.controller import CoSeeController


def main() -> None:
    images: List[Image.Image] = []
    question = "What is this document about?"

    scanner = DummyAgent(
        name="Scanner",
        role_prompt="Scan the document and note high-level structure.",
        max_note_steps=2,
    )
    cross_checker = DummyAgent(
        name="CrossChecker",
        role_prompt="Review the board and propose a final answer.",
        max_note_steps=1,
    )

    controller = CoSeeController(
        agents=[scanner, cross_checker],
        max_steps=4,
    )

    final_answer, final_board = controller.run(
        images=images,
        question=question,
    )

    print("=== DUMMY COSEE RUN ===")
    print(f"Question: {question}")
    print("\nFinal Answer:\n")
    print(final_answer)
    print("\nFinal Board Summary:\n")
    print(final_board.to_text(max_cells_per_page=8, max_total_chars=500))


if __name__ == "__main__":
    main()
