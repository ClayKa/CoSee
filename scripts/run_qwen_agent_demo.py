import os

from PIL import Image

from cosee.agents import QwenAgent
from cosee.board import Board, View
from cosee.models.qwen_vl_wrapper import QwenVLClient


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}. Set COSEE_TEST_IMAGE to a valid file.")
    return Image.open(path).convert("RGB")


def main() -> None:
    model_path = os.environ.get("COSEE_MODEL_PATH", "./models/Qwen3-VL-4B-Instruct")
    image_path = os.environ.get("COSEE_TEST_IMAGE", "./test.jpg")

    image = load_image(image_path)

    board = Board()
    board.add_cell(
        view=View(page=1, description="seed note"),
        content="Initial note: quick scan of the document.",
        tags=["seed"],
        author="Seeder",
        step=0,
    )

    client = QwenVLClient(model_path=model_path, device="cpu", dtype="auto")

    agent = QwenAgent(
        name="QwenNoteAgent",
        role_prompt="Write one concise observation grounded in the image and board.",
        qwen_client=client,
        allow_final_answer=False,
        default_gen_kwargs={"max_new_tokens": 48},
    )

    question = "What is shown in this image?"
    action = agent.act(
        board=board,
        images=[image],
        question=question,
        step=1,
    )

    print("Returned Action:")
    print(action)

    if action.type == "WRITE_NOTE" and action.view and action.content:
        board.add_cell(
            view=action.view,
            content=action.content,
            tags=action.tags,
            author=agent.name,
            step=1,
        )

    print("\nBoard Summary:")
    print(board.to_text(max_cells_per_page=8, max_total_chars=1000))


if __name__ == "__main__":
    main()
