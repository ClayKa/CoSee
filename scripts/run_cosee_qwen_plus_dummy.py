import os
from typing import List

from PIL import Image

from cosee.agents import DummyAgent, QwenAgent
from cosee.controller import CoSeeController
from cosee.models.qwen_vl_wrapper import QwenVLClient


def main() -> None:
    model_path = os.environ.get("COSEE_MODEL_PATH", "./models/Qwen3-VL-4B-Instruct")
    image_path = os.environ.get("COSEE_TEST_IMAGE", "test.jpg")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model path not found at {model_path}. "
            "Set COSEE_MODEL_PATH to a valid local Qwen3-VL-4B-Instruct directory."
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Test image not found at {image_path}. "
            "Set COSEE_TEST_IMAGE to a valid image path."
        )

    print(f"Using model_path: {model_path}")
    print(f"Using image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    images: List[Image.Image] = [image]
    question = "What is this document or image about?"

    qwen_client = QwenVLClient(
        model_path=model_path,
        device="cpu",
        dtype="auto",
    )

    qwen_agent = QwenAgent(
        name="QwenScanner",
        role_prompt="Read the page and add useful multimodal observations to the shared board.",
        qwen_client=qwen_client,
        allow_final_answer=False,
        default_gen_kwargs={
            "max_new_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.8,
        },
    )

    cross_checker = DummyAgent(
        name="CrossChecker",
        role_prompt="Review the shared board and propose a concise final answer.",
        max_note_steps=0,
    )

    controller = CoSeeController(
        agents=[qwen_agent, cross_checker],
        max_steps=3,
    )

    final_answer, final_board = controller.run(
        images=images,
        question=question,
    )

    print("=== COSEE QWEN + DUMMY RUN ===")
    print(f"Question: {question}\n")

    print("Final Answer:\n")
    print(final_answer)

    print("\nFinal Board Summary:\n")
    print(
        final_board.to_text(
            max_cells_per_page=8,
            max_total_chars=1000,
        )
    )


if __name__ == "__main__":
    main()
