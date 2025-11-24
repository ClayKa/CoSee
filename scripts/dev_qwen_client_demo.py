import os
from typing import Optional

from PIL import Image

from cosee.models.qwen_vl_wrapper import QwenVLClient


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}. Please place a test image there.")
    return Image.open(path).convert("RGB")


def main() -> None:
    model_path = os.environ.get("COSEE_MODEL_PATH", "./models/Qwen3-VL-4B-Instruct")
    image_path: Optional[str] = os.environ.get("COSEE_TEST_IMAGE", "test.jpg")

    image = load_image(image_path)

    client = QwenVLClient(model_path=model_path, device="cpu", dtype="auto")

    question = "Describe this image in one sentence."
    answer = client.generate(
        images=[image],
        question=question,
        board_text=None,
        role_prompt="You are a helpful multimodal assistant.",
        max_new_tokens=32,
    )

    print("MODEL OUTPUT:")
    print(answer)


if __name__ == "__main__":
    main()
