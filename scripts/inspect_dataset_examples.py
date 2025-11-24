from __future__ import annotations

from typing import List

from PIL import Image

from cosee.data.datasets import Example, ROOT, load_toy_split


def inspect_slidevqa_examples(
    split: str = "train",
    max_examples: int = 3,
) -> None:
    """
    Load a few SlideVQA toy examples and print their contents,
    including basic checks on image paths.
    """
    examples: List[Example] = load_toy_split(
        dataset="slidevqa",
        split=split,
        max_examples=max_examples,
    )

    print(f"Loaded {len(examples)} slidevqa examples from split='{split}'")

    for idx, ex in enumerate(examples):
        print("\n" + "=" * 80)
        print(f"Example #{idx}")
        print(f"  id:       {ex.id}")
        print(f"  dataset:  {ex.dataset}")
        print(f"  split:    {ex.split}")
        print(f"  question: {ex.question}")
        print(f"  answer:   {ex.answer}")
        print(f"  num images: {len(ex.image_paths)}")

        for img_idx, rel_path in enumerate(ex.image_paths[:5]):
            abs_path = (ROOT / rel_path).resolve()
            status = "OK" if abs_path.is_file() else "MISSING"
            print(f"    [{img_idx}] {rel_path}  ({status})")

        if ex.image_paths:
            first_rel = ex.image_paths[0]
            first_abs = (ROOT / first_rel).resolve()
            try:
                with Image.open(first_abs) as img:
                    print(
                        f"  First image opened successfully: size={img.size}, mode={img.mode}"
                    )
            except Exception as e:
                print(f"  [WARN] Failed to open first image: {first_abs} -> {e}")


def main() -> None:
    inspect_slidevqa_examples(split="train", max_examples=3)


if __name__ == "__main__":
    main()
