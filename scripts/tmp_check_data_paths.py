import json
from pathlib import Path
from typing import List

from PIL import Image

# Treat this file as located under scripts/, so ROOT is the repo root.
ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("slidevqa", ROOT / "data" / "slidevqa" / "annotations" / "slidevqa.jsonl"),
    ("infochartqa", ROOT / "data" / "infochartqa" / "annotations" / "infochartqa.jsonl"),
    ("vqaonline", ROOT / "data" / "vqaonline" / "annotations" / "vqaonline.jsonl"),
]

MAX_EXAMPLES_PER_DATASET = 3  # we only inspect a few examples per dataset


def check_dataset(name: str, ann_path: Path) -> None:
    print(f"\n=== Checking dataset: {name} ===")
    if not ann_path.exists():
        print(f"[WARN] Annotation file not found: {ann_path}")
        return

    count = 0
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode error in {ann_path}: {e}")
                break

            ex_id = ex.get("id", "<no id>")
            image_paths: List[str] = ex.get("image_paths", [])
            question = ex.get("question", "")
            answer = ex.get("answer", "")

            print(f"\nExample id: {ex_id}")
            print(f"  question: {question}")
            print(f"  answer:   {answer}")
            print("  image_paths:")

            if not image_paths:
                print("    [WARN] image_paths is empty")

            for rel_path in image_paths:
                img_path = ROOT / rel_path
                exists = img_path.exists()
                print(f"    - {rel_path}  ({'OK' if exists else 'MISSING'})")

                if exists:
                    try:
                        with Image.open(img_path) as im:
                            im.verify()  # lightweight consistency check
                    except Exception as e:
                        print(f"      [ERROR] Failed to open image: {e}")

            count += 1
            if count >= MAX_EXAMPLES_PER_DATASET:
                break

    if count == 0:
        print(f"[INFO] No examples read from {ann_path} (empty file or only blank lines).")


def main() -> None:
    for name, ann_path in DATASETS:
        check_dataset(name, ann_path)


if __name__ == "__main__":
    main()
