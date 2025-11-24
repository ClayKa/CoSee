import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # repo root

IMG_ROOT = ROOT / "data" / "slidevqa" / "images"
ANN_PATH = ROOT / "data" / "slidevqa" / "annotations" / "slidevqa.jsonl"

# Limit how many train examples to export for the toy subset
MAX_EXAMPLES = 50  # adjust as needed

# Optional: reuse HF cache inside repo
HF_CACHE_DIR = ROOT / "data" / "hf_cache" / "slidevqa"


def main() -> None:
    ds_train = load_dataset(
        "NTT-hil-insight/SlideVQA",
        split="train",
        cache_dir=str(HF_CACHE_DIR),
    )

    if MAX_EXAMPLES is not None and MAX_EXAMPLES < len(ds_train):
        ds_toy = ds_train.select(range(MAX_EXAMPLES))
    else:
        ds_toy = ds_train

    print(f"Exporting {len(ds_toy)} examples to {ANN_PATH}")

    IMG_ROOT.mkdir(parents=True, exist_ok=True)
    ANN_PATH.parent.mkdir(parents=True, exist_ok=True)

    saved_pages: Set[Tuple[str, int]] = set()
    exported = 0

    with ANN_PATH.open("w", encoding="utf-8") as fout:
        for idx, ex in enumerate(ds_toy):
            deck_name: str = ex["deck_name"]
            pages: List[str] = []

            for page_idx in range(1, 21):
                field_name = f"page_{page_idx}"
                img = ex.get(field_name)
                if img is None:
                    continue

                rel_path = f"data/slidevqa/images/{deck_name}/page_{page_idx:02d}.png"
                abs_path = ROOT / rel_path
                page_key = (deck_name, page_idx)

                if page_key not in saved_pages:
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(abs_path)
                    saved_pages.add(page_key)

                pages.append(rel_path)

            if not pages:
                print(f"[WARN] No pages for example idx={idx}, deck={deck_name}. Skipping.")
                continue

            record: Dict[str, Any] = {
                "id": f"slidevqa_train_{idx:06d}",
                "dataset": "slidevqa",
                "split": "train",
                "image_paths": pages,
                "question": ex.get("question", ""),
                "answer": ex.get("answer", ""),
                "meta": {
                    "deck_name": ex.get("deck_name"),
                    "deck_url": ex.get("deck_url"),
                    "qa_id": ex.get("qa_id"),
                    "arithmetic_expression": ex.get("arithmetic_expression"),
                    "evidence_pages": ex.get("evidence_pages"),
                },
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1

    print(f"Exported {exported} examples to {ANN_PATH}")
    print(f"Saved {len(saved_pages)} unique deck pages under {IMG_ROOT}")


if __name__ == "__main__":
    main()
