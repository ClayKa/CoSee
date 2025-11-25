from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "ChongyanChen/VQAonline"
RAW_DIR = ROOT / "data" / "hf_raw" / "vqaonline"

OUT_VQAONLINE_IMG_DIR = ROOT / "data" / "vqaonline" / "images"
OUT_VQAONLINE_ANN_DIR = ROOT / "data" / "vqaonline" / "annotations"
OUT_VQAONLINE_JSONL = OUT_VQAONLINE_ANN_DIR / "vqaonline_trainval.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VQAonline train+val subset.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Original splits to include (e.g., train validation).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=-1,
        help="If >0, sample up to this many examples after merging splits; else use all.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/vqaonline",
        help="Output root for images/annotations.",
    )
    return parser.parse_args()


def load_split(split: str) -> Tuple[List[Dict[str, Any]], str]:
    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        cache_dir=str(RAW_DIR),
        allow_patterns=[f"{split}.json"],
    )
    json_path = Path(snapshot_path) / f"{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON for split {split}: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data, split


def find_and_copy_image(image_name: str, img_cache: Dict[str, str]) -> str | None:
    if image_name in img_cache:
        return img_cache[image_name]

    src_path = None
    for folder_idx in range(1, 8):
        rel_path = f"images{folder_idx}/{image_name}"
        try:
            candidate = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=rel_path,
                cache_dir=str(RAW_DIR),
            )
            src_path = Path(candidate)
            break
        except Exception:
            continue

    if src_path is None:
        return None

    OUT_VQAONLINE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    dst_path = OUT_VQAONLINE_IMG_DIR / image_name
    if not dst_path.exists():
        shutil.copy2(src_path, dst_path)
    rel_image_path = str(dst_path.relative_to(ROOT))
    img_cache[image_name] = rel_image_path
    return rel_image_path


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_VQAONLINE_ANN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VQAONLINE_IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_examples: List[Tuple[Dict[str, Any], str]] = []
    for split in args.splits:
        data, orig_split = load_split(split)
        all_examples.extend((entry, orig_split) for entry in data)

    if args.max_examples > 0 and len(all_examples) > args.max_examples:
        rng.shuffle(all_examples)
        all_examples = all_examples[: args.max_examples]

    img_cache: Dict[str, str] = {}
    num_written = 0

    with OUT_VQAONLINE_JSONL.open("w", encoding="utf-8") as fout:
        for entry, orig_split in all_examples:
            image_name = entry.get("image")
            if not image_name:
                continue

            rel_image_path = find_and_copy_image(image_name, img_cache)
            if rel_image_path is None:
                print(f"[WARN] Missing image {image_name}, skipping example.")
                continue

            question = str(entry.get("question", "")).strip()
            answer = str(entry.get("answer", "")).strip()

            record = {
                "id": f"vqaonline_trainval_{num_written:06d}",
                "dataset": "vqaonline",
                "split": "trainval",
                "image_paths": [rel_image_path],
                "question": question,
                "answer": answer,
                "meta": {
                    "orig_split": orig_split,
                    "context": entry.get("context", "").strip() if isinstance(entry.get("context", ""), str) else entry.get("context", ""),
                    "topic": entry.get("topic", ""),
                    "url": entry.get("url", ""),
                    "raw_image_name": image_name,
                },
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Wrote {num_written} examples to {OUT_VQAONLINE_JSONL}")
    print(f"Saved {len(img_cache)} unique images under {OUT_VQAONLINE_IMG_DIR}")


if __name__ == "__main__":
    main()
