from __future__ import annotations

import argparse
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _detect_column(column_names: List[str], candidates: List[str], kind: str) -> str:
    lower_to_orig = {name.lower(): name for name in column_names}
    for c in candidates:
        if c in column_names:
            return c
        if c.lower() in lower_to_orig:
            return lower_to_orig[c.lower()]
    raise ValueError(
        f"Could not find a {kind} column; tried {candidates}, available={column_names}"
    )


def normalize_text_field(value: Any) -> str:
    """
    Normalize a Question/Answer field from the raw ChartQAPro example
    into a clean plain-text string.
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            text = ""
        elif len(value) == 1:
            text = value[0]
        else:
            text = " ".join(str(v) for v in value)
    else:
        text = value
    return str(text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a ChartQAPro subset.")
    parser.add_argument("--split", type=str, default="test", help="Split to export.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="If >0, sample this many examples; else use all.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/chartqapro",
        help="Output root for images/annotations.",
    )
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> tuple[Path, Path, Path]:
    img_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_path = ann_dir / "chartqapro_1k.jsonl"
    return img_dir, ann_dir, ann_path


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    img_dir, _, ann_path = ensure_dirs(out_dir)

    ds_dict = load_dataset(
        "ahmed-masry/ChartQAPro",
        cache_dir=str(ROOT / "data" / "hf_cache" / "chartqapro"),
    )
    if args.split not in ds_dict:
        raise ValueError(f"Split '{args.split}' not found in dataset. Available: {list(ds_dict.keys())}")
    ds = ds_dict[args.split]
    print(f"Using split='{args.split}' with {len(ds)} examples.")
    print("Columns:", ds.column_names)

    image_col = _detect_column(ds.column_names, ["image", "img"], "image")
    question_col = _detect_column(ds.column_names, ["question", "question_text", "query"], "question")
    answer_col = _detect_column(ds.column_names, ["answer", "answers", "answer_text", "label"], "answer")

    meta_keys = [c for c in ds.column_names if c not in {image_col, question_col, answer_col}]

    indices = list(range(len(ds)))
    if args.max_examples > 0 and len(indices) > args.max_examples:
        rng.shuffle(indices)
        indices = indices[: args.max_examples]

    selected = [ds[i] for i in indices]

    saved_images: Dict[str, str] = {}
    num_written = 0

    with ann_path.open("w", encoding="utf-8") as fout:
        for idx, ex in enumerate(selected):
            chart_id = str(ex.get("id", f"chart_{idx:06d}") or f"chart_{idx:06d}")
            chart_id = chart_id.replace("/", "_").replace("\\", "_")

            if chart_id in saved_images:
                rel_image_path = saved_images[chart_id]
            else:
                img = ex[image_col]
                if not isinstance(img, Image.Image):
                    if isinstance(img, (bytes, bytearray)):
                        img = Image.open(BytesIO(img)).convert("RGB")
                    else:
                        raise TypeError(f"Expected image column '{image_col}' to be PIL.Image.Image or bytes.")
                image_filename = f"{chart_id}.png"
                abs_image_path = img_dir / image_filename
                abs_image_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(abs_image_path)
                rel_image_path = f"data/chartqapro/images/{image_filename}"
                saved_images[chart_id] = rel_image_path

            question = normalize_text_field(ex.get(question_col, ""))
            answer = normalize_text_field(ex.get(answer_col, ""))

            meta = {}
            for key in meta_keys:
                if key == "Paragraph":
                    continue
                value = ex.get(key)
                try:
                    json.dumps(value)
                    meta[key] = value
                except TypeError:
                    meta[key] = str(value)

            record = {
                "id": f"chartqapro_{args.split}_{num_written:06d}",
                "dataset": "chartqapro",
                "split": args.split,
                "image_paths": [rel_image_path],
                "question": question,
                "answer": answer,
                "meta": meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Wrote {num_written} examples to {ann_path}")
    print(f"Saved {len(saved_images)} unique charts under {img_dir}")


if __name__ == "__main__":
    main()
