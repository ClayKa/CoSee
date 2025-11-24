from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

HF_CACHE_DIR = ROOT / "data" / "hf_cache" / "chartqapro"
OUTPUT_IMAGES_DIR = ROOT / "data" / "chartqapro" / "images"
OUTPUT_ANN_DIR = ROOT / "data" / "chartqapro" / "annotations"
OUTPUT_ANN_PATH = OUTPUT_ANN_DIR / "chartqapro.jsonl"

# Configure this manually as needed
MAX_EXAMPLES = 200  # toy subset size; can be changed


def _choose_split(ds_dict) -> str:
    """Pick a split from a DatasetDict with a reasonable priority."""
    split_priority = ["test", "validation", "dev", "train"]
    for s in split_priority:
        if s in ds_dict:
            return s
    return next(iter(ds_dict.keys()))


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


def main() -> None:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading ChartQAPro from cache dir: {HF_CACHE_DIR}")
    ds_dict = load_dataset(
        "ahmed-masry/ChartQAPro",
        cache_dir=str(HF_CACHE_DIR),
    )

    split = _choose_split(ds_dict)
    ds = ds_dict[split]
    print(f"Using split='{split}' with {len(ds)} examples.")
    print("Columns:", ds.column_names)

    image_col = _detect_column(ds.column_names, ["image", "chart", "figure"], "image")
    question_col = _detect_column(
        ds.column_names, ["question", "question_text", "query"], "question"
    )
    answer_col = _detect_column(
        ds.column_names, ["answer", "answer_text", "label"], "answer"
    )

    meta_keys = [
        c for c in ds.column_names if c not in {image_col, question_col, answer_col}
    ]

    saved_charts: Dict[str, str] = {}  # chart_id -> relative image path

    def infer_chart_id(example: Dict[str, Any], idx: int) -> str:
        """
        Infer a stable chart identifier from the example.
        If we find a suitable ID-like column, we reuse it.
        Otherwise, fallback to a synthetic f"chart_{idx:06d}".
        """
        for key in [
            "image_id",
            "chart_id",
            "figure_id",
            "img_id",
            "imgname",
            "img_name",
            "name",
            "id",
        ]:
            if key in example and example[key] is not None:
                value = str(example[key])
                value = value.replace("/", "_").replace("\\", "_").strip()
                if value:
                    return value
        return f"chart_{idx:06d}"

    num_written = 0
    num_unique_charts = 0

    with OUTPUT_ANN_PATH.open("w", encoding="utf-8") as fout:
        total = min(len(ds), MAX_EXAMPLES)
        for idx in range(total):
            example = ds[idx]

            chart_id = infer_chart_id(example, idx)

            if chart_id in saved_charts:
                rel_image_path = saved_charts[chart_id]
            else:
                img = example[image_col]
                if not isinstance(img, Image.Image):
                    if isinstance(img, (bytes, bytearray)):
                        img = Image.open(BytesIO(img)).convert("RGB")
                    else:
                        raise TypeError(
                            f"Expected PIL.Image.Image for column '{image_col}', got type={type(img)}"
                        )

                image_filename = f"{chart_id}.png"
                abs_image_path = OUTPUT_IMAGES_DIR / image_filename
                abs_image_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(abs_image_path)
                rel_image_path = f"data/chartqapro/images/{image_filename}"
                saved_charts[chart_id] = rel_image_path
                num_unique_charts += 1

            question = str(example[question_col])
            answer = str(example[answer_col])

            meta: Dict[str, Any] = {}
            for key in meta_keys:
                value = example.get(key)
                try:
                    json.dumps(value)
                    meta[key] = value
                except TypeError:
                    meta[key] = str(value)

            record = {
                "id": f"chartqapro_{split}_{idx:06d}",
                "dataset": "chartqapro",
                "split": split,
                "image_paths": [rel_image_path],
                "question": question,
                "answer": answer,
                "meta": meta,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Exported {num_written} examples to {OUTPUT_ANN_PATH}")
    print(f"Saved {num_unique_charts} unique charts under {OUTPUT_IMAGES_DIR}")


if __name__ == "__main__":
    main()
