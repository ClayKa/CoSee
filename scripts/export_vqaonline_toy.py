from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "ChongyanChen/VQAonline"
RAW_DIR = ROOT / "data" / "hf_raw" / "vqaonline"

OUT_VQAONLINE_IMG_DIR = ROOT / "data" / "vqaonline" / "images"
OUT_VQAONLINE_ANN_DIR = ROOT / "data" / "vqaonline" / "annotations"
OUT_VQAONLINE_JSONL = OUT_VQAONLINE_ANN_DIR / "vqaonline.jsonl"

SPLIT_FOR_EXPORT = "test"
MAX_EXAMPLES = 200


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VQAONLINE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VQAONLINE_ANN_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        cache_dir=str(RAW_DIR),
        allow_patterns=[f"{SPLIT_FOR_EXPORT}.json"],
    )
    json_path = Path(snapshot_path) / f"{SPLIT_FOR_EXPORT}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Could not find JSON file at {json_path}")

    print(f"Loading JSON from {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if MAX_EXAMPLES is not None:
        data = data[:MAX_EXAMPLES]

    print(f"Exporting {len(data)} examples from split '{SPLIT_FOR_EXPORT}'")

    img_cache: Dict[str, str] = {}  # image_name -> rel path
    exported = 0
    for idx, entry in enumerate(data):
        image_name = entry.get("image")
        if not image_name:
            print(f"[WARN] Missing image name for idx={idx}; skipping.")
            continue

        if image_name in img_cache:
            rel_image_path = img_cache[image_name]
        else:
            src_path = None
            found_folder = None
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
                    found_folder = f"images{folder_idx}"
                    break
                except Exception:
                    continue

            if src_path is None:
                print(f"[WARN] Could not find image for {image_name} in images1..images7; skipping.")
                continue

            dst_path = OUT_VQAONLINE_IMG_DIR / image_name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
            rel_image_path = str(dst_path.relative_to(ROOT))
            img_cache[image_name] = rel_image_path

        record = {
            "id": f"vqaonline_{SPLIT_FOR_EXPORT}_{idx:06d}",
            "dataset": "vqaonline",
            "split": SPLIT_FOR_EXPORT,
            "image_paths": [rel_image_path],
            "question": entry.get("question", "").strip(),
            "answer": entry.get("answer", "").strip(),
            "meta": {
                "context": entry.get("context", "").strip(),
                "topic": entry.get("topic", None),
                "url": entry.get("url", None),
                "source_split": SPLIT_FOR_EXPORT,
                "raw_image_name": image_name,
            },
        }

        OUT_VQAONLINE_JSONL.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if exported == 0 else "a"
        with OUT_VQAONLINE_JSONL.open(mode, encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        exported += 1

    print(f"Exported {exported} examples to {OUT_VQAONLINE_JSONL}")
    print(f"Saved {len(img_cache)} unique images under {OUT_VQAONLINE_IMG_DIR}")


if __name__ == "__main__":
    main()
