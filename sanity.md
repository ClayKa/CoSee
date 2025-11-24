### Task: Temporary data sanity-check script for CoSee datasets

We are building a project called **CoSee** with the following local data layout:

```text
data/
  slidevqa/
    annotations/
      slidevqa.jsonl
    images/
      ...
  infochartqa/
    annotations/
      infochartqa.jsonl
    images/
      ...
  vqaonline/
    annotations/
      vqaonline.jsonl
    images/
      ...
```

Each `*.jsonl` file will contain **one JSON object per line**, following a unified schema:

```jsonc
{
  "id": "slidevqa_toy_0001",
  "dataset": "slidevqa",            // "slidevqa" | "infochartqa" | "vqaonline"
  "split": "toy",                   // "train" | "val" | "test" | "toy"
  "image_paths": [
    "data/slidevqa/images/deck_0001/page_0001.png",
    "data/slidevqa/images/deck_0001/page_0002.png"
  ],
  "question": "What is the main topic of this slide deck?",
  "answer": "Neural machine translation",
  "meta": {                         // optional extra fields
    "source_id": "SlideVQA_QA_0001",
    "answer_type": "span"
  }
}
```

We want a **temporary sanity-check script** that:

1. Reads a small number of examples from each JSONL file.
2. Prints basic fields (`id`, `question`, `answer`, `image_paths`).
3. Checks whether each `image_paths` entry exists on disk.
4. Tries to open each image once with PIL to verify it is not corrupted.
5. Reports warnings/errors but does not crash the entire script on a single bad example.

---

### Implementation requirements

Create a new file:

```text
scripts/tmp_check_data_paths.py
```

Implement it with the following details.

#### 1. Imports and constants

* Use `pathlib.Path` for file paths.
* Use Python standard `json` module to parse JSONL lines.
* Use `PIL.Image` to verify images.

The top of the file should look like:

```python
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
```

#### 2. `check_dataset` function

Implement a function that:

* Takes a dataset name and annotation path.
* Checks if the annotation file exists; if not, prints a warning and returns.
* Iterates over the JSONL file line by line.
* Skips blank lines.
* Parses JSON; if parsing fails, prints an error and stops for that dataset.
* For each parsed example, extracts:

  * `id` (default `"<no id>"` if missing)
  * `image_paths` (default `[]`)
  * `question` and `answer` (default `""`)
* Prints each example in a readable format.
* For each image path:

  * Converts it to an absolute path using `ROOT / relative_path`.
  * Checks existence (`Path.exists()`).
  * Prints `"OK"` or `"MISSING"` next to it.
  * If it exists, tries to open it with `Image.open` and call `im.verify()` inside a `with` block.
  * Catches any exceptions from `Image.open` / `verify()` and prints an error line.

Code sketch (Codex should fill in exact details, but preserve structure and messages):

```python
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
```

#### 3. `main` function and CLI entrypoint

Add a `main()` that loops over the `DATASETS` list and calls `check_dataset` for each one:

```python
def main():
    for name, ann_path in DATASETS:
        check_dataset(name, ann_path)


if __name__ == "__main__":
    main()
```

This should allow running from the repo root as:

```bash
python -m scripts.tmp_check_data_paths
```

---

### How this script will be used

1. We will manually put **1–3 example lines** into each JSONL file:

   * `data/slidevqa/annotations/slidevqa.jsonl`
   * `data/infochartqa/annotations/infochartqa.jsonl`
   * `data/vqaonline/annotations/vqaonline.jsonl`

   Each line must follow the unified schema described above (`id`, `dataset`, `split`, `image_paths`, `question`, `answer`, optional `meta`).

2. We run:

   ```bash
   python -m scripts.tmp_check_data_paths
   ```

3. The script should:

   * Print several examples per dataset.
   * For each image path, print whether the file exists and whether it can be opened.
   * Print warnings for missing annotation files, empty JSONL files, missing `image_paths`, JSON decode errors, or image open errors.

4. We will use this script only as a **sanity check** before implementing the formal dataset loader (`cosee/data/datasets.py`). It doesn’t need to be production-ready, but it must be robust enough not to crash on the first bad example and give clear diagnostic messages in the console.
