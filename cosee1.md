
---

**Prompt 给 Codex：**

You are helping me modify a Python script in my project.

The relevant file is:

* `scripts/export_chartqapro_subset.py`

This script currently exports ChartQAPro examples to JSONL with a unified schema:

```python
{
  "id": str,
  "dataset": "chartqapro",
  "split": str,
  "image_paths": List[str],
  "question": str,
  "answer": str,
  "meta": dict,
}
```

Right now, two things are wrong in the exported JSONL:

1. The `question` and `answer` fields are often stringified Python lists instead of plain strings.
   For example, I see records like:

   ```json
   "question": "['after which year from the x-axis labels, the hour prices in canada first exceeded new zealand?']",
   "answer": "['2017']"
   ```

   But I want:

   ```json
   "question": "after which year from the x-axis labels, the hour prices in canada first exceeded new zealand?",
   "answer": "2017"
   ```

2. The `meta` dict always contains `"Paragraph": ""` for ChartQAPro, and this field is not useful in the HF version I’m using.
   I want to **remove `"Paragraph"` from the exported meta entirely** (i.e., the `meta` dict should not contain this key at all).

Please update `scripts/export_chartqapro_subset.py` to fix both issues.

### Requirements

1. **Normalize question/answer to plain text**

Add a small helper function inside `export_chartqapro_subset.py`, e.g.:

```python
from typing import Any

def normalize_text_field(value: Any) -> str:
    """
    Normalize a Question/Answer field from the raw ChartQAPro example
    into a clean plain-text string.

    - If value is a list/tuple:
        - if empty -> ""
        - if length 1 -> that single element
        - else -> join elements with a single space
    - Otherwise: cast to string
    - Strip surrounding whitespace
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

    # Convert to string and trim whitespace
    text = str(text).strip()
    return text
```

Then, where you currently extract the raw question/answer from the HF example (something like `ex["Question"]`, `ex["Answer"]`, or similar), update that code to use this helper:

```python
raw_question = ex.get("Question") or ex.get("question")
raw_answer = ex.get("Answer") or ex.get("answer")

question_str = normalize_text_field(raw_question)
answer_str = normalize_text_field(raw_answer)
```

Use `question_str` and `answer_str` in the exported record, e.g.:

```python
record = {
    "id": f"chartqapro_{args.split}_{global_idx:06d}",
    "dataset": "chartqapro",
    "split": args.split,
    "image_paths": [local_image_path],
    "question": question_str,
    "answer": answer_str,
    "meta": {
        "Question Type": ex.get("Question Type", ""),
        "Year": ex.get("Year", []),
        # No "Paragraph" field here anymore
    },
}
```

Make sure you do **not** re-wrap `question_str` or `answer_str` with `str()` in a way that would reintroduce `"['...']"` artifacts.

2. **Remove `"Paragraph"` from meta**

Right now the exporter likely does something like:

```python
"meta": {
    "Question Type": ex.get("Question Type", ""),
    "Year": ex.get("Year", []),
    "Paragraph": ex.get("Paragraph", ""),
},
```

Update this so that `"Paragraph"` is no longer included at all.
The final `meta` dict should only contain the keys that are genuinely useful, e.g.:

```python
"meta": {
    "Question Type": ex.get("Question Type", ""),
    "Year": ex.get("Year", []),
}
```

Do not include `"Paragraph"` even if it exists in the raw example.

### After the change

After you update the script, the following command should produce clean output:

```bash
python -m scripts.export_chartqapro_subset \
  --split test \
  --max-examples 10 \
  --seed 42 \
  --out-dir data/chartqapro_test
```

In `data/chartqapro_test/annotations/chartqapro_1k.jsonl` or similar, I should see JSON records where:

* `question` and `answer` are plain text strings, with no `['...']` artifacts.
* `meta` has `"Question Type"` and `"Year"`, but **no `"Paragraph"` field**.

Please implement these changes in `scripts/export_chartqapro_subset.py` only, keeping the rest of the exporting logic intact (image download, ID format, etc.).

---