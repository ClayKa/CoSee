# Data layout and annotation schema

We maintain a unified annotation schema for all toy datasets used by CoSee.
Each dataset has its own folder under `data/`:

- `data/slidevqa/`
- `data/chartqapro/`
- `data/vqaonline/`

Inside each dataset folder, we use the following structure:

- `images/` — all image files (possibly multiple images per example)
- `annotations/` — JSONL files with normalized annotations used by CoSee

## Unified `Example` schema

All annotation files under `annotations/` are stored as **JSON Lines** (`*.jsonl`).
Each line corresponds to one example with the following fields:

```jsonc
{
  "id": "slidevqa_train_000001",
  "dataset": "slidevqa",              // one of: "slidevqa", "chartqapro", "vqaonline"
  "split": "train",                   // "train" | "val" | "test" | "toy"
  "image_paths": [
    "data/slidevqa/images/deck_0001/page_0001.png",
    "data/slidevqa/images/deck_0001/page_0002.png"
  ],
  "question": "What is the main topic of this slide deck?",
  "answer": "Neural machine translation",
  "meta": {                           // optional dataset-specific metadata
    "source_id": "qa_id_1234",
    "evidence_pages": [1, 2, 3],
    "answer_type": "span"
  }
}
```

Conventions:

* `id` is a globally unique identifier within the project (we typically prefix with the dataset and split).
* `dataset` indicates the original dataset (“slidevqa”, “infochartqa”, or “vqaonline”).
* `split` indicates which split we are using (“train”, “val”, “test”), or `"toy"` for small debug subsets.
* `image_paths` is a **list of relative paths** (from the repository root) to one or more image files.

  * For **SlideVQA**, this will usually contain **multiple pages** of a slide deck.
  * For **ChartQAPro**, this will contain a single chart image per example.
  * For **VQAonline**, we currently plan to use **one image per example**.
* `question` is the natural language question we pass to the agents.
* `answer` is the canonical ground-truth answer string used for evaluation (after any normalization).
* `meta` is an optional dictionary allowing us to keep dataset-specific fields (e.g., SlideVQA’s
  `qa_id`, `reasoning_type`, `answer_type`, or InfoChartQA’s chart type) without affecting the
  core CoSee pipeline.

## Per-dataset notes

* **SlideVQA**
  The original annotations contain fields such as `deck_name`, `qa_id`, `question`, `answer`,
  `arithmetic_expression`, `evidence_pages`, `reasoning_type`, and `answer_type`.
  In our normalized schema, we keep:

  * `question` and `answer` directly;
  * `image_paths` as the list of local slide images for that deck;
  * additional fields like `qa_id`, `evidence_pages`, `answer_type` in `meta`.

* **ChartQAPro**
  Each example has a chart image and a question/answer pair (often with metadata such as chart type,
  source, or question type). In our schema:

  * `image_paths` contains one entry: the local chart image.
  * `question` / `answer` are taken directly from the dataset.
  * Additional fields (e.g., question_type, chart_type, source, task_type) are stored in `meta`.

### VQAonline

- Raw source: `ChongyanChen/VQAonline` (HF dataset repo).
- Raw JSON files downloaded to: `data/hf_raw/vqaonline/.../{train,val,test}.json`
- Unified images: `data/vqaonline/images/*.png`
- Unified annotations: `data/vqaonline/annotations/vqaonline.jsonl`

Each JSONL line has:

```json
{
  "id": "vqaonline_<split>_<6-digit index>",
  "dataset": "vqaonline",
  "split": "<train|val|test>",
  "image_paths": ["data/vqaonline/images/<image_name>.png"],
  "question": "<string>",
  "answer": "<string>",
  "meta": {
    "context": "<string>",
    "topic": "<string or null>",
    "url": "<string or null>",
    "source_split": "<original split name>",
    "raw_image_name": "<original PNG name>"
  }
}
```

* **VQAonline**
  VQAonline provides real community questions, images, long answers, and textual context.
  Since not all questions are text-centric, we treat it as an **auxiliary dataset** and may create
  a small text-focused subset. In our schema:

  * `image_paths` contains the screenshot or image;
  * `question` and `answer` are taken from the original VQA pair;
  * any additional context or metadata is stored in `meta` (e.g., thread ID, text-detection flags).

#### VQAonline: short-answer subset for quantitative evaluation

The raw VQAonline dataset contains many open-ended, StackExchange-style questions where:

- The gold answer is a long, free-form explanation (often multiple sentences).
- Answers frequently include URLs and other metadata (e.g., references, code snippets).

Since our primary evaluation protocol is based on normalized string matching (exact / loose match), and CoSee is focused on **short-answer document and UI understanding**, we define a filtered short-answer subset of VQAonline:

- We **exclude** examples where the gold answer:
  - Contains `http` or `www` (i.e., URLs).
  - Contains more than a small number of newline characters (multi-paragraph answers).
  - Exceeds a predefined length threshold (e.g., > N characters or > M tokens; the concrete threshold is configurable in `cosee/data/datasets.py`).
- We **retain** examples where:
  - The gold answer is a relatively short phrase, number, or short sentence.
  - The answer can reasonably be compared via normalized string matching.

We use this short-answer subset for all **quantitative metrics** on VQAonline.  
The full, unfiltered dataset may still be used for **qualitative case studies**, especially to analyze CoSee’s behavior on realistic, text-heavy web screenshots.

#### ChartQAPro: chart reasoning characteristics and evaluation

ChartQAPro is only available with a `test` split. We treat it as an evaluation-only chart reasoning benchmark and export a toy subset into:

- `data/chartqapro/images/…` – individual chart images.
- `data/chartqapro/annotations/chartqapro.jsonl` – unified examples.

Key properties:

- Source columns in the HF dataset include `Question`, `Answer`, `Question Type`, `image`, `Year`, and `Paragraph`.
- Many questions require:
  - Interpreting axes and legends.
  - Identifying when a curve crosses a given threshold (e.g., “when does wind capacity first exceed 100 GW?”).
  - Estimating or comparing numerical values from trends.

For evaluation:

- We map `Answer` fields to short normalized strings (e.g., `2037-38` → `2037 38`) and use the same exact / loose match protocol as other datasets.
- This means that any numerical misestimation (e.g., predicting `2024-25` instead of `2037-38`) is counted as an error, even if the predicted range is semantically plausible.
- In our experiments, we expect the vanilla Qwen3-VL-4B baseline to perform relatively poorly on ChartQAPro, which makes it a good stress test for whether CoSee’s multi-step, board-based reading can help the model focus on the relevant regions of the chart.
