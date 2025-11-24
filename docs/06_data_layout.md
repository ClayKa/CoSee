# Data layout and annotation schema

We maintain a unified annotation schema for all toy datasets used by CoSee.
Each dataset has its own folder under `data/`:

- `data/slidevqa/`
- `data/infochartqa/`
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
  "dataset": "slidevqa",              // one of: "slidevqa", "infochartqa", "vqaonline"
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
  * For **InfoChartQA**, we may either:

    * use only the **infographic chart** (one image per example), or
    * include both the infographic and plain chart as two entries in `image_paths`.
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

* **InfoChartQA**
  The dataset contains paired **plain charts** and **infographic charts** for the same underlying data,
  with visual-element-based questions about titles, legends, annotations, etc.
  In our schema, we:

  * set `image_paths` either to the **infographic chart only**, or to a two-image list
    `[infographic_chart, plain_chart]` depending on the experiment design;
  * keep chart type / difficulty / question type in `meta`.

* **VQAonline**
  VQAonline provides real community questions, images, long answers, and textual context.
  Since not all questions are text-centric, we treat it as an **auxiliary dataset** and may create
  a small text-focused subset. In our schema:

  * `image_paths` contains the screenshot or image;
  * `question` and `answer` are taken from the original VQA pair;
  * any additional context or metadata is stored in `meta` (e.g., thread ID, text-detection flags).
