
---

### Task: Implement `run_cosee_on_dataset.py` – multi-agent CoSee runner with multiple configurations

Please add a new script to the repo:

* `scripts/run_cosee_on_dataset.py`

This script should mirror the behavior of `scripts/run_qwen_single_baseline.py`, but instead of calling a single Qwen model directly, it should:

* Construct a **CoSeeController** with one or more **QwenAgent** (and possibly other agents).
* Run multi-step CoSee episodes per example.
* Evaluate final answers with the **same normalization and accuracy metrics** as the baseline.
* Optionally support several **agent configurations** via a CLI flag so we can compare different multi-agent setups.

#### 1. CLI interface

Create a `main()` with `argparse` and expose it as a module:

```bash
python -m scripts.run_cosee_on_dataset \
  --dataset slidevqa \
  --split train \
  --max-examples 2 \
  --agent-config two_qwen \
  --max-steps 3 \
  --device cpu
```

Required / suggested CLI arguments:

* `--dataset`

  * Type: `str`
  * Choices: `slidevqa`, `chartqapro`, `vqaonline`
* `--split`

  * Type: `str`
  * For now: accept whatever `load_toy_split(dataset=..., split=...)` supports.
  * E.g., `train` for slidevqa, `test` for chartqapro and vqaonline.
* `--max-examples`

  * Type: `int`, default e.g. `50`
  * For local sanity tests we will use very small values (1–2).
* `--model-path`

  * Type: `str`, default `./models/Qwen3-VL-4B-Instruct`
* `--device`

  * Type: `str`, default `"cpu"`
  * Should accept `"cpu"` and `"cuda"` (we will later use GPU servers).
* `--max-new-tokens`

  * Type: `int`, default e.g. `64`
  * Passed into Qwen generation.
* `--max-steps`

  * Type: `int`, default `3` or `4`
  * Maximum number of controller steps per episode.
* `--max-images-per-example`

  * Type: `int`, default `None`
  * If set, cap the number of images per `Example` (e.g., for SlideVQA decks).
* `--agent-config`

  * Type: `str`, default `"two_qwen"`
  * Choices:

    * `"single_qwen_board"` – single QwenAgent using the board.
    * `"two_qwen"` – Qwen scanner + Qwen cross-checker.
    * `"three_qwen"` – Qwen scanner + Qwen detail-reader + Qwen cross-checker.
* `--output`

  * Type: `str`, optional
  * Path for the JSONL results file; if not provided, default to something like:

    * `results/cosee_{agent_config}_{dataset}_{split}.jsonl`

---

#### 2. Dataset loading and image handling

Reuse the existing dataset loader so we stay aligned with the baseline:

* Import `load_toy_split` from `cosee.data.datasets`.

In `main()`:

```python
from pathlib import Path
from cosee.data.datasets import load_toy_split
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
```

* Load examples:

```python
examples = load_toy_split(
    dataset=args.dataset,
    split=args.split,
    max_examples=args.max_examples,
)
print(f"Loaded {len(examples)} examples from {args.dataset}/{args.split}")
```

* For each `example`:

  * Start from `example.image_paths`.

  * If `args.max_images_per_example` is not `None`, slice:

    ```python
    image_paths = example.image_paths
    if args.max_images_per_example is not None:
        image_paths = image_paths[: args.max_images_per_example]
    ```

  * Load images:

    ```python
    images = [
        Image.open(ROOT / p).convert("RGB")
        for p in image_paths
    ]
    ```

---

#### 3. Qwen client and CoSee components

Reuse the existing CoSee building blocks:

* `QwenVLClient` from `cosee.models.qwen_vl_wrapper`
* `Board`, `View` from `cosee.board`
* `Agent`, `QwenAgent`, `DummyAgent`, `Action` from `cosee.agents`
* `CoSeeController` from `cosee.controller`

In `main()`:

1. Initialize a **single** `QwenVLClient`:

   ```python
   from cosee.models.qwen_vl_wrapper import QwenVLClient

   qwen_client = QwenVLClient(
       model_path=args.model_path,
       device=args.device,
       dtype="auto",  # or keep consistent with existing usage
   )
   ```

2. Implement a helper function to **instantiate agents + controller** based on `args.agent_config`:

   ```python
   def build_controller(qwen_client: QwenVLClient, max_steps: int, config_name: str) -> CoSeeController:
       # returns a CoSeeController with appropriate agents
   ```

---

#### 4. Agent configurations

We want at least three configurations to experiment with.
All Qwen-based agents should **share the same** `qwen_client` instance.

Use short, role-specific prompts. Below are suggested configurations:

##### 4.1 `single_qwen_board`

* Agents:

  * `QwenAgent(name="QwenSingle", role_prompt=..., allow_final_answer=True, final_answer_step=1 or 2)`

* Behavior:

  * Single QwenAgent both writes notes and eventually produces the final answer.
  * It still uses the `Board` to store its own notes, so we can inspect intermediate reasoning.

* Suggested `role_prompt`:

  ```text
  You are a careful multimodal reasoner. You can add short observations to the shared board first, then give a single concise final answer to the question. Use the board to keep track of important findings from the images and avoid repeating yourself.
  ```

* Implementation hint:

  * Use `board_max_cells_per_page` and `board_max_chars` defaults consistent with existing `QwenAgent`.
  * For example:

    ```python
    qwen_single = QwenAgent(
        name="QwenSingle",
        role_prompt=ROLE_PROMPT_SINGLE,
        qwen_client=qwen_client,
        allow_final_answer=True,
        final_answer_step=1,  # or 2 if you want at least one note before answering
        default_gen_kwargs={"max_new_tokens": args.max_new_tokens},
    )
    controller = CoSeeController(agents=[qwen_single], max_steps=args.max_steps)
    ```

##### 4.2 `two_qwen` (scanner + cross-checker)

* Agents:

  1. `QwenScanner` (note-only)

     * `allow_final_answer=False`
     * Role: scan the images and write 1–2 useful observations to the board.
  2. `QwenCrossChecker` (final answer)

     * `allow_final_answer=True`
     * `final_answer_step=0` or `1` (it can answer whenever it gets a turn)
     * Role: read the board and produce a concise final answer, grounded in board notes.

* Suggested prompts:

  * Scanner:

    ```text
    You are a scanning agent. Quickly skim the images and write 1–2 concise observations that may be useful for answering the question. Do not answer the question. Focus on key text, numbers, and visual structure.
    ```

  * Cross-checker:

    ```text
    You are a cross-checking agent. Read the shared board notes and the images, then give a single concise final answer to the question. Use the board as your evidence; do not restate all notes, just answer.
    ```

* Example construction:

  ```python
  scanner = QwenAgent(
      name="QwenScanner",
      role_prompt=ROLE_PROMPT_SCANNER,
      qwen_client=qwen_client,
      allow_final_answer=False,
      default_gen_kwargs={"max_new_tokens": args.max_new_tokens},
  )

  cross_checker = QwenAgent(
      name="QwenCrossChecker",
      role_prompt=ROLE_PROMPT_CROSSCHECKER,
      qwen_client=qwen_client,
      allow_final_answer=True,
      final_answer_step=0,
      default_gen_kwargs={"max_new_tokens": args.max_new_tokens},
  )

  controller = CoSeeController(
      agents=[scanner, cross_checker],
      max_steps=args.max_steps,
  )
  ```

* The controller will alternate between agents in round-robin; the scanner will write notes when called, and the cross-checker will output a final answer when allowed.

##### 4.3 `three_qwen` (scanner + detail-reader + cross-checker)

* Agents:

  1. `QwenScanner` – as above, broad overview notes.
  2. `QwenDetailReader` – focuses on more fine-grained reading (numbers, legends, table entries).
  3. `QwenCrossChecker` – final answer based on accumulated board notes.

* Suggested prompts:

  * Detail reader:

    ```text
    You are a detail-reading agent. Focus on reading fine-grained text, numbers, labels, and legends that are directly relevant to the question. Add precise observations to the shared board. Do not answer the question directly.
    ```

* Construction example:

  ```python
  scanner = QwenAgent(
      name="QwenScanner",
      role_prompt=ROLE_PROMPT_SCANNER,
      qwen_client=qwen_client,
      allow_final_answer=False,
      default_gen_kwargs={"max_new_tokens": args.max_new_tokens},
  )

  detail_reader = QwenAgent(
      name="QwenDetailReader",
      role_prompt=ROLE_PROMPT_DETAIL,
      qwen_client=qwen_client,
      allow_final_answer=False,
      default_gen_kwargs={"max_new_tokens": args.max_new_tokens},
  )

  cross_checker = QwenAgent(
      name="QwenCrossChecker",
      role_prompt=ROLE_PROMPT_CROSSCHECKER,
      qwen_client=qwen_client,
      allow_final_answer=True,
      final_answer_step=1,  # only answer on its turn, after at least one scanner/detail round
      default_gen_kwargs={"max_new_tokens": args.max_new_tokens},
  )

  controller = CoSeeController(
      agents=[scanner, detail_reader, cross_checker],
      max_steps=args.max_steps,
  )
  ```

We can extend to more configs later; for now these three are enough for early experiments.

---

#### 5. Episode loop, normalization, and metrics

We want the CoSee runner to **reuse the same normalization and scoring logic** as `run_qwen_single_baseline.py`.

* If convenient, factor the normalization helpers out into a small module, e.g., `cosee/eval_utils.py`, and import them from both scripts.
* If refactoring is too invasive for now, you can temporarily duplicate the same normalization functions in the new script, but please keep logic identical.

Expected helpers (or equivalent):

* `normalize_answer(text: str) -> str`

  * Lowercase
  * Strip surrounding whitespace
  * Remove punctuation and simple currency symbols
  * Collapse multiple spaces
* A small function to compute `correct_exact` and `correct_loose`:

  ```python
  gold_norm = normalize_answer(example.answer)
  pred_norm = normalize_answer(final_answer)
  correct_exact = (pred_norm == gold_norm)
  correct_loose = correct_exact  # or extend later if we adopt a looser rule
  ```

Main evaluation loop:

```python
results = []
num_exact = 0
num_loose = 0

for idx, example in enumerate(examples):
    # 1) Load images
    image_paths = example.image_paths
    if args.max_images_per_example is not None:
        image_paths = image_paths[: args.max_images_per_example]
    images = [Image.open(ROOT / p).convert("RGB") for p in image_paths]

    # 2) Build a fresh controller and board for this example
    controller = build_controller(qwen_client, args.max_steps, args.agent_config)

    # 3) Run CoSee episode
    final_answer, final_board = controller.run(images, example.question)

    # 4) Summarize board
    board_summary = final_board.to_text(
        max_cells_per_page=8,
        max_total_chars=1500,
    )

    # 5) Normalize and score
    gold_answer = example.answer
    gold_norm = normalize_answer(gold_answer)
    pred_norm = normalize_answer(final_answer or "")

    correct_exact = (pred_norm == gold_norm)
    correct_loose = correct_exact  # can be extended later

    if correct_exact:
        num_exact += 1
    if correct_loose:
        num_loose += 1

    # 6) Per-example logging
    print(
        f"[{idx+1}/{len(examples)}] id={example.id} "
        f"exact={int(correct_exact)} loose={int(correct_loose)}",
        flush=True,
    )

    # 7) Collect result object
    results.append({
        "id": example.id,
        "dataset": example.dataset,  # or args.dataset
        "split": example.split,      # if available in Example
        "question": example.question,
        "gold_answer": gold_answer,
        "pred_answer": final_answer,
        "gold_norm": gold_norm,
        "pred_norm": pred_norm,
        "correct_exact": bool(correct_exact),
        "correct_loose": bool(correct_loose),
        "agent_config": args.agent_config,
        "max_steps": args.max_steps,
        "num_images": len(image_paths),
        "board_summary": board_summary,
    })
```

---

#### 6. Writing results and summary

At the end of `main()`:

1. Decide the output path:

   ```python
   results_dir = ROOT / "results"
   results_dir.mkdir(parents=True, exist_ok=True)

   if args.output:
       out_path = Path(args.output)
   else:
       out_name = f"cosee_{args.agent_config}_{args.dataset}_{args.split}.jsonl"
       out_path = results_dir / out_name
   ```

2. Write JSONL results:

   ```python
   import json

   with out_path.open("w", encoding="utf-8") as f:
       for obj in results:
           f.write(json.dumps(obj, ensure_ascii=False) + "\n")
           f.flush()
   ```

3. Print summary metrics:

   ```python
   total = len(results)
   print(f"Finished {total} examples from {args.dataset}/{args.split} with config={args.agent_config}.")
   if total > 0:
       exact_acc = num_exact / total
       loose_acc = num_loose / total
       print(f"Exact match accuracy: {exact_acc:.3f} ({num_exact} / {total})")
       print(f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})")
   ```

---

#### 7. Sanity test usage

Once implemented, we should be able to run small CPU-based sanity checks like:

```bash
# SlideVQA (cap images per example for local testing)
python -m scripts.run_cosee_on_dataset \
  --dataset slidevqa \
  --split train \
  --max-examples 1 \
  --max-images-per-example 4 \
  --agent-config two_qwen \
  --max-steps 3 \
  --device cpu

# ChartQAPro
python -m scripts.run_cosee_on_dataset \
  --dataset chartqapro \
  --split test \
  --max-examples 1 \
  --agent-config three_qwen \
  --max-steps 3 \
  --device cpu

# VQAonline (short-answer subset handled inside load_toy_split)
python -m scripts.run_cosee_on_dataset \
  --dataset vqaonline \
  --split test \
  --max-examples 2 \
  --agent-config single_qwen_board \
  --max-steps 3 \
  --device cpu
```

These runs do not need to give good accuracy yet; they are just to verify:

* CoSee episodes run end-to-end on real examples.
* Final answers and board summaries are written to JSONL.
* The same normalization and metrics as the baseline are applied.

---