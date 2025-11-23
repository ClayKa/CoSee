# 05 – To-Do List and Sprints

This document defines the **working plan** for CoSee as a sequence of short sprints and concrete tasks.  
It is written so that both a human developer and a code-assist tool can use it as the **authoritative TODO list**.

Time horizon: from **now** until the **ACL 2026** submission deadline.  
Priority: keep the scope feasible for a single author with limited compute.

---

## 1. Principles

- **Short, focused sprints**: each sprint has a clear theme and a small set of deliverables.
- **Code first, then experiments**: get end-to-end CoSee running on a small dev subset as early as possible.
- **Compute-conscious**: always keep token budgets and GPU usage in mind.
- **Paper aware**: design experiments and logging so that tables/figures can be generated with minimal extra work.

Each sprint below has:

- A **goal**,
- A list of **core tasks** (blocking),
- A list of **optional/bonus tasks** (nice to have if time permits).

Checkboxes (`[ ]`) are for manual tracking; they are initially all unchecked.

---

## 2. Sprint 0 – Repository & Environment (Very Short)

**Goal:** Make the repo and environment ready so that CoSee code and experiments can be developed smoothly.

### 2.1 Core tasks

- [ ] Initialize repository structure:
  - [ ] Create `cosee/`, `cosee/models/`, `cosee/datasets/`, `cosee/eval/`, `scripts/`, `docs/`.
  - [ ] Add `__init__.py` files to make `cosee` a Python package.
- [ ] Place design documents:
  - [ ] Add `docs/00_overview.md`, `01_design_cosee.md`, `02_api_spec.md`, `03_experiments_plan.md`, `04_dev_env_setup.md`, `05_todo_and_sprints.md`.
- [ ] Set up local Python environment (macOS):
  - [ ] Create and activate virtual env / conda env.
  - [ ] Install dependencies (`torch` CPU, `transformers>=4.57.0`, `accelerate`, `safetensors`, `pillow`, `opencv-python`, `datasets`, `numpy`, `tqdm`, `huggingface_hub`).
- [ ] Download Qwen3-VL-4B-Instruct locally:
  - [ ] Use `huggingface-cli` or equivalent to populate `models/Qwen3-VL-4B-Instruct/`.
  - [ ] Run a simple CPU test to confirm the model and processor load correctly.
- [ ] Prepare GPU server:
  - [ ] Create and activate a `cosee` environment.
  - [ ] Install GPU-enabled PyTorch (matching CUDA), plus the same core dependencies.
  - [ ] Upload the local Qwen3-VL model directory to the server (e.g., `/data/models/Qwen3-VL-4B-Instruct`) and verify loading.

### 2.2 Optional tasks

- [ ] Generate `requirements.txt` and/or `environment.yml` from the working environment.
- [ ] Set up a simple logging directory structure (e.g., `logs/mpdocvqa/`, `logs/slidevqa/`).

---

## 3. Sprint 1 – Core Infrastructure (Qwen wrapper, Board, Dummy Agents)

**Goal:** Implement the minimal code needed to run a **toy CoSee episode** end-to-end (with dummy agents) on local CPU.

### 3.1 Core tasks

#### 3.1.1 QwenVLClient

- [ ] Implement `cosee/models/qwen_vl_wrapper.py` with the `QwenVLClient` API:
  - [ ] `__init__(model_path, device, dtype, load_in_8bit, load_in_4bit)`
  - [ ] `build_messages(images, question, board_text, role_prompt)`
  - [ ] `generate(..., max_new_tokens, temperature, top_p, top_k, repetition_penalty, presence_penalty, ...)`
  - [ ] `tokenize_prompt_length(messages)`
- [ ] Verify a simple multimodal demo:
  - [ ] Single image + question → reasonable caption or answer.
  - [ ] Confirm `return_full_text=False` returns only the assistant answer segment.

#### 3.1.2 Board

- [ ] Implement `cosee/board.py`:
  - [ ] Internal representation of `Cell` and `View`.
  - [ ] `add_cell(view, content, tags, author, step) -> CellID`
  - [ ] `get_cell`, `list_cells`, `get_cells_by_page`, `get_cells_by_author`, `get_cells_by_tags`.
  - [ ] `to_text(max_cells_per_page, max_total_chars)` with deterministic formatting and length control.
- [ ] Create a small unit-style test script:
  - [ ] Add several cells across multiple pages.
  - [ ] Inspect `to_text()` output and confirm grouping, ordering, and truncation behave as expected.

#### 3.1.3 Base Agent & DummyAgent

- [ ] Implement `cosee/agents.py`:
  - [ ] Base `Agent` class with `name`, `role_prompt`, `qwen_client`, and abstract `act(...)`.
  - [ ] A `DummyAgent` that ignores the model and returns a fixed `Action` (e.g., an `INSPECT` action with dummy content).
- [ ] Write a small script that:
  - [ ] Creates a `Board`,
  - [ ] Runs a few steps with `DummyAgent`,
  - [ ] Prints the Board summary.

#### 3.1.4 CoSeeController skeleton

- [ ] Implement `cosee/controller.py` with:
  - [ ] Constructor accepting `agents`, `max_steps`, `default_gen_kwargs`.
  - [ ] `validate_action(action)`.
  - [ ] `apply_action(board, action, step, author)`.
  - [ ] `should_stop(step, hypotheses, board)`.
  - [ ] `decide_final_answer(hypotheses, board, fallback_answer)`.
  - [ ] `run(images, question, per_agent_gen_kwargs=None)`.
- [ ] Connect `CoSeeController` with `DummyAgent`:
  - [ ] Run a toy episode using a small set of dummy images and a question.
  - [ ] Confirm the flow: controller → agent → action → Board update → controller.

### 3.2 Optional tasks

- [ ] Add simple logging inside the controller (`print` or minimal logging library) for debugging.
- [ ] Define a minimal `trace` structure returned by `run(...)` (e.g., list of per-step actions).

---

## 4. Sprint 2 – Real Agents, Dev Subsets, and Baselines

**Goal:** Replace dummy behavior with real Qwen-based agents, set up dataset loaders, and implement baseline methods.  
End state: run CoSee and baselines on **small dev subsets** of target datasets.

### 4.1 Core tasks

#### 4.1.1 Real agents

- [ ] Define role prompts for:
  - [ ] `ScannerAgent` (global structure and page relevance),
  - [ ] `DetailReaderAgent` (local fine-grained reading),
  - [ ] `CrossCheckerAgent` (verification and hypothesis checking).
- [ ] Implement concrete agent classes in `cosee/agents.py`:
  - [ ] Subclass from `Agent`, set `name` and `role_prompt`.
  - [ ] Implement `act(...)` to:
    - [ ] Obtain `board_text = board.to_text(...)`.
    - [ ] Call `qwen_client.generate(...)` with images, question, board_text, role_prompt.
    - [ ] Parse model output into an `Action` dict (prefer JSON format in prompts).
- [ ] Test each agent individually on a couple of images + a toy Board summary.

#### 4.1.2 Dataset loaders and dev subsets

- [ ] Implement dataset loaders in `cosee/datasets/`:
  - [ ] `mpdocvqa.py` with `load_mpdocvqa_split(split)` and `build_dev_subset(split, max_samples)`.
  - [ ] `slidevqa.py` with analogous APIs.
- [ ] Ensure each loader returns a standardized `Sample`:
  - [ ] `images`, `question`, `answer`, `meta`.
- [ ] Build small dev subsets:
  - [ ] For MP-DocVQA (e.g., 50–200 samples),
  - [ ] For SlideVQA (e.g., 50–200 samples).

#### 4.1.3 Metrics and baselines

- [ ] Implement `cosee/eval/metrics.py`:
  - [ ] `normalize_answer`, `exact_match`, `f1_score`, `anls`, `aggregate_metrics`.
- [ ] Implement baselines in `cosee/eval/baselines.py`:
  - [ ] `single_agent_cot(...)`.
  - [ ] `self_consistency(...)`.
  - [ ] `multi_agent_chat_no_board(...)`.
- [ ] Write dev scripts:
  - [ ] `scripts/run_mpdocvqa_baselines.py` that:
    - [ ] Loads the dev subset,
    - [ ] Runs baselines,
    - [ ] Prints metrics and average token usage.
  - [ ] `scripts/run_mpdocvqa_cosee_dev.py` that:
    - [ ] Instantiates CoSee agents and controller,
    - [ ] Runs on the same dev subset,
    - [ ] Logs metrics and Board summaries.

### 4.2 Optional tasks

- [ ] Add simple progress bars and ETA (`tqdm`) to the dev scripts.
- [ ] Log per-sample results in JSONL format (predictions, gold answers, token counts, method tags).

---

## 5. Sprint 3 – Full Experiments and Ablations

**Goal:** Run CoSee and baselines on **full evaluation splits** for MP-DocVQA and SlideVQA, then perform key ablations.  
End state: have all numbers required for the main tables in the ACL paper.

### 5.1 Core tasks

#### 5.1.1 Full evaluation runs

- [ ] Implement `scripts/run_mpdocvqa_cosee.py`:
  - [ ] Load full MP-DocVQA evaluation split.
  - [ ] Use final CoSee configuration(s) selected from dev runs.
  - [ ] Run CoSee on all samples, log:
    - [ ] Predictions,
    - [ ] Metrics per sample,
    - [ ] Token usage per sample,
    - [ ] Board summaries (optionally truncated).
- [ ] Implement `scripts/run_mpdocvqa_baselines_full.py` for baselines:
  - [ ] Single-Agent CoT,
  - [ ] Self-Consistency,
  - [ ] Multi-Agent Chat.
- [ ] Repeat analogous scripts and runs for **SlideVQA**.

#### 5.1.2 Ablations

- [ ] Number of agents:
  - [ ] Run CoSee variants with 1, 2, 3 agents on one dataset (at least dev or a subset of eval).
- [ ] Steps `T`:
  - [ ] Compare `T = 1`, `T = 2`, and if feasible `T = 3`.
- [ ] Board vs. no Board:
  - [ ] Compare standard CoSee to a “no Board” variant where agents only see chat history.
- [ ] Token budgets:
  - [ ] Vary `max_new_tokens` and Board caps and measure accuracy vs token usage.

For each ablation run, log:

- [ ] Per-sample predictions and metrics.
- [ ] Aggregate results in easily loadable JSON/CSV.

### 5.2 Optional tasks

- [ ] Implement a small analysis script to:
  - [ ] Merge logs from multiple runs,
  - [ ] Produce summary CSVs and Markdown-ready tables for the paper.

---

## 6. Sprint 4 – Analysis, Figures, and Paper Integration

**Goal:** Turn raw experimental logs into **stories, tables, and figures** suitable for the ACL paper.

### 6.1 Core tasks

- [ ] Implement analysis scripts in `scripts/` (or `cosee/analysis/`):
  - [ ] Compute final dataset-level metrics for each method and export as CSV.
  - [ ] Compute efficiency statistics (tokens, latency) for each method.
  - [ ] Extract collaboration metrics from CoSee Board traces (coverage, redundancy, corrections).
- [ ] Design and generate plots:
  - [ ] Accuracy vs token usage for CoSee vs baselines.
  - [ ] Ablation plots (e.g., number of agents, steps `T`).
  - [ ] Qualitative visualization of Board traces for selected examples.
- [ ] Select case studies:
  - [ ] Good CoSee examples where collaboration clearly helps.
  - [ ] Failure cases where CoSee overcomplicates or miscoordinated reasoning.

### 6.2 Optional tasks

- [ ] Implement a simple HTML or notebook viewer to inspect Board traces interactively.
- [ ] Add minimal unit tests or regression tests for critical components (Board formatting, metrics).

---

## 7. Sprint 5 – Paper Finalization and Release Preparation

**Goal:** Integrate experimental results into the ACL paper, ensure coherence between code, logs, and text, and prepare for a minimal release.

### 7.1 Core tasks

- [ ] Update the ACL paper draft:
  - [ ] Fill in all quantitative tables with final numbers.
  - [ ] Add or adjust figures (architecture, collaboration examples, ablation plots).
  - [ ] Ensure the narrative matches the actual experiments and findings.
- [ ] Double-check consistency:
  - [ ] Verify that numbers in the text, tables, and logs match.
  - [ ] Confirm that all methods and settings mentioned in the paper exist in code and configs.
- [ ] Prepare minimal code release:
  - [ ] Clean up the repository (remove unused files, temporary scripts).
  - [ ] Write a concise `README.md` explaining:
    - [ ] Setup,
    - [ ] Running CoSee on dev subsets,
    - [ ] Reproducing main results (at least on a small sample).
  - [ ] Optionally include `configs/` for key experiments.

### 7.2 Optional tasks

- [ ] Prepare a short “quickstart” notebook demonstrating CoSee on a single example.
- [ ] Add comments or docstrings in key modules (`QwenVLClient`, `Board`, `Agent`, `CoSeeController`) for future readers.

---

## 8. How code-assist tools should use this file

When assisting with development in this repository, a code-assist tool should:

1. **Respect the sprint ordering**  
   - Prioritize tasks in the **current sprint** (starting from Sprint 1),  
     unless explicitly told otherwise.
2. **Use the design docs as constraints**  
   - Consult `00_overview.md`, `01_design_cosee.md`, and `02_api_spec.md`  
     before designing new classes or changing interfaces.
3. **Treat checkboxes as progress markers**  
   - Assume all tasks are initially unchecked; the human developer will mark them manually.
4. **Avoid scope creep**  
   - Do not introduce training pipelines, new datasets, or unrelated features that contradict
     the compute-conscious, small-scope goals described here and in `00_overview.md`.

This file is the **authoritative TODO and sprint plan** for CoSee. Any substantial change in direction should update this document accordingly.
