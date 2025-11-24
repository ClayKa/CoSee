# 03 – Experiments Plan

This document specifies the experimental plan for CoSee, including datasets, baselines, configurations, metrics, and analysis.  
It is intended to guide both implementation and paper-writing, while keeping experiments **compute-conscious** and feasible for a single researcher.

---

## 1. Objectives

The experiments for CoSee aim to answer the following questions:

1. **Answer quality under small compute budgets**  
   - Does CoSee (multi-agent + shared Board) improve DocVQA and slide-based VQA performance compared to:
     - Single-agent chain-of-thought (CoT),
     - Self-consistency (answer-only ensembling),
     - Multi-agent chat without a Board,
   - when all methods are constrained to similar **token budgets** and use the same **4B-scale VLM**?

2. **Collaboration patterns and interpretability**  
   - How do different agents (Scanner, Detail Reader, Cross-Checker) use the Board?
   - Do CoSee runs exhibit:
     - Better **evidence coverage** (more relevant pages/regions),
     - Less **redundancy** (fewer duplicated notes),
     - More **corrections** where one agent fixes another’s mistake?

3. **Compute-efficiency trade-offs**  
   - For a fixed quality level, does CoSee require fewer tokens than naive self-consistency?
   - How do the number of agents and collaboration rounds affect:
     - Accuracy,
     - Token usage,
     - Wall-clock latency?

The experiments are therefore structured around **(quality, efficiency, behavior)**.

---

## 2. Datasets

We target two main benchmarks:

1. **MP-DocVQA**  
   - Multi-page document QA with scanned documents or rendered PDFs.
   - Primary metric: **ANLS** (Average Normalized Levenshtein Similarity).

2. **SlideVQA**  
   - Multi-slide question answering on slide decks or presentation-like documents.
   - Primary metrics: **EM** (Exact Match) and **F1**.

### 2.1 Data splits and dev subsets

For each dataset, we define:

- **Full evaluation split**  
  - MP-DocVQA: official validation or test split (depending on availability).  
  - SlideVQA: official validation or test split.

- **Dev subset** (for fast iteration)  
  - Small held-out subset from the training/validation data:
    - Size: e.g., 50–200 samples per dataset (configurable).
    - Sampling: random or stratified by question type / difficulty.
  - Used for:
    - Prompt engineering,
    - Agent/Board configuration tuning,
    - Sanity checks.

Implementation-wise:

- `build_dev_subset(split="train", max_samples=N)` returns a list of normalized `Sample` objects for quick experiments.
- Final reported results will be computed on full validation/test splits with **fixed configurations** determined on the dev subsets.

---

## 3. Backbone model and agents

### 3.1 Backbone VLM

- Qwen3-VL-4B-Instruct as the **only** backbone model.
- All agents share the same model weights and processor via `QwenVLClient`.
- Model is loaded from a **local path** (no online fetching during experiments).

### 3.2 Agent set

We start with **three agents**, all wrapping the same `QwenVLClient` but with different role prompts:

1. **ScannerAgent**
   - Role: global overview and structure discovery.
   - Behavior:
     - Identify which pages are relevant to the question.
     - Write high-level notes about content (titles, headers, table presence).

2. **DetailReaderAgent**
   - Role: local precision and fine-grained reading.
   - Behavior:
     - Focus on specific pages/regions identified by Scanner or prior Board cells.
     - Extract key numbers, names, and phrases.

3. **CrossCheckerAgent**
   - Role: verification and conflict resolution.
   - Behavior:
     - Revisit pages/regions supporting candidate answers.
     - Check consistency across multiple notes.
     - Propose final hypotheses or corrections.

Later ablations may:

- Vary the number of agents (1, 2, 3),
- Drop specific roles to see their contribution.

---

## 4. Baselines

We implement several baselines using the same backbone model (Qwen3-VL-4B) to ensure fair comparison:

1. **Single-Agent CoT (CoT-Single)**  
   - One agent (generic “DocVQA expert”) receives all images and the question.
   - Prompt encourages step-by-step reasoning and then a final answer.
   - No explicit Board or structured memory.

2. **Self-Consistency (CoT-SC)**  
   - Use the same single-agent CoT prompt.
   - Run the model `N` times with sampling enabled (e.g., `N = 3`).
   - Aggregate final answers by majority vote (or tie-breaking by simple heuristics).
   - Compute budget matched to CoSee by adjusting `N` and `max_new_tokens`.

3. **Multi-Agent Chat without Board (MA-Chat)**  
   - Instantiate the same roles as CoSee (Scanner, Detail Reader, Cross-Checker).
   - Agents talk in a **text-only chat history** shared by all.
   - No explicit Board or structured cells; all memory is unstructured conversation.
   - Run for `T` turns and pick the final answer from the last agent or via a simple aggregator.

4. **Direct Answer (No CoT)** (optional)  
   - Single Qwen3-VL call with a short prompt (no explicit reasoning request).
   - Used as a lower-bound baseline, mainly for context.

Each baseline is configured to **roughly match token usage** with CoSee, to the extent possible.

---

## 5. CoSee configurations

### 5.1 Controller settings

Key parameters:

- `M`: number of agents (default 3).
- `T`: maximum collaboration steps/rounds (default 2 or 3).
- `max_new_tokens` per agent action (e.g., 64).
- Board summary limits:
  - `max_cells_per_page`: e.g., 6–8.
  - `max_total_chars`: e.g., 1500–2000 characters.

We will define and evaluate a few standard CoSee configurations:

1. **CoSee-Base**
   - Agents: Scanner, Detail Reader, Cross-Checker.
   - `T = 2` steps.
   - Moderate Board limits.
   - Conservative `max_new_tokens` (e.g., 64).

2. **CoSee-Light**
   - Agents: Scanner, Detail Reader.
   - `T = 2` steps.
   - Tighter Board / token limits.
   - Designed for lower compute settings.

3. **CoSee-Expanded** (optional)
   - `T = 3` steps.
   - Slightly larger Board and `max_new_tokens`.
   - Used to see whether extra collaboration helps.

### 5.2 Generation settings

For all methods (CoSee and baselines), generation hyperparameters should be consistent and documented:

- Default:
  - `temperature = 0.2`
  - `top_p = 0.8`
  - `top_k = 20`
  - `repetition_penalty = 1.0`
  - `presence_penalty = 1.5`
- Self-consistency may use slightly higher temperature to encourage diversity (e.g., `0.7`), but within a controlled range.
- Random seeds should be fixed per run for reproducibility.

---

## 6. Metrics and logging

### 6.1 Primary answer metrics

For each dataset:

- **MP-DocVQA**:
  - ANLS (Average Normalized Levenshtein Similarity),
  - Optionally EM/F1 for reference.

- **SlideVQA**:
  - EM,
  - F1 (token-level, after normalization),
  - Optionally ANLS.

We will report:

- Per-dataset metrics for:
  - CoSee variants,
  - Each baseline.

### 6.2 Efficiency metrics

To highlight compute-conscious design, we also track:

- **Token usage per question**:
  - Input tokens (`prompt_tokens`),
  - Output tokens (`completion_tokens`),
  - Total tokens (`total_tokens`).

- **Wall-clock latency per question** (optional):
  - Measured on the target GPU (e.g., RTX 4090),
  - For CoSee and baselines.

These metrics should be logged per sample and summarized as:

- Mean / median tokens,
- Mean / median latency (if measured).

### 6.3 Collaboration metrics (CoSee only)

Using Board traces and hypotheses, we can compute:

- **Coverage**:
  - Number of unique pages touched by Board cells.
  - Fraction of pages with at least one note.

- **Redundancy**:
  - Number of notes referring to the same or near-identical content.
  - Approximate via heuristic matching of cell content.

- **Corrections**:
  - Count of `REVISE` actions.
  - Fraction of questions where at least one correction occurred before the final hypothesis.

These metrics help quantify **how agents collaborate** beyond raw answer accuracy.

### 6.4 Logging format

For each sample, we log a JSONL record containing:

- Metadata:
  - Dataset, split, sample ID.
- Model/method:
  - `method`: `"cosee_base"`, `"cot_single"`, `"cot_sc"`, `"ma_chat"`, etc.
- Input summary:
  - Number of pages, question text (optionally truncated).
- Output:
  - Final prediction,
  - Gold answer(s).
- Metrics:
  - EM, F1, ANLS for that sample (if applicable).
- Efficiency:
  - Token counts,
  - Latency (if measured).
- CoSee-specific:
  - Final Board snapshot (cells),
  - Hypotheses list (actions),
  - Optional step-level trace.

---

## 7. Experimental procedures

### 7.1 Dev-time procedure

1. **Dev subset selection**  
   - For each dataset, choose a small dev subset (e.g., 50–200 samples).

2. **Prompt & role tuning**  
   - Design role prompts for Scanner, Detail Reader, Cross-Checker.
   - Iterate on:
     - Action schema formatting (JSON structure),
     - Board summary format (`Board.to_text`),
     - Stopping criteria.

3. **Configuration sweeps (small)**  
   - On dev subsets, run small sweeps over:
     - Number of agents (e.g., 2 vs 3),
     - `T` (2 vs 3 steps),
     - Board limits (e.g., `max_total_chars` 1000 vs 2000),
     - `max_new_tokens` per action (e.g., 48 vs 64).

4. **Selection of final configs**  
   - Pick 1–2 CoSee configurations that:
     - Are stable (few parsing errors),
     - Fit within token budgets,
     - Show clear improvements over baselines on dev subsets.

### 7.2 Final evaluation

1. **Freeze configurations**  
   - Fix:
     - Role prompts,
     - Controller hyperparameters (`M`, `T`, thresholds),
     - Generation settings,
     - Board parameters.

2. **Run baselines**  
   - Run each baseline (CoT-Single, CoT-SC, MA-Chat, Direct Answer) on full evaluation splits.
   - Log predictions, metrics, and token usage.

3. **Run CoSee**  
   - Run CoSee configurations on the same splits.
   - Ensure each run uses:
     - Same random seed strategy,
     - Same hardware configuration,
     - Comparable compute budgets where applicable.

4. **Aggregate and compare**  
   - Compute dataset-level metrics:
     - ANLS / EM / F1,
     - Token statistics.
   - Build tables comparing:
     - Baselines vs CoSee,
     - Different CoSee variants.

### 7.3 Qualitative analysis

Select a small set of **representative examples** for qualitative study:

- Cases where CoSee improves over baselines:
  - Multi-page aggregation,
  - Resolving conflicting signals,
  - Correcting initial wrong guesses.
- Cases where CoSee fails or adds noise:
  - Overcomplicated reasoning,
  - Board clutter without added value,
  - Mis-coordination between agents.

For each such example:

- Render Board traces in a human-readable format (could be used directly in the paper).
- Describe the collaboration dynamics and what went right/wrong.

---

## 8. Ablation studies

We plan several ablations to isolate design choices:

1. **Number of agents**
   - Compare:
     - Single-agent CoSee (one agent + Board),
     - Two-agent CoSee (e.g., Scanner + Detail Reader),
     - Three-agent CoSee (full configuration).

2. **Board vs no Board**
   - Compare:
     - CoSee with Board (standard),
     - A variant where agents only see chat history (like MA-Chat),
     - A “Board but single agent” variant (to see if structure alone helps).

3. **Number of collaboration steps `T`**
   - Evaluate:
     - `T = 1` (no collaboration; essentially single-pass),
     - `T = 2`,
     - `T = 3`.
   - Measure how additional rounds affect performance and cost.

4. **Token budget sensitivity**
   - Vary:
     - `max_new_tokens` per agent,
     - Board length caps.
   - Plot accuracy vs average token usage.

5. **Prompt sensitivity (optional)**
   - Swap different variants of role prompts for agents.
   - Check robustness to minor wording changes.

Each ablation should be run on at least:

- The dev subsets (for tuning),
- A smaller fraction of the evaluation split (if full runs are too expensive).

---

## 9. Reproducibility and resource considerations

To keep the project feasible and reproducible:

- **Single-GPU assumption**  
  - Assume experiments run on a single GPU (e.g., RTX 4090, 24 GB VRAM).
  - Avoid methods that require multi-GPU model parallelism.

- **Static model weights**  
  - No training or fine-tuning is performed; model weights remain fixed.
  - All variability comes from prompts, controller logic, and sampling.

- **Fixed random seeds**  
  - Use fixed seeds for:
    - Generation randomness,
    - Sample shuffling (when building dev subsets).
  - Report seeds used in experiment logs.

- **Lightweight logging**  
  - Use JSONL logs with one record per question.
  - Compress logs when storing long Board traces.

- **Released configuration files**  
  - Store experiment configs (YAML/JSON) in a `configs/` directory, including:
    - Dataset paths and splits,
    - CoSee hyperparameters,
    - Baseline hyperparameters.

This plan balances:

- Scientific clarity (clean comparisons and ablations),
- Practical constraints (limited compute, single author),
- And narrative needs for the paper (quantitative tables + qualitative stories grounded in the Board).

# 03.5 - Dataset requirements

For the first-stage toy document / multi-image VQA experiments, the dataset must satisfy the following requirements:

1. The images must primarily consist of **documents, posters, charts/tables, or multi-page screenshots**, rather than purely natural scene images.
2. Each example must contain at least the following fields:
   - One or more **image paths**;
   - A natural-language **question**;
   - A natural-language **answer** (short span or phrase is preferred).
3. The total number of examples for the toy setting should be **controllable in the range of approximately 50–200 samples**.
4. The total dataset size (images + annotations) should be **reasonably small** (e.g., **< 1 GB**) to support local download and server upload under limited network conditions.
5. The dataset’s **license/terms of use must permit research usage**, and there must be no known legal or ethical restrictions that would prevent its use in an academic paper.

# Dataset candidates and roles

For CoSee’s first-stage experiments, we focus on text-rich visual question answering benchmarks released in the last few years. We select one **primary** dataset and two **secondary/auxiliary** datasets, each covering a different type of multi-page or multi-text visual input.

1. **Primary dataset – SlideVQA (multi-page slide decks)**  
   SlideVQA is a multi-image document VQA dataset built from slide decks. Each example consists of a **slide deck composed of multiple slide images** and a **question about the whole deck**. The dataset contains **2.6k+ slide decks, 52k+ slide images, and 14.5k questions**, and many questions require multi-hop reasoning and numerical reasoning across pages.   
   - Role in CoSee: **main toy dataset for “multi-page document VQA”**. It directly matches our setting where multiple agents collaborate over a shared board to scan, note, and cross-check information across several text-dense pages.

2. **Secondary dataset – InfoChartQA (infographic charts)**  
   InfoChartQA is a recent benchmark for multimodal question answering on **infographic charts**. It contains **5,642 pairs of infographic and plain charts**, where each pair shares the same underlying data but differs in visual presentation (plain chart vs. design-heavy infographic).   
   - Role in CoSee: **second dataset focusing on text-rich infographic and chart understanding**. It complements SlideVQA by testing CoSee on dense chart titles, legends, labels, and explanatory text embedded in visually complex infographics.

3. **Auxiliary dataset – VQAonline (authentic community VQA)**  
   VQAonline is an ECCV 2024 dataset built from real questions and images sourced from online Q&A communities (e.g., StackExchange), and is the first VQA dataset whose images, questions, answers, and context all come from an end-to-end authentic use case.   
   - Many images are **screenshots, UI, diagrams, or figures** that do contain text; others are natural images with little or no text. The questions are not always explicitly about reading text, but often involve mixed semantic understanding and context.
   - Role in CoSee: **auxiliary / future extension dataset**. We do not treat VQAonline as a primary “text-only” benchmark, but we may:
     - Use it in discussion to situate CoSee relative to more realistic, heterogeneous VQA settings;  
     - Optionally construct a small **text-focused subset** (e.g., filtering images with detected text and text-centric questions) as a future extension once the core experiments on SlideVQA and InfoChartQA are stable.
