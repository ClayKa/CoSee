# 01 – CoSee Design Specification

This document specifies the **software architecture** of CoSee. It is intended as the primary reference for implementing the core system in this repository (for both humans and code-assist tools).

The focus here is **what modules exist, what responsibilities they have, and how they interact**, not low-level implementation details. Exact class and function signatures are further detailed in `02_api_spec.md`.

---

## 1. Architectural overview

CoSee is structured as a small, modular system around four core components:

1. **Model wrapper (`QwenVLClient`)**  
   A thin wrapper around the Qwen3-VL-4B-Instruct model and its processor.  
   It is responsible for:
   - Loading the model and processor from a **local path**.
   - Converting `(images, question, board_text, role_prompt)` into the **messages** format expected by Qwen3-VL.
   - Calling `processor.apply_chat_template(...)` and `model.generate(...)`.
   - Returning decoded text outputs.

2. **Board (shared visual workspace)**  
   A structured in-memory representation of collaborative notes:
   - Stores **cells** containing localized visual observations, hypotheses, and links.
   - Each cell is tied to a **view** (page/region) and has metadata: tags, author, step.
   - Provides `to_text()` to render an interpretable, length-controlled summary for the agents.

3. **Agents (role-specialized workers)**  
   Each agent is a lightweight wrapper around the same underlying VLM (via `QwenVLClient`), but:
   - Has its own **role prompt**.
   - Reads the current Board summary and relevant images.
   - Emits a **structured action** (JSON-like dict) describing what it did.

4. **Controller (`CoSeeController`)**  
   Orchestrates multi-round collaboration:
   - Initializes the Board.
   - At each step, calls all agents in parallel or sequence.
   - Applies their actions to update the Board.
   - Decides when to stop and how to form the final answer.

Surrounding these core components are **dataset loaders**, **metrics**, and **experiment scripts**, which are described at the end.

---

## 2. Data model

### 2.1 Sample

A **sample** is a single question-answer instance from a dataset:

- `images`: ordered list of page images (e.g., PIL images or paths).
- `question`: natural language question string.
- `answer`: ground-truth answer string (or list of acceptable variants).
- `meta`: dataset-specific metadata (IDs, source, etc.).

Datasets (e.g., MP-DocVQA, SlideVQA) will normalize their raw format into this common structure.

### 2.2 View

A **view** describes where a Board cell is grounded in the visual input:

- `page`: 1-based page index into `images`.
- `bbox` (optional): `[x_min, y_min, x_max, y_max]` in normalized or pixel coordinates.  
  For now, this can be optional and may be `None` if the note refers to the entire page.
- `description` (optional): short human-readable description, e.g. `"upper-right table"`.

Views are primarily used for **interpretability** and optional visualizations; they are not required for the model to function (the model sees the full images), but they are useful for logging and future extensions.

### 2.3 Board cell

A **cell** is a single item on the Board (one line in the shared visual whiteboard). It has:

- `id`: integer, unique within a Board.
- `view`: a `View` struct or dict describing where this note lives.
- `content`: short text content (observation, hypothesis, correction).
- `tags`: list of string tags, e.g. `["title"]`, `["date", "footnote"]`.
- `author`: agent name string (e.g., `"scanner"`, `"detail_reader"`).
- `step`: integer step index in the collaboration process.

Cells are append-only in the simplest implementation: revising an earlier cell can be done by adding a new cell of type “correction” referring to the old cell ID.

### 2.4 Board

The **Board** is a collection of cells plus helper methods:

- Internal state:
  - `cells`: list of all cells (in creation order).
  - Optional indices by `page`, `author`, `tags` for efficient queries.

- Core operations:
  - `add_cell(...)` → returns new cell ID.
  - `get_cells(...)` → simple filters (by page, author, tags).
  - `to_text(max_cells_per_page, max_total_chars)` → renders a concise text summary.

Text rendering should:

- Group cells by page and step.
- Truncate older or less relevant cells if exceeding `max_cells_per_page` or `max_total_chars`.
- Produce a stable, readable format like:

  ```text
  [Page 1]
  - (#3, scanner, step 1) Title: "Quarterly Sales Report Q2"
  - (#7, detail_reader, step 2) The total revenue is $3.4M.

  [Page 2]
  - (#5, scanner, step 1) Contains a summary table of regions.
  ```

This representation is fed into the VLM as part of the input text for each agent.

### 2.5 Action

Agents interact with the Board by emitting **actions**. An action is a JSON-like dict with fields such as:

* `action`:

  * `"INSPECT"` – add new observation based on one or more pages/regions.
  * `"LINK"` – link an existing cell to additional evidence.
  * `"HYPOTHESIZE"` – propose an answer + supporting cell IDs.
  * `"REVISE"` – correct or refine prior notes or hypotheses.

* `view`: (optional) a `View` indicating where the observation comes from.

* `target_cell_id`: (optional) ID of a cell being revised or linked.

* `content`: short text describing what the agent found or proposes.

* `tags`: list of tags for the resulting cell.

* `confidence`: optional numeric or qualitative confidence indicator (e.g., `"high"`, `"medium"`).

The `CoSeeController` is responsible for:

* Validating the action structure.
* Updating the Board accordingly.
* Tracking hypotheses and deciding when a final answer is ready.

---

## 3. QwenVLClient: Model wrapper design

File: `cosee/models/qwen_vl_wrapper.py`

### 3.1 Responsibilities

`QwenVLClient` is the only place that:

* Knows how to load Qwen3-VL-4B-Instruct and its `AutoProcessor` from a given `model_path`.
* Knows how to build the **messages** structure expected by Qwen3-VL for multimodal chat.
* Calls `processor.apply_chat_template(...)` and `model.generate(...)`.
* Decodes output IDs back into text.

All higher-level modules should treat it as a simple “black-box” text generator conditioned on images, question, Board summary, and role prompt.

### 3.2 Inputs and outputs

**Inputs:**

* `images`: list of PIL images (or image-like objects) to be passed to Qwen3-VL.
* `question`: question text.
* `board_text`: optional Board summary text.
* `role_prompt`: optional role description for the agent (e.g., “You are a global scanner…”).
* `gen_kwargs`: generation hyperparameters (e.g., `max_new_tokens`, `temperature`, `top_p`).

**Outputs:**

* `raw_output`: full decoded model output (often including dialog headers like `user` / `assistant`).
* `answer_text`: postprocessed string that keeps only the relevant portion (e.g., the assistant’s final answer or JSON).

### 3.3 Message construction

`QwenVLClient` will construct `messages` with the following pattern:

* Single `"user"` turn, with content being a sequence of blocks:

  1. One or more `"image"` blocks:

     ```python
     {"type": "image", "image": image_obj}
     ```

  2. A `"text"` block with the question:

     ```python
     {"type": "text", "text": f"Question: {question}"}
     ```

  3. Optional `"text"` block with the Board summary:

     ```python
     {"type": "text", "text": f"Shared board (summary):\n{board_text}"}
     ```

  4. Optional `"text"` block with the role prompt:

     ```python
     {"type": "text", "text": role_prompt}
     ```

`messages` is then passed into:

```python
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
```

and moved to the model device.

### 3.4 Token and compute control

`QwenVLClient` should **not** enforce a global budget policy itself, but it should:

* Accept `max_new_tokens` as an explicit parameter (default small, e.g., 64).
* Allow the caller to control `temperature`, `top_p`, etc.
* Optionally return rough token counts (e.g., via `inputs["input_ids"].shape[1]`, `len(output_ids[0])`).

Global budgeting (per-question token tracking) can be implemented at the controller or experiment layer, using this information.

---

## 4. Board: shared workspace design

File: `cosee/board.py`

### 4.1 Responsibilities

The Board must:

* Provide a **simple, immutable record** of what each agent has written and when.
* Support efficient creation of new cells.
* Support compact, controllable text rendering for use in prompts.
* Allow simple introspection for analysis (e.g., coverage, redundancy, corrections).

### 4.2 Core operations

Minimal required methods:

* `add_cell(view, content, tags, author, step) -> int`

  * Creates a new cell and returns its ID.
  * Internally increments a counter to ensure unique IDs.

* `list_cells(filter_fn=None) -> list[Cell]`

  * Return all cells, optionally filtered by a predicate.

* `to_text(max_cells_per_page=8, max_total_chars=2000) -> str`

  * Converts the Board into a textual summary:

    * Group by page number.
    * Within each page, sort by `step` then cell ID.
    * Truncate per-page count and total characters.
    * Produce a human-readable format suitable for a prompt.

Optional methods (for later analysis):

* `get_cells_by_page(page)`, `get_cells_by_author(author)`, `get_cells_by_tags(tags)`.

### 4.3 Design constraints

* **Deterministic ordering**: the textual representation should be stable and predictable for the same Board state.
* **Length control**: `to_text` must enforce an upper bound on length to avoid blowing up the prompt.
* **Simplicity**: no complex caching or indexing is required initially; a simple list-based implementation is fine.

---

## 5. Agents: role specialization and actions

File: `cosee/agents.py`

### 5.1 Responsibilities

Agents are the main behavioral units that:

* Interpret the question and Board summary.
* Decide what to look at (which pages/regions).
* Emit **actions** that update the Board and/or propose answers.

They are all powered by the same `QwenVLClient` instance but differ in their:

* **Role prompt**.
* **Strategy** (implicitly guided by the role prompt and examples).
* **Default generation settings** (e.g., slightly different temperatures).

### 5.2 Base Agent interface

A base `Agent` class should define:

* `name`: unique identifier (e.g., `"scanner"`, `"detail_reader"`).
* `role_prompt`: fixed prompt describing its role.
* `qwen_client`: instance of `QwenVLClient`.

Core method:

* `act(images, question, board: Board, gen_kwargs) -> dict`:

  * Calls `board.to_text()` to get a Board summary.
  * Calls `qwen_client.generate(...)` with images, question, board summary, and role prompt.
  * Parses the model output into an action dict following the Action schema.

Parsing can be done via:

* Strict JSON output (prompted as “respond with a JSON dict”), or
* Heuristic parsing (less preferred) if JSON is imperfect.

### 5.3 Example roles

Initial set of roles (subject to refinement):

* **Scanner**:

  * Focus: global overview, page-level structure.
  * Action patterns: INSPECT on multiple pages, tags like `"section"`, `"table"`, `"figure"`.

* **Detail Reader**:

  * Focus: local details, small text, numbers.
  * Action patterns: INSPECT targeted views (“top-right table on page 2”), tags like `"number"`, `"field"`.

* **Cross-Checker**:

  * Focus: verifying candidate answers and resolving conflicts.
  * Action patterns: HYPOTHESIZE and REVISE, referencing existing cell IDs and supporting evidence.

All agent implementations should be thin wrappers over the base `Agent`, differing mostly in `role_prompt` and maybe very small behavior tweaks.

---

## 6. CoSeeController: multi-round orchestration

File: `cosee/controller.py`

### 6.1 Responsibilities

The `CoSeeController` manages the *lifecycle of a single QA instance*:

1. Initialize an empty Board.
2. For up to `T` steps:

   * Render Board summary.
   * Call each agent’s `act(...)` (could be sequential or parallel).
   * Parse and validate their actions.
   * Update the Board accordingly.
   * Track any `HYPOTHESIZE` actions and their supporting cells.
   * Decide whether to stop early (e.g., if one or more high-confidence hypotheses exist).
3. Produce the final answer and return both the answer and the Board trace.

### 6.2 Control flow pseudocode

At a high level:

```python
def run(images, question):
    board = Board()
    hypotheses = []

    for step in range(1, max_steps + 1):
        for agent in agents:
            action = agent.act(images, question, board, gen_kwargs)
            validated_action = validate_action(action)
            apply_action_to_board(validated_action, board, step)
            if validated_action["action"] == "HYPOTHESIZE":
                hypotheses.append(validated_action)

        if should_stop(hypotheses, step, board):
            break

    final_answer = decide_final_answer(hypotheses, board)
    return {
        "answer": final_answer,
        "board": board,
        "trace": None or some logging structure,
    }
```

### 6.3 Stopping and decision policies

The controller should implement simple but explicit policies:

* **Early stopping** (`should_stop`) could be based on:

  * Presence of at least one high-confidence hypothesis.
  * Number of hypotheses reaching agreement (e.g., multiple agents suggesting the same answer).
  * Maximum step `T` reached (fall back to best available guess).

* **Final answer decision** (`decide_final_answer`) could:

  * Pick the highest-confidence hypothesis.
  * Break ties by majority vote over suggested answers.
  * Optionally aggregate evidence: append a short “because” explanation derived from supporting cells (for analysis, not necessarily for evaluation).

### 6.4 Error handling

Actions may be malformed or incomplete. The controller should:

* Validate actions before applying them:

  * Ensure `action` field is known.
  * Check required fields for each action type.
* If parsing fails or action is invalid:

  * Log the error.
  * Optionally add a special error cell to the Board (for debugging).
  * Proceed without crashing the entire run.

---

## 7. Dataset loaders and evaluation

### 7.1 Dataset loaders

Files: `cosee/datasets/mpdocvqa.py`, `cosee/datasets/slidevqa.py`, etc.

Responsibilities:

* Load raw data (files, JSON annotations, etc.).

* Convert each instance to the common `Sample` structure:

  ```python
  {
      "images": [...],
      "question": "...",
      "answer": "...",
      "meta": {...}
  }
  ```

* Provide simple iteration interfaces (e.g., Python generators, PyTorch datasets).

### 7.2 Evaluation utilities

Files: `cosee/eval/metrics.py`, `cosee/eval/baselines.py`.

Responsibilities:

* **Metrics**:

  * Implement ANLS, EM, F1.
  * Provide helpers to accumulate metrics over a dataset.

* **Baselines**:

  * Single-agent CoT:

    * One call to QwenVLClient with a CoT-style prompt.
  * Self-consistency:

    * N independent runs with random sampling, majority vote on final answers.
  * Multi-agent chat (no Board):

    * Agents talk via text turns only, no structured Board.

The evaluation layer should be independent from the core CoSee modules: it calls either baselines or `CoSeeController` and measures performance.

---

## 8. Configuration and logging

### 8.1 Configuration

Experiments should be configurable via:

* YAML/JSON config files, or
* Command-line arguments + small config objects.

Key parameters:

* `M`: number of agents.
* `T`: maximum collaboration rounds.
* Generation parameters: `max_new_tokens`, `temperature`, `top_p`.
* Board rendering parameters: `max_cells_per_page`, `max_total_chars`.
* Dataset-specific paths and splits.

### 8.2 Logging

For each run (per sample), we may log:

* The final answer and whether it was correct.
* Token counts (inputs and outputs).
* Board snapshot (cells) at the end of the run.
* Optional step-wise traces for analysis (especially for qualitative case studies).

Logging should be:

* Lightweight (e.g., JSONL lines per sample).
* Optional and controllable (to avoid excessive disk usage).

---

## 9. Implementation priorities

In development, modules should be implemented in roughly this order:

1. `QwenVLClient` (basic loading + single-image QA demo).
2. `Board` (data model + `to_text`).
3. Base `Agent` + `DummyAgent`.
4. `CoSeeController` skeleton wired with `DummyAgent`.
5. Real agents using `QwenVLClient`.
6. Dataset loaders for dev mini-sets.
7. Baselines and evaluation scripts.
8. Full dataset runners and logging.

This document should be kept in sync with the implementation to ensure CoSee remains small, modular, and aligned with the project’s compute-conscious, multi-agent, visual Board-centric design.