# 02 – CoSee API Specification

This document defines the **concrete Python APIs** for the core CoSee components.  
It is meant to be a working contract for implementation in this repository.

All type annotations use standard Python type hints (Python 3.10+).  
Where necessary, `typing` aliases (e.g., `Protocol`, `TypedDict`) can be introduced in the actual code.

---

## 1. Common types

These are conceptual types used throughout the API.  
They may be implemented as `dataclasses`, `TypedDict`, or simple `dict` structures.

```python
from typing import Any, Dict, List, Optional, Protocol, Tuple

PageIndex = int  # 1-based index into the list of images

BBox = Tuple[float, float, float, float]
# [x_min, y_min, x_max, y_max], either normalized (0–1) or pixel coordinates.

Tag = str
CellID = int
StepIndex = int
AgentName = str
AnswerText = str
````

### 1.1 View

A **view** describes where a note is grounded in the visual input.

```python
class View(Dict[str, Any]):
    """
    Required keys:
        - "page": PageIndex
    Optional keys:
        - "bbox": BBox
        - "description": str
    """
```

Example:

```python
view = {
    "page": 2,
    "bbox": (0.1, 0.2, 0.8, 0.6),
    "description": "top-right table",
}
```

### 1.2 Cell

A **cell** is one entry on the Board.

```python
class Cell(Dict[str, Any]):
    """
    Keys:
        - "id": CellID
        - "view": View
        - "content": str
        - "tags": List[Tag]
        - "author": AgentName
        - "step": StepIndex
    """
```

Example:

```python
cell = {
    "id": 7,
    "view": {"page": 1},
    "content": "Title: 'Quarterly Sales Report Q2'",
    "tags": ["title"],
    "author": "scanner",
    "step": 1,
}
```

### 1.3 Action

An **action** is a structured description of what an agent did in one turn.

```python
class Action(Dict[str, Any]):
    """
    Required keys:
        - "action": str   # "INSPECT" | "LINK" | "HYPOTHESIZE" | "REVISE"
        - "content": str  # textual description, note, hypothesis, etc.

    Optional keys:
        - "view": View
        - "target_cell_id": CellID
        - "tags": List[Tag]
        - "confidence": float | str  # e.g., 0.9 or "high"
        - "answer": str              # for HYPOTHESIZE
        - "supporting_cells": List[CellID]  # for HYPOTHESIZE / REVISE
    """
```

Example (hypothesis):

```python
action = {
    "action": "HYPOTHESIZE",
    "answer": "3.4 million dollars",
    "content": "The total revenue is 3.4M USD based on the summary table.",
    "supporting_cells": [3, 9],
    "tags": ["hypothesis"],
    "confidence": 0.85,
}
```

---

## 2. QwenVLClient API

File: `cosee/models/qwen_vl_wrapper.py`

`QwenVLClient` is the only component that directly interacts with the Qwen3-VL-4B-Instruct model and its processor.

### 2.1 Class interface

```python
from typing import List, Optional, Dict, Any
from PIL import Image

class QwenVLClient:
    """
    Wrapper around Qwen3-VL-4B-Instruct and its AutoProcessor.

    Responsibilities:
    - Load model and processor from a local path.
    - Construct multimodal chat messages from (images, question, board_text, role_prompt).
    - Call processor.apply_chat_template(...) and model.generate(...).
    - Decode outputs and optionally postprocess them.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Initialize the client.

        Args:
            model_path: Local directory containing the Qwen3-VL-4B-Instruct weights
                        and processor config.
            device: Device spec ("cpu", "cuda", "cuda:0", "auto", etc.).
            dtype: Dtype for model loading, e.g., "auto", "bfloat16".
            load_in_8bit: Whether to use 8-bit loading (if supported).
            load_in_4bit: Whether to use 4-bit loading (if supported).
        """

    def build_messages(
        self,
        images: List[Image.Image],
        question: str,
        board_text: Optional[str] = None,
        role_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Construct the chat messages structure expected by Qwen3-VL.

        The default pattern is:
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_1},
                    ...,
                    {"type": "image", "image": img_k},
                    {"type": "text", "text": f"Question: {question}"},
                    {"type": "text", "text": f"Shared board (summary):\\n{board_text}"},
                    {"type": "text", "text": role_prompt},
                ],
            }
        ]

        Args:
            images: List of page images for this round.
            question: Question text.
            board_text: Optional Board summary string.
            role_prompt: Optional role description for the agent.

        Returns:
            A list of messages (single user turn) to be passed to
            processor.apply_chat_template(...).
        """

    def generate(
        self,
        images: List[Image.Image],
        question: str,
        board_text: Optional[str] = None,
        role_prompt: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        return_full_text: bool = False,
        **gen_kwargs: Any,
    ) -> str:
        """
        Run a single multimodal generation call.

        Args:
            images: List of page images.
            question: Question text.
            board_text: Optional Board summary.
            role_prompt: Optional role prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature, top_p, top_k, repetition_penalty, presence_penalty:
                Standard generation hyperparameters.
            return_full_text: If True, return the full decoded chat string
                (including 'user'/'assistant' headers). If False, return only
                the assistant's answer segment.
            gen_kwargs: Extra arguments forwarded to model.generate().

        Returns:
            A text string. By default, this is the assistant's answer segment
            with headers removed.
        """

    def tokenize_prompt_length(
        self,
        messages: List[Dict[str, Any]],
    ) -> int:
        """
        Compute the approximate token length of the input prompt given messages.

        Args:
            messages: Chat messages in the same format as build_messages().

        Returns:
            Integer token count for the prompt (input length).
        """
```

---

## 3. Board API

File: `cosee/board.py`

The Board maintains the shared visual workspace.

### 3.1 Class interface

```python
from typing import List, Callable, Optional

class Board:
    """
    Shared visual workspace storing cells with localized notes and metadata.
    """

    def __init__(self) -> None:
        """
        Initialize an empty Board.
        """

    def add_cell(
        self,
        view: View,
        content: str,
        tags: Optional[List[Tag]] = None,
        author: AgentName = "unknown",
        step: StepIndex = 0,
    ) -> CellID:
        """
        Add a new cell to the Board.

        Args:
            view: View information (page index, optional bbox/description).
            content: Text content of the note.
            tags: Optional list of tags.
            author: Name of the agent who created this cell.
            step: Collaboration step index.

        Returns:
            The unique CellID of the newly created cell.
        """

    def get_cell(self, cell_id: CellID) -> Optional[Cell]:
        """
        Retrieve a cell by its ID.

        Args:
            cell_id: ID of the cell.

        Returns:
            The Cell dict if found, otherwise None.
        """

    def list_cells(
        self,
        filter_fn: Optional[Callable[[Cell], bool]] = None,
    ) -> List[Cell]:
        """
        List cells, optionally filtered by a predicate.

        Args:
            filter_fn: A function Cell -> bool. If provided, only cells
                       for which filter_fn(cell) is True will be returned.

        Returns:
            A list of Cell dictionaries.
        """

    def get_cells_by_page(self, page: PageIndex) -> List[Cell]:
        """
        Convenience method to list all cells that refer to a given page.
        """

    def get_cells_by_author(self, author: AgentName) -> List[Cell]:
        """
        Convenience method to list all cells authored by a given agent.
        """

    def get_cells_by_tags(self, tags: List[Tag]) -> List[Cell]:
        """
        Convenience method to list all cells that contain ANY of the tags.
        """

    def to_text(
        self,
        max_cells_per_page: int = 8,
        max_total_chars: int = 2000,
    ) -> str:
        """
        Render the Board into a concise text summary appropriate for prompting.

        Behavior:
            - Group cells by page index.
            - Within each page, sort by (step, id).
            - Truncate per-page cell count to max_cells_per_page.
            - Ensure the total character length does not exceed max_total_chars
              by dropping oldest cells if necessary.

        Output format (example):

            [Page 1]
            - (#3, scanner, step 1) Title: "Quarterly Sales Report Q2"
            - (#7, detail_reader, step 2) The total revenue is $3.4M.

            [Page 2]
            - (#5, scanner, step 1) Contains a summary table of regions.

        Args:
            max_cells_per_page: Maximum number of cells to show per page.
            max_total_chars: Maximum total character count of the rendered string.

        Returns:
            A single string representing the Board summary.
        """

    def __len__(self) -> int:
        """
        Return the number of cells currently on the Board.
        """
```

---

## 4. Agent API

File: `cosee/agents.py`

Agents encapsulate role-specific behavior on top of `QwenVLClient`.

### 4.1 Base Agent interface

```python
from typing import Dict, Any

class Agent:
    """
    Base class for CoSee agents.

    Each agent:
        - Has a unique name.
        - Has a role-specific prompt.
        - Uses QwenVLClient to query the VLM.
    """

    def __init__(
        self,
        name: AgentName,
        role_prompt: str,
        qwen_client: QwenVLClient,
    ) -> None:
        self.name = name
        self.role_prompt = role_prompt
        self.qwen_client = qwen_client

    def act(
        self,
        images: List[Image.Image],
        question: str,
        board: Board,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Action:
        """
        Perform a single agent step.

        Procedure:
            - Obtain Board summary via board.to_text().
            - Call qwen_client.generate(...) with images, question, board summary,
              and this agent's role_prompt.
            - Parse the returned text into an Action dict.

        Args:
            images: List of page images relevant for this step.
            question: Question string.
            board: Current Board instance.
            gen_kwargs: Optional generation hyperparameters. If None, use
                        agent-specific defaults.

        Returns:
            An Action dict describing what the agent did (INSPECT/LINK/HYPOTHESIZE/REVISE).
        """
        raise NotImplementedError
```

### 4.2 Concrete agents

Agents like `ScannerAgent`, `DetailReaderAgent`, `CrossCheckerAgent` will subclass `Agent` and primarily override:

* The default `role_prompt`.
* Optional helper methods for action parsing.

Example:

```python
class ScannerAgent(Agent):
    """
    Agent focusing on global structure and locating candidate regions/pages.
    """

    def __init__(self, qwen_client: QwenVLClient) -> None:
        super().__init__(
            name="scanner",
            role_prompt=SCANNER_ROLE_PROMPT,
            qwen_client=qwen_client,
        )
```

---

## 5. CoSeeController API

File: `cosee/controller.py`

The controller orchestrates multi-round collaboration for a single QA instance.

### 5.1 Class interface

```python
from typing import List, Dict, Any, Optional

class CoSeeController:
    """
    Orchestrates multiple agents collaborating over a shared Board to solve a QA task.
    """

    def __init__(
        self,
        agents: List[Agent],
        max_steps: int = 3,
        default_gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            agents: List of Agent instances participating in collaboration.
            max_steps: Maximum number of collaboration rounds.
            default_gen_kwargs: Default generation kwargs passed to each agent's act()
                                if not overridden per call.
        """

    def validate_action(self, action: Action) -> Optional[Action]:
        """
        Validate and normalize an action returned by an agent.

        Behavior:
            - Check that 'action' field is one of the supported types.
            - Check required fields for each action type.
            - Fill in missing optional fields with safe defaults if possible.
            - Return None if the action is invalid.

        Args:
            action: Raw Action dict as returned by an agent.

        Returns:
            A validated Action dict, or None if the action is invalid.
        """

    def apply_action(
        self,
        board: Board,
        action: Action,
        step: StepIndex,
        author: AgentName,
    ) -> Optional[CellID]:
        """
        Apply a validated action to the Board.

        Behavior (examples):
            - For INSPECT: create a new cell with given view/content/tags.
            - For LINK: may add a new cell describing a link or update tags.
            - For HYPOTHESIZE: add a hypothesis cell, possibly with tags like "hypothesis".
            - For REVISE: add a correction cell referencing target_cell_id.

        Args:
            board: Board instance to modify.
            action: Validated Action dict.
            step: Current collaboration step index.
            author: Name of the agent producing this action.

        Returns:
            The new CellID if a cell was created, otherwise None.
        """

    def should_stop(
        self,
        step: StepIndex,
        hypotheses: List[Action],
        board: Board,
    ) -> bool:
        """
        Decide whether to stop collaboration early.

        Typical policy:
            - Stop if there is at least one high-confidence HYPOTHESIZE action
              and max_steps has not been reached.
            - Otherwise continue until max_steps.

        Args:
            step: Current step (1-based).
            hypotheses: List of accumulated HYPOTHESIZE actions.
            board: Current Board (for optional analysis).

        Returns:
            True if collaboration should stop, False otherwise.
        """

    def decide_final_answer(
        self,
        hypotheses: List[Action],
        board: Board,
        fallback_answer: str = "",
    ) -> AnswerText:
        """
        Choose a final answer based on accumulated hypotheses.

        Typical policy:
            - If no hypotheses, return fallback_answer or empty string.
            - If one hypothesis, return its 'answer' or 'content'.
            - If multiple, select the one with highest confidence, or majority vote
              over proposed answers.

        Args:
            hypotheses: List of HYPOTHESIZE actions.
            board: Final Board state.
            fallback_answer: Used if no valid hypothesis exists.

        Returns:
            Final answer string.
        """

    def run(
        self,
        images: List[Image.Image],
        question: str,
        per_agent_gen_kwargs: Optional[Dict[AgentName, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run a full CoSee collaboration episode for a single question.

        Behavior:
            - Initialize an empty Board.
            - For step in 1..max_steps:
                - For each agent in self.agents:
                    - Call agent.act(images, question, board, gen_kwargs).
                    - Validate action via validate_action().
                    - If valid, apply_action() to the Board.
                    - If action is HYPOTHESIZE, store it in a hypotheses list.
                - If should_stop(step, hypotheses, board) is True, break.

            - Decide final answer via decide_final_answer().
            - Optionally construct a simple trace for logging.

        Args:
            images: List of page images (shared among all agents).
            question: Question string.
            per_agent_gen_kwargs: Optional mapping from agent.name to its
                                  generation kwargs. If not provided, use
                                  self.default_gen_kwargs.

        Returns:
            A dictionary containing at least:
            {
                "answer": AnswerText,
                "board": Board,
                "hypotheses": List[Action],
                # Optional:
                "trace": Any,   # step-wise log structure, if implemented.
            }
        """
```

---

## 6. Dataset and Metrics API

### 6.1 Dataset loaders

File: `cosee/datasets/mpdocvqa.py` (similarly for other datasets).

```python
from typing import Iterator

class Sample(Dict[str, Any]):
    """
    Keys:
        - "images": List[Image.Image]
        - "question": str
        - "answer": str | List[str]
        - "meta": Dict[str, Any]
    """

def load_mpdocvqa_split(split: str) -> Iterator[Sample]:
    """
    Load a given split of MP-DocVQA (e.g., "train", "val", "test").

    Yields:
        Sample instances normalized into the common format.
    """

def build_dev_subset(
    split: str,
    max_samples: int,
) -> List[Sample]:
    """
    Build a small dev subset for fast experimentation.

    Args:
        split: Base split name.
        max_samples: Maximum number of samples to include.

    Returns:
        A list of Sample dictionaries.
    """
```

### 6.2 Metrics

File: `cosee/eval/metrics.py`

```python
from typing import List, Tuple

def normalize_answer(text: str) -> str:
    """
    Normalize an answer string:
        - lowercase
        - strip whitespace
        - optionally remove punctuation or articles (dataset-dependent)
    """

def exact_match(pred: str, gold: str | List[str]) -> float:
    """
    Compute exact match (EM) between a prediction and one or more gold answers.

    Args:
        pred: Predicted answer.
        gold: Single gold answer or a list of acceptable answers.

    Returns:
        1.0 if EM holds, else 0.0.
    """

def f1_score(pred: str, gold: str | List[str]) -> float:
    """
    Compute token-level F1 between prediction and gold answer(s).

    Returns:
        F1 score between 0.0 and 1.0.
    """

def anls(pred: str, gold: str | List[str]) -> float:
    """
    Compute the ANLS (Average Normalized Levenshtein Similarity) metric.

    Returns:
        ANLS score between 0.0 and 1.0.
    """

def aggregate_metrics(
    preds: List[str],
    golds: List[str | List[str]],
) -> Dict[str, float]:
    """
    Aggregate EM / F1 / ANLS over a dataset.

    Args:
        preds: List of predicted answers.
        golds: List of gold answers (each can be str or List[str]).

    Returns:
        A dict like:
        {
            "em": float,
            "f1": float,
            "anls": float,
        }
    """
```

---

## 7. Baselines API

File: `cosee/eval/baselines.py`

These functions implement non-CoSee baselines using the same `QwenVLClient` interface.

```python
from typing import Optional

def single_agent_cot(
    qwen_client: QwenVLClient,
    images: List[Image.Image],
    question: str,
    max_new_tokens: int = 128,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> AnswerText:
    """
    Single-agent chain-of-thought baseline.

    Behavior:
        - Build a prompt that explicitly asks the model to reason step by step.
        - Call qwen_client.generate(...) once.
        - Postprocess to extract final answer (if needed).

    Returns:
        Predicted answer text.
    """

def self_consistency(
    qwen_client: QwenVLClient,
    images: List[Image.Image],
    question: str,
    num_samples: int = 3,
    max_new_tokens: int = 128,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> AnswerText:
    """
    Self-consistency baseline.

    Behavior:
        - Run single-agent CoT num_samples times with sampling enabled.
        - Aggregate final answers by majority vote.
        - Return the majority answer (breaking ties arbitrarily or by confidence).

    Returns:
        Aggregated predicted answer.
    """

def multi_agent_chat_no_board(
    agents: List[Agent],
    images: List[Image.Image],
    question: str,
    max_turns: int = 3,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> AnswerText:
    """
    Multi-agent chat baseline without a shared Board.

    Behavior:
        - Maintain a text-only chat history buffer.
        - For each turn, each agent reads the chat history and appends a new message.
        - After max_turns, select the final answer from the last agent or a simple
          aggregation rule.

    Returns:
        Predicted answer text.
    """
```

---

## 8. Script-level API (entry points)

Scripts under `scripts/` (e.g., `run_mpdocvqa_cosee.py`, `run_mpdocvqa_baselines.py`) are *not* specified exhaustively here, but they should adhere to the following pattern:

* Parse command-line arguments:

  * Model path.
  * Dataset split.
  * Max steps / max rounds / number of agents.
  * Output directory for logs.
* Construct:

  * `QwenVLClient` instance.
  * Relevant `Agent` instances.
  * `CoSeeController` or baseline runner.
* Iterate over samples:

  * For each sample: run CoSee or baseline and collect predictions.
* Aggregate metrics via `metrics.aggregate_metrics`.
* Optionally dump:

  * Per-sample predictions.
  * Board traces (for CoSee).
  * Simple JSONL logs.

---

This specification intentionally focuses on **interfaces and responsibilities**.
Implementation details (e.g., how exactly to parse JSON from model outputs) are left flexible, as long as the public API contracts in this document are respected.