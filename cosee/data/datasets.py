from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COSEE_DATA_ROOT", ROOT / "data")).expanduser()

_ANNOTATION_FILENAMES = {
    "slidevqa": "slidevqa.jsonl",
    "chartqapro": "chartqapro.jsonl",
    "vqaonline": "vqaonline.jsonl",
}


@dataclass(frozen=True)
class Example:
    id: str
    dataset: str
    split: str
    image_paths: List[str]
    question: str
    answer: str
    meta: Dict[str, Any]


def annotation_path(dataset: str) -> Path:
    """Return the normalized JSONL annotation path for a dataset."""
    if dataset not in _ANNOTATION_FILENAMES:
        known = ", ".join(sorted(_ANNOTATION_FILENAMES))
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {known}")
    return DATA_ROOT / dataset / "annotations" / _ANNOTATION_FILENAMES[dataset]


def _coerce_str(value: Any, field_name: str, line_no: int, path: Path) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    raise ValueError(
        f"{path}:{line_no}: field '{field_name}' must be string-like, got {type(value).__name__}"
    )


def _coerce_image_paths(value: Any, line_no: int, path: Path) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path}:{line_no}: field 'image_paths' must be a list")
    image_paths: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path}:{line_no}: image_paths entries must be non-empty strings")
        image_paths.append(item.strip())
    return image_paths


def _example_from_record(record: Dict[str, Any], line_no: int, path: Path) -> Example:
    dataset = _coerce_str(record.get("dataset"), "dataset", line_no, path).strip()
    split = _coerce_str(record.get("split"), "split", line_no, path).strip()
    ex_id = _coerce_str(record.get("id"), "id", line_no, path).strip()
    if not dataset or not split or not ex_id:
        raise ValueError(f"{path}:{line_no}: id, dataset, and split are required")

    meta = record.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path}:{line_no}: field 'meta' must be an object when present")

    return Example(
        id=ex_id,
        dataset=dataset,
        split=split,
        image_paths=_coerce_image_paths(record.get("image_paths", []), line_no, path),
        question=_coerce_str(record.get("question"), "question", line_no, path).strip(),
        answer=_coerce_str(record.get("answer"), "answer", line_no, path).strip(),
        meta=meta,
    )


def load_jsonl(path: Path, *, dataset: Optional[str] = None, split: Optional[str] = None) -> List[Example]:
    """Load normalized CoSee examples from a JSONL annotation file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {path}. "
            "Export a dataset first with one of the scripts/export_* helpers."
        )

    examples: List[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")

            example = _example_from_record(record, line_no, path)
            if dataset is not None and example.dataset != dataset:
                continue
            if split is not None and example.split != split:
                continue
            examples.append(example)
    return examples


def _limit(examples: Iterable[Example], max_examples: Optional[int]) -> List[Example]:
    if max_examples is None or max_examples < 0:
        return list(examples)
    limited: List[Example] = []
    for ex in examples:
        if len(limited) >= max_examples:
            break
        limited.append(ex)
    return limited


def load_toy_split(
    dataset: str,
    split: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> List[Example]:
    """
    Load a normalized dataset split from data/<dataset>/annotations/*.jsonl.

    The historical script name says "toy", but this function simply reads the
    normalized JSONL exported by this repository's dataset scripts.
    """
    path = annotation_path(dataset)
    examples = load_jsonl(path, dataset=dataset, split=split)
    return _limit(examples, max_examples)
