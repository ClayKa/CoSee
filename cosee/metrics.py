from __future__ import annotations

from collections import Counter
from typing import List

_VQAONLINE_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "at",
    "as",
    "by",
    "from",
    "about",
    "you",
    "your",
    "i",
    "we",
    "they",
    "he",
    "she",
    "him",
    "her",
    "them",
    "not",
    "no",
    "yes",
}


def _normalize_text_for_overlap(text: str) -> List[str]:
    """
    Normalize a long free-form answer for VQAonline-style lexical overlap:

    - lowercase
    - remove basic punctuation
    - split on whitespace
    - drop very short tokens and common stopwords
    """
    if text is None:
        return []

    text = text.lower()

    punct = ",.;:!?\"'()[]{}<>`~+-=/\\|"
    for ch in punct:
        text = text.replace(ch, " ")

    tokens = [t.strip() for t in text.split() if t.strip()]

    filtered = [t for t in tokens if t not in _VQAONLINE_STOPWORDS and len(t) > 1]
    return filtered


def compute_vqaonline_f1(gold: str, pred: str) -> float:
    """
    Compute a token-level F1 score between gold and predicted answers
    for VQAonline-style long answers.
    """
    gold_tokens = _normalize_text_for_overlap(gold)
    pred_tokens = _normalize_text_for_overlap(pred)

    if len(gold_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)

    common = gold_counts & pred_counts
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
