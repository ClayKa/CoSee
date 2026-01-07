from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate baseline or CoSee results JSONL.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "cosee"],
        required=True,
        help="Result type: 'baseline' for run_qwen_single_baseline outputs, 'cosee' for run_cosee_on_dataset outputs.",
    )
    parser.add_argument(
        "--dataset",
        choices=["slidevqa", "chartqapro", "vqaonline"],
        required=True,
        help="Dataset name for metric semantics.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        help="Optional split name for reporting (not used for filtering).",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL result files to aggregate.",
    )
    return parser.parse_args()


def load_records(paths: List[str]) -> Dict[str, Any]:
    """
    Load all JSONL records from the given files and deduplicate by id.
    If the same id appears multiple times, keep the last occurrence.
    """
    records_by_id: Dict[str, Any] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ex_id = obj.get("id")
                if ex_id is None:
                    continue
                records_by_id[ex_id] = obj
    return records_by_id


def is_chartqapro_answerable(rec: Dict[str, Any]) -> bool:
    gold_norm = (rec.get("gold_norm") or "").strip().lower()
    raw_answer = rec.get("answer") or rec.get("gold_answer") or ""
    raw_answer = raw_answer.strip().lower()
    if gold_norm == "unanswerable" or raw_answer == "unanswerable":
        return False
    return True


def aggregate_baseline(dataset: str, records: List[Dict[str, Any]]) -> None:
    # Filter by dataset just in case
    records = [r for r in records if r.get("dataset") == dataset]
    if not records:
        print("[WARN] No records to aggregate after dataset filter.")
        return

    total = len(records)
    num_strict = sum(1 for r in records if r.get("correct_exact"))
    num_loose = sum(1 for r in records if r.get("correct_loose"))

    print(f"Total examples: {total}")
    if dataset == "vqaonline":
        f1_vals = [r.get("score_f1") for r in records if isinstance(r.get("score_f1"), (int, float))]
        avg_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
        strict_acc = num_strict / total
        loose_acc = num_loose / total
        print(f"VQAonline mean token-F1: {avg_f1:.3f}")
        print(f"VQAonline strict (F1>=0.5): {strict_acc:.3f} ({num_strict} / {total})")
        print(f"VQAonline relaxed (F1>=0.3): {loose_acc:.3f} ({num_loose} / {total})")
    else:
        strict_acc = num_strict / total
        loose_acc = num_loose / total
        print(f"Exact match accuracy: {strict_acc:.3f} ({num_strict} / {total})")
        print(f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})")

    if dataset == "chartqapro":
        ans_records = [r for r in records if is_chartqapro_answerable(r)]
        if ans_records:
            ans_total = len(ans_records)
            ans_strict = sum(1 for r in ans_records if r.get("correct_exact"))
            ans_loose = sum(1 for r in ans_records if r.get("correct_loose"))
            print(
                f"Answerable-only exact: {ans_strict/ans_total:.3f} "
                f"({ans_strict} / {ans_total})"
            )
            print(
                f"Answerable-only loose: {ans_loose/ans_total:.3f} "
                f"({ans_loose} / {ans_total})"
            )


def aggregate_cosee(dataset: str, records: List[Dict[str, Any]]) -> None:
    # Filter by dataset
    records = [r for r in records if r.get("dataset") == dataset]
    if not records:
        print("[WARN] No records to aggregate after dataset filter.")
        return

    # Group by agent_config
    by_config: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        cfg = r.get("agent_config", "unknown")
        by_config.setdefault(cfg, []).append(r)

    for cfg, recs in by_config.items():
        total = len(recs)
        num_strict = sum(1 for r in recs if r.get("correct_exact"))
        num_loose = sum(1 for r in recs if r.get("correct_loose"))

        print(f"\nAgent config: {cfg}")
        print(f"Total examples: {total}")

        if dataset == "vqaonline":
            f1_vals = [r.get("score_f1") for r in recs if isinstance(r.get("score_f1"), (int, float))]
            avg_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
            strict_acc = num_strict / total
            loose_acc = num_loose / total
            print(f"VQAonline mean token-F1: {avg_f1:.3f}")
            print(f"VQAonline strict (F1>=0.5): {strict_acc:.3f} ({num_strict} / {total})")
            print(f"VQAonline relaxed (F1>=0.3): {loose_acc:.3f} ({num_loose} / {total})")
        else:
            strict_acc = num_strict / total
            loose_acc = num_loose / total
            print(f"Exact match accuracy: {strict_acc:.3f} ({num_strict} / {total})")
            print(f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})")

        if dataset == "chartqapro":
            ans_records = [r for r in recs if is_chartqapro_answerable(r)]
            if ans_records:
                ans_total = len(ans_records)
                ans_strict = sum(1 for r in ans_records if r.get("correct_exact"))
                ans_loose = sum(1 for r in ans_records if r.get("correct_loose"))
                print(
                    f"Answerable-only exact: {ans_strict/ans_total:.3f} "
                    f"({ans_strict} / {ans_total})"
                )
                print(
                    f"Answerable-only loose: {ans_loose/ans_total:.3f} "
                    f"({ans_loose} / {ans_total})"
                )


def main() -> None:
    args = parse_args()
    records_by_id = load_records(args.inputs)
    records = list(records_by_id.values())
    if len(records) == 0:
        print("[WARN] No records loaded.")
        return

    print(f"Aggregating {len(records)} deduplicated records for dataset={args.dataset}")
    if args.mode == "baseline":
        aggregate_baseline(args.dataset, records)
    else:
        aggregate_cosee(args.dataset, records)


if __name__ == "__main__":
    main()
