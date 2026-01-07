from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess ChartQAPro predictions and recompute metrics.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL with raw predictions.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL with postprocessed fields.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file.")
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_loose_match(gold_norm: str, pred_norm: str) -> bool:
    return bool(gold_norm) and (gold_norm in pred_norm or pred_norm in gold_norm)


OPTION_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def extract_option_letter(text: str) -> str | None:
    # Look for standalone A/B/C/D possibly in parentheses or prefixed by "option"/"answer"
    matches = OPTION_RE.findall(text)
    if matches:
        return matches[0].upper()
    return None


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def first_number_token(text: str) -> str | None:
    # numeric digit
    m = re.search(r"\d+(\.\d+)?", text)
    if m:
        return m.group(0)
    # number word
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    for tok in tokens:
        if tok in NUMBER_WORDS:
            return NUMBER_WORDS[tok]
    return None


def postprocess_answer(raw: str, meta: Dict[str, Any]) -> str:
    raw = raw.strip()
    qtype = (meta.get("Question Type") or meta.get("question_type") or "").strip()

    # Multi-choice: extract option letter
    if qtype.lower() == "multi choice":
        letter = extract_option_letter(raw)
        if letter:
            return letter
        return raw  # fallback

    # Factoid / numeric heuristics
    lower = raw.lower()

    # If contains "answer:"
    if "answer:" in lower:
        idx = lower.find("answer:")
        snippet = raw[idx + len("answer:") :].strip()
        snippet = snippet.split("\n")[0]
        snippet = snippet.split(".")[0]
        if snippet:
            return snippet.strip(" \"')(")

    # If contains colon
    if ":" in raw:
        parts = raw.split(":", 1)
        tail = parts[1].strip()
        if len(tail.split()) <= 6:
            return tail.strip(" \"')(")

    # Count-like questions
    if any(kw in lower for kw in ["how many", "number of", "total", "count"]):
        num = first_number_token(raw)
        if num:
            return num

    # Single sentence with is/are
    m = re.search(r"\b(is|are)\b\s+([^\.]+)", raw, re.IGNORECASE)
    if m:
        candidate = m.group(2).strip()
        candidate = candidate.split(".")[0]
        if candidate:
            return candidate.strip(" \"')(")

    # Fallback: strip punctuation/quotes
    return raw.strip(" \"')(")


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        print(f"[ERROR] Output file exists: {output_path}. Use --overwrite to replace.")
        return

    records = load_records(input_path)
    if not records:
        print("[WARN] No records loaded.")
        return

    total = len(records)
    num_mc = 0
    num_fact = 0
    raw_exact = sum(1 for r in records if r.get("correct_exact"))
    raw_loose = sum(1 for r in records if r.get("correct_loose"))
    pp_exact = 0
    pp_loose = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in records:
            pred = rec.get("pred_answer")
            if not pred:
                print(f"[WARN] Missing pred_answer for id={rec.get('id')}, skipping.")
                continue

            meta = rec.get("meta", {}) or {}
            qtype = (meta.get("Question Type") or meta.get("question_type") or "").strip()
            if qtype.lower() == "multi choice":
                num_mc += 1
            else:
                num_fact += 1

            pred_pp = postprocess_answer(pred, meta)
            pred_norm_pp = normalize_answer(pred_pp)
            gold_norm = rec.get("gold_norm", normalize_answer(rec.get("gold_answer", "")))

            correct_exact_pp = pred_norm_pp == gold_norm
            correct_loose_pp = is_loose_match(gold_norm, pred_norm_pp)

            rec["pred_answer_pp"] = pred_pp
            rec["pred_norm_pp"] = pred_norm_pp
            rec["correct_exact_pp"] = correct_exact_pp
            rec["correct_loose_pp"] = correct_loose_pp

            pp_exact += int(correct_exact_pp)
            pp_loose += int(correct_loose_pp)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Loaded {total} records.")
    print(f"Multi Choice examples: {num_mc}, Factoid/Other: {num_fact}")
    print(f"Raw exact/loose: {raw_exact}/{total} ({raw_exact/total:.3f}), {raw_loose}/{total} ({raw_loose/total:.3f})")
    print(f"Postprocessed exact/loose: {pp_exact}/{total} ({pp_exact/total:.3f}), {pp_loose}/{total} ({pp_loose/total:.3f})")


if __name__ == "__main__":
    main()
