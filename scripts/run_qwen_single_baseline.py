from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import torch

from cosee.data.datasets import ROOT, load_toy_split
from cosee.models.qwen_vl_wrapper import QwenVLClient
from cosee.metrics import compute_vqaonline_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-model Qwen baseline on toy datasets.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.run_qwen_single_baseline --dataset chartqapro --split test "
            "--only-ids ids/chartqapro_200.txt --log-compute --two-stage --device cuda\n"
            "  python -m scripts.run_qwen_single_baseline --dataset vqaonline --split trainval "
            "--only-ids ids/vqaonline_200.txt --log-compute --two-stage --device cuda\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["slidevqa", "chartqapro", "vqaonline"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split. Defaults: slidevqa=train, chartqapro=test, vqaonline=test.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Maximum number of examples to evaluate.",
    )
    parser.add_argument(
        "--only-ids",
        type=str,
        default=None,
        help="Optional path to newline-separated example ids to run (in that exact order).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local Qwen3-VL-4B-Instruct. Defaults to COSEE_MODEL_PATH or ./models/Qwen3-VL-4B-Instruct.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path for per-example results.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="If set, run evidence generation followed by a final answer stage.",
    )
    parser.add_argument(
        "--evidence-max-new-tokens",
        type=int,
        default=None,
        help="Max tokens for evidence stage (defaults to min(128, max_new_tokens)).",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-compute",
        action="store_true",
        help="If set, log per-example compute proxies (generate calls and token counts).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to an existing JSONL results file to resume from. "
            "If provided and the file exists, skip examples already present "
            "in that file and continue with remaining ones."
        ),
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help=(
            "Optional explicit path for the output JSONL results file. "
            "If not provided, defaults to standard naming. "
            "When used with --resume-from, new results are appended to this path "
            "(or to --resume-from if results-path is None)."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards to split the loaded examples into.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard id (0-based). Only examples with idx %% num_shards == shard_id are processed.",
    )
    parser.add_argument(
        "--run-all-shards",
        action="store_true",
        help="If set, iterate over all shard IDs [0..num_shards-1] in this process, reusing the same model.",
    )
    return parser.parse_args()


def resolve_model_path(path_arg: str | None) -> str:
    if path_arg:
        return path_arg
    return os.environ.get("COSEE_MODEL_PATH", "./models/Qwen3-VL-4B-Instruct")


def default_split_for(dataset: str) -> str:
    if dataset == "slidevqa":
        return "train"
    return "test"


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_loose_match(gold_norm: str, pred_norm: str) -> bool:
    return bool(gold_norm) and (gold_norm in pred_norm or pred_norm in gold_norm)


def is_chartqapro_answerable_from_record(rec: Dict[str, Any]) -> bool:
    gold_norm = (rec.get("gold_norm") or "").strip().lower()
    raw_answer = (rec.get("gold_answer") or rec.get("answer") or "").strip().lower()
    return not (gold_norm == "unanswerable" or raw_answer == "unanswerable")


def build_input_text_for_example(ex, max_context_chars: int = 1500) -> str:
    """
    Build the text input for the model, optionally including context for VQAonline.
    """
    if getattr(ex, "dataset", None) == "vqaonline":
        # For vqaonline, question text is passed separately; context is handled downstream.
        return ex.question
    return ex.question


def load_images(image_paths: List[str]) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for p in image_paths:
        path = ROOT / p
        img = Image.open(path).convert("RGB")
        imgs.append(img)
    return imgs


def read_only_ids(path: str) -> List[str]:
    ids: List[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def _sort_results_file(path: Path) -> None:
    """
    Sort JSONL results by result_serial (if present) or id, and rewrite the file.
    """
    try:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except json.JSONDecodeError:
                    continue

        def sort_key(obj: Dict[str, Any]):
            val = obj.get("result_serial")
            try:
                return (0, int(val))
            except Exception:
                return (1, str(obj.get("id", "")))

        records.sort(key=sort_key)

        with path.open("w", encoding="utf-8") as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort; do not crash the script if sorting fails
        return


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model_path)
    split = args.split or default_split_for(args.dataset)
    default_results_path = Path("results") / f"baseline_qwen_single_{args.dataset}_{split}.jsonl"
    output_path = Path(args.results_path) if args.results_path else default_results_path
    resume_path = Path(args.resume_from) if args.resume_from else output_path

    if args.resume_from and not Path(args.resume_from).exists():
        print(f"[ERROR] Resume file not found: {args.resume_from}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # Resume bookkeeping
    existing_ids = set()
    num_exact_done = 0
    num_loose_done = 0
    total_f1_done = 0.0
    n_done = 0
    answerable_total_done = 0
    answerable_correct_exact_done = 0
    answerable_correct_loose_done = 0

    if resume_path and resume_path.exists():
        with resume_path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ex_id = rec.get("id")
                if not ex_id or ex_id in existing_ids:
                    continue

                existing_ids.add(ex_id)
                n_done += 1
                if rec.get("correct_exact"):
                    num_exact_done += 1
                if rec.get("correct_loose"):
                    num_loose_done += 1
                if args.dataset == "chartqapro" and is_chartqapro_answerable_from_record(rec):
                    answerable_total_done += 1
                    if rec.get("correct_exact"):
                        answerable_correct_exact_done += 1
                    if rec.get("correct_loose"):
                        answerable_correct_loose_done += 1
                if (
                    args.dataset == "vqaonline"
                    and "score_f1" in rec
                    and isinstance(rec["score_f1"], (int, float))
                ):
                    total_f1_done += rec["score_f1"]

        print(
            f"[RESUME] Found {n_done} existing results in {resume_path}. "
            f"{len(existing_ids)} unique example ids."
        )

    try:
        examples = load_toy_split(
            dataset=args.dataset,
            split=split,
            max_examples=None if args.only_ids else args.max_examples,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        if args.dataset == "vqaonline":
            print(
                "Hint: export the toy split first via `python -m scripts.export_vqaonline_toy` "
                "and ensure the requested split exists in the JSONL."
            )
        return
    if args.only_ids:
        only_ids = read_only_ids(args.only_ids)
        id_to_ex = {ex.id: ex for ex in examples}
        filtered: List[Any] = []
        for ex_id in only_ids:
            ex = id_to_ex.get(ex_id)
            if ex is None:
                print(f"[WARN] only-ids: id not found in dataset: {ex_id}")
                continue
            filtered.append(ex)
        if args.max_examples is not None:
            filtered = filtered[: args.max_examples]
        examples = filtered

    print(f"Loaded {len(examples)} examples from {args.dataset}/{split}")

    num_shards = max(1, args.num_shards)
    shard_id = args.shard_id
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id={shard_id} for num_shards={num_shards}")

    original_count = len(examples)
    shard_ids = list(range(num_shards)) if args.run_all_shards else [shard_id]
    if not args.run_all_shards and num_shards > 1:
        examples = [ex for idx, ex in enumerate(examples) if idx % num_shards == shard_id]
        print(
            f"Sharding enabled: num_shards={num_shards}, shard_id={shard_id}. "
            f"Using {len(examples)} / {original_count} examples in this run."
        )
    else:
        print(
            f"Sharding: num_shards={num_shards}, running shard ids {shard_ids}. "
            f"Total loaded examples: {len(examples)}"
        )

    n_total_planned = len(examples)

    # Dataset-specific default for max_new_tokens (only if user did not set it)
    if args.max_new_tokens is None:
        args.max_new_tokens = 48 if args.dataset == "vqaonline" else 64
    if args.evidence_max_new_tokens is None:
        args.evidence_max_new_tokens = min(128, args.max_new_tokens)

    client = QwenVLClient(
        model_path=model_path,
        device=args.device,
        dtype="auto",
    )

    role_prompt = (
        "You are a careful multimodal assistant that answers questions about one or more images. "
        "Answer concisely, without explanation."
    )

    total = n_done
    num_exact = num_exact_done
    num_loose = num_loose_done
    f1_sum = total_f1_done
    answerable_total = answerable_total_done
    answerable_correct_exact = answerable_correct_exact_done
    answerable_correct_loose = answerable_correct_loose_done
    result_idx = n_done
    total_examples = len(examples)
    total_processed = n_done
    compute_examples = 0
    compute_calls_total = 0
    compute_gen_tokens_total = 0

    mode = "a" if output_path.exists() and resume_path else "w"
    with output_path.open(mode, encoding="utf-8") as fout:
        if resume_path and resume_path.exists() and output_path != resume_path and mode == "w":
            with resume_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

        for shard_id in shard_ids:
            print(f"[INFO] Running shard {shard_id}/{args.num_shards - 1}")
            for idx, ex in enumerate(examples):
                if args.num_shards > 1 and idx % args.num_shards != shard_id:
                    continue
                if ex.id in existing_ids:
                    continue

                images = load_images(ex.image_paths)
                question_input = build_input_text_for_example(ex)
                meta = getattr(ex, "meta", {}) or {}
                context = meta.get("context", "") if args.dataset == "vqaonline" else None
                if context:
                    context = context.strip()
                    if len(context) > 1500:
                        context = context[:1500] + " ..."
                question_text = (
                    f"Question: {question_input}\n"
                    "Please answer with a short phrase or number, without explanation."
                )

                if args.two_stage:
                    evidence_prompt = (
                        f"Question: {question_input}\n\n"
                        "Write 3-6 bullet-point evidence notes grounded in the images. "
                        "Do NOT answer the question."
                    )
                    if args.log_compute:
                        client.reset_compute()
                    evidence_text = client.generate(
                        images=images,
                        question=evidence_prompt,
                        context=context,
                        dataset=args.dataset,
                        board_text=None,
                        role_prompt=role_prompt,
                        max_new_tokens=args.evidence_max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    ).strip()
                    evidence_compute = client.get_compute() if args.log_compute else {"calls": 0, "gen_tokens_total": 0}

                    if args.log_compute:
                        client.reset_compute()
                    if args.dataset == "chartqapro":
                        final_prompt = (
                            f"Question: {question_input}\n\n"
                            f"Evidence notes:\n{evidence_text}\n\n"
                            "Answer with ONLY one of: A, B, C, D, or Unanswerable."
                        )
                    elif args.dataset == "vqaonline":
                        final_prompt = (
                            f"Question: {question_input}\n\n"
                            f"Evidence notes:\n{evidence_text}\n\n"
                            "Answer with a short sentence or phrase. No explanation."
                        )
                    else:
                        final_prompt = (
                            f"Question: {question_input}\n\n"
                            f"Evidence notes:\n{evidence_text}\n\n"
                            "Answer with a short phrase or number, without explanation."
                        )
                    final_answer_raw = client.generate(
                        images=images,
                        question=final_prompt,
                        context=context,
                        dataset=args.dataset,
                        board_text=None,
                        role_prompt=role_prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        return_full_text=True,
                    ).strip()
                    pred_text = client._extract_assistant_answer(final_answer_raw)
                    final_compute = client.get_compute() if args.log_compute else {"calls": 0, "gen_tokens_total": 0}
                    compute_breakdown = None
                    if args.log_compute:
                        compute_breakdown = [
                            {
                                "name": "qwen_client:evidence",
                                "calls": evidence_compute["calls"],
                                "gen_tokens_total": evidence_compute["gen_tokens_total"],
                            },
                            {
                                "name": "qwen_client:final",
                                "calls": final_compute["calls"],
                                "gen_tokens_total": final_compute["gen_tokens_total"],
                            },
                        ]
                        total_calls = evidence_compute["calls"] + final_compute["calls"]
                        total_tokens = evidence_compute["gen_tokens_total"] + final_compute["gen_tokens_total"]
                        compute_calls_total += total_calls
                        compute_gen_tokens_total += total_tokens
                        compute_examples += 1
                else:
                    if args.log_compute:
                        client.reset_compute()
                    pred_text = client.generate(
                        images=images,
                        question=question_text,
                        context=context,
                        dataset=args.dataset,
                        board_text=None,
                        role_prompt=role_prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    ).strip()

                    compute_breakdown = None
                    if args.log_compute:
                        compute = client.get_compute()
                        compute_calls_total += compute["calls"]
                        compute_gen_tokens_total += compute["gen_tokens_total"]
                        compute_examples += 1
                        compute_breakdown = [
                            {
                                "name": "qwen_client",
                                "calls": compute["calls"],
                                "gen_tokens_total": compute["gen_tokens_total"],
                            }
                        ]


                gold_norm = normalize_answer(ex.answer)
                pred_norm = normalize_answer(pred_text)

                if args.dataset == "vqaonline":
                    score_f1 = compute_vqaonline_f1(ex.answer, pred_text)
                    correct_exact = score_f1 >= 0.5
                    correct_loose = score_f1 >= 0.3
                    f1_sum += score_f1
                else:
                    correct_exact = pred_norm == gold_norm
                    correct_loose = is_loose_match(gold_norm, pred_norm)

                total += 1
                num_exact += int(correct_exact)
                num_loose += int(correct_loose)
                if args.dataset == "chartqapro":
                    gold_raw_str = str(ex.answer).strip().lower()
                    is_unanswerable = gold_raw_str == "unanswerable"
                    if not is_unanswerable:
                        answerable_total += 1
                        answerable_correct_exact += int(correct_exact)
                        answerable_correct_loose += int(correct_loose)

                total_processed += 1
                if total_processed % 5 == 0:
                    print(
                        f"[{total_processed}/{total_examples}] id={ex.id} dataset={args.dataset}/{split}"
                    )

                if total % 10 == 0:
                    if args.dataset == "vqaonline":
                        print(
                            f"[{total}/{n_total_planned}] id={ex.id} f1={score_f1:.3f} "
                            f"strict(>=0.5)={int(correct_exact)} loose(>=0.3)={int(correct_loose)}"
                        )
                    else:
                        print(
                            f"[{total}/{n_total_planned}] id={ex.id} exact={int(correct_exact)} loose={int(correct_loose)}"
                        )

                record: Dict[str, Any] = {
                    "id": ex.id,
                    "dataset": ex.dataset,
                    "split": ex.split,
                    "question": ex.question,
                    "gold_answer": ex.answer,
                    "pred_answer": pred_text,
                    "gold_norm": gold_norm,
                    "pred_norm": pred_norm,
                    "correct_exact": correct_exact,
                    "correct_loose": correct_loose,
                    "meta": ex.meta or {},
                    "result_serial": f"{result_idx:06d}",
                }
                if args.dataset == "vqaonline":
                    record["score_f1"] = score_f1
                if args.log_compute:
                    record["num_generate_calls"] = sum(item["calls"] for item in compute_breakdown)
                    record["gen_tokens_total"] = sum(item["gen_tokens_total"] for item in compute_breakdown)
                    record["compute_version"] = "v1"
                    record["compute_breakdown"] = compute_breakdown
                if args.two_stage:
                    record["evidence_text"] = evidence_text
                    record["final_answer_raw"] = final_answer_raw
                    record["final_answer_source"] = "two_stage_evidence_then_final"
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing_ids.add(ex.id)
                result_idx += 1

    if total == 0:
        print("[WARN] No examples were evaluated (total==0).")
        return

    exact_acc = num_exact / total if total else 0.0
    loose_acc = num_loose / total if total else 0.0
    if args.dataset == "vqaonline":
        avg_f1 = f1_sum / total if total else 0.0
        print(
            f"Finished {total} examples from {args.dataset}/{split}.\n"
            f"F1 average: {avg_f1:.3f}\n"
            f"Strict (F1>=0.5): {exact_acc:.3f} ({num_exact} / {total})\n"
            f"Relaxed (F1>=0.3): {loose_acc:.3f} ({num_loose} / {total})"
        )
    elif args.dataset == "chartqapro":
        print(
            f"Finished {total} examples from {args.dataset}/{split}.\n"
            f"Exact match accuracy: {exact_acc:.3f} ({num_exact} / {total})\n"
            f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})"
        )
        if answerable_total > 0:
            exact_ans = answerable_correct_exact / answerable_total
            loose_ans = answerable_correct_loose / answerable_total
            print(
                f"Exact match accuracy (answerable-only): {exact_ans:.3f} "
                f"({answerable_correct_exact} / {answerable_total})"
            )
            print(
                f"Loose match accuracy (answerable-only): {loose_ans:.3f} "
                f"({answerable_correct_loose} / {answerable_total})"
            )
    else:
        print(
            f"Finished {total} examples from {args.dataset}/{split}.\n"
            f"Exact match accuracy: {exact_acc:.3f} ({num_exact} / {total})\n"
            f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})"
        )

    _sort_results_file(output_path)

    if args.log_compute and compute_examples > 0:
        mean_calls = compute_calls_total / compute_examples
        mean_tokens = compute_gen_tokens_total / compute_examples
        print(
            f"Compute summary (v1): mean generate calls={mean_calls:.2f}, "
            f"mean gen tokens={mean_tokens:.1f} over {compute_examples} examples."
        )


if __name__ == "__main__":
    main()
