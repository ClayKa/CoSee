"""
Run a single-agent Qwen-4B model with a shared textual board.

This script is parallel to run_qwen_single_baseline.py and run_cosee_on_dataset.py.
It keeps the same dataset loading, metrics, and decoding settings as the baseline,
but replaces the single-pass call with a multi-step controller where a single agent
reads and writes to a shared Board for T steps before answering.

Usage example:

  python -m scripts.run_single_qwen_board \
    --dataset slidevqa \
    --split train \
    --max-examples 200 \
    --num-shards 2 \
    --run-all-shards \
    --max-new-tokens 32 \
    --device cuda
"""

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

from cosee.agents import QwenAgent
from cosee.board import Board
from cosee.controller import CoSeeController
from cosee.data.datasets import ROOT, load_toy_split
from cosee.metrics import compute_vqaonline_f1
from cosee.models.qwen_vl_wrapper import QwenVLClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-agent Qwen with a shared board.")
    parser.add_argument("--dataset", type=str, required=True, choices=["slidevqa", "chartqapro", "vqaonline"])
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from existing JSONL; skips already processed ids.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Optional explicit output JSONL; defaults to results/single_qwen_board_{dataset}_{split}.jsonl",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--run-all-shards",
        action="store_true",
        help="If set, iterate over all shard IDs [0..num_shards-1] in this process, reusing the same model.",
    )
    parser.add_argument(
        "--log-compute",
        action="store_true",
        help="If set, log per-example compute proxies (generate calls and token counts).",
    )
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument(
        "--only-ids",
        type=str,
        default=None,
        help="Optional path to newline-separated example ids to run. If provided, only those ids are evaluated.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, skip ids already present in the output file with a non-note-like pred_answer.",
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


def read_only_ids(path: str) -> List[str]:
    ids: List[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def is_loose_match(gold_norm: str, pred_norm: str) -> bool:
    return bool(gold_norm) and (gold_norm in pred_norm or pred_norm in gold_norm)


def is_note_like(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    return t.startswith("Page 1:") or t.startswith("[step=") or ("[BoardAgent]" in t) or ("tags=qwen-note" in t)


def build_input_text_for_example(ex, max_context_chars: int = 1500) -> str:
    """
    Build the text input for the model, optionally including context for VQAonline.
    """
    if getattr(ex, "dataset", None) == "vqaonline":
        meta = getattr(ex, "meta", {}) or {}
        context = meta.get("context", "")
        context = context.strip()
        if context:
            if len(context) > max_context_chars:
                context = context[:max_context_chars] + " ..."
            return ex.question.strip() + "\n\nContext (copied from the webpage):\n" + context
        return ex.question
    return ex.question


def build_single_agent(qwen_client: QwenVLClient, dataset: str, max_new_tokens: int, temperature: float, top_p: float) -> QwenAgent:
    # Reuse scanner-like prompt as a general board-aware agent
    role_prompt = (
        "You are a careful multimodal agent. You will add short observations to a shared board "
        "for a few steps, then give one concise final answer."
    )
    return QwenAgent(
        name="BoardAgent",
        role_prompt=role_prompt,
        qwen_client=qwen_client,
        role="scanner",
        dataset=dataset,
        allow_final_answer=True,
        final_answer_step=1,
        default_gen_kwargs={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    )


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model_path)
    split = args.split or default_split_for(args.dataset)
    default_output = Path("results") / f"single_qwen_board_{args.dataset}_{split}.jsonl"
    output_path = Path(args.results_path) if args.results_path else default_output
    resume_path = Path(args.resume_from) if args.resume_from else output_path

    if args.resume_from and not Path(args.resume_from).exists():
        print(f"[ERROR] Resume file not found: {args.resume_from}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.shard_id >= args.num_shards:
        print(f"[ERROR] shard-id {args.shard_id} is out of range for num-shards {args.num_shards}")
        return

    torch.manual_seed(args.seed)

    existing_ids = set()
    num_exact_done = 0
    num_loose_done = 0
    total_f1_done = 0.0
    n_done = 0

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
                pred_existing = rec.get("pred_answer", "")
                if args.skip_existing and pred_existing and not is_note_like(pred_existing):
                    existing_ids.add(ex_id)
                elif not args.skip_existing:
                    existing_ids.add(ex_id)

                if ex_id in existing_ids:
                    n_done += 1
                    if rec.get("correct_exact"):
                        num_exact_done += 1
                    if rec.get("correct_loose"):
                        num_loose_done += 1
                    if args.dataset == "vqaonline" and isinstance(rec.get("score_f1"), (int, float)):
                        total_f1_done += rec["score_f1"]

        print(
            f"[RESUME] Found {n_done} existing results in {resume_path}. {len(existing_ids)} unique example ids."
        )

    try:
        examples = load_toy_split(
            dataset=args.dataset,
            split=split,
            max_examples=None if args.only_ids else args.max_examples,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Hint: export the requested dataset first with one of the scripts/export_* helpers.")
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
    shard_ids = list(range(num_shards)) if args.run_all_shards else [args.shard_id]
    if not args.run_all_shards and num_shards > 1:
        examples = [ex for idx, ex in enumerate(examples) if idx % num_shards == args.shard_id]
        print(
            f"Sharding enabled: num_shards={num_shards}, shard_id={args.shard_id}. Using {len(examples)} examples in this run."
        )
    else:
        print(
            f"Sharding: num_shards={num_shards}, running shard ids {shard_ids}. Total loaded examples: {len(examples)}"
        )

    # Dataset-specific default max_new_tokens if user did not set
    if args.max_new_tokens is None:
        args.max_new_tokens = 48 if args.dataset == "vqaonline" else 64

    n_total_planned = len(examples)
    total_examples = len(examples)
    total_processed = n_done

    client = QwenVLClient(
        model_path=model_path,
        device=args.device,
        dtype="auto",
    )

    # Build single agent and controller
    agent = build_single_agent(
        qwen_client=client,
        dataset=args.dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    controller = CoSeeController(agents=[agent], max_steps=args.max_steps)

    total = n_done
    num_exact = num_exact_done
    num_loose = num_loose_done
    f1_sum = total_f1_done

    note_like_count = 0
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
            print(f"[INFO] Running shard {shard_id}/{num_shards - 1}")
            for idx, ex in enumerate(examples):
                if num_shards > 1 and idx % num_shards != shard_id:
                    continue
                if ex.id in existing_ids:
                    continue

                images = [Image.open(ROOT / p).convert("RGB") for p in ex.image_paths]
                meta = getattr(ex, "meta", {}) or {}
                question_input = build_input_text_for_example(ex)

                if args.log_compute:
                    client.reset_compute()

                # Run single-agent controller
                result = controller.run(
                    images=images,
                    question=question_input,
                )
                if isinstance(result, tuple):
                    _, final_board = result
                elif isinstance(result, dict):
                    final_board = result.get("board")
                else:
                    raise ValueError("Unexpected controller.run return type")

                board_summary = final_board.to_text(max_cells_per_page=8, max_total_chars=1500)

                prompt = (
                    f"Question: {question_input}\n\n"
                    f"Board notes:\n{board_summary}\n\n"
                )
                qtype = (meta.get("Question Type") or meta.get("question_type") or "").strip().lower()
                if args.dataset == "chartqapro" and qtype == "multi choice":
                    prompt += "Answer with ONLY the option letter (A/B/C/D)."
                else:
                    prompt += "Answer with ONLY the short answer. No explanation."

                forced_answer_raw = client.generate(
                    images=images,
                    question=prompt,
                    board_text=None,
                    role_prompt=None,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                final_answer = forced_answer_raw.strip()
                final_answer_source = "forced_final_gen_always"

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

                if is_note_like(final_answer):
                    note_like_count += 1
                    print(f"[WARN] Note-like pred_answer for id={ex.id}: {final_answer[:200]}")

                gold_norm = normalize_answer(ex.answer)
                pred_norm = normalize_answer(final_answer or "")

                if args.dataset == "vqaonline":
                    score_f1 = compute_vqaonline_f1(ex.answer, final_answer or "")
                    correct_exact = score_f1 >= 0.5
                    correct_loose = score_f1 >= 0.3
                    f1_sum += score_f1
                else:
                    correct_exact = pred_norm == gold_norm
                    correct_loose = is_loose_match(gold_norm, pred_norm)

                total += 1
                num_exact += int(correct_exact)
                num_loose += int(correct_loose)

                total_processed += 1
                if total_processed % 5 == 0:
                    print(
                        f"[{total_processed}/{total_examples}] id={ex.id} dataset={args.dataset}/{split}"
                    )

                record: Dict[str, Any] = {
                    "id": ex.id,
                    "dataset": ex.dataset,
                    "split": ex.split,
                    "question": ex.question,
                    "gold_answer": ex.answer,
                    "pred_answer": final_answer,
                    "final_answer_raw": forced_answer_raw,
                    "gold_norm": gold_norm,
                    "pred_norm": pred_norm,
                    "correct_exact": correct_exact,
                    "correct_loose": correct_loose,
                    "meta": ex.meta or {},
                    "board_summary": board_summary,
                    "final_answer_source": final_answer_source,
                }
                if args.dataset == "vqaonline":
                    record["score_f1"] = score_f1
                if args.log_compute:
                    record["num_generate_calls"] = compute_breakdown[0]["calls"]
                    record["gen_tokens_total"] = compute_breakdown[0]["gen_tokens_total"]
                    record["compute_version"] = "v1"
                    record["compute_breakdown"] = compute_breakdown
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing_ids.add(ex.id)

    if total == 0:
        print("[WARN] No examples were evaluated (total==0).")
        return

    exact_acc = num_exact / total
    loose_acc = num_loose / total
    if args.dataset == "vqaonline":
        avg_f1 = f1_sum / total
        print(
            f"Finished {total} examples from {args.dataset}/{split}.\n"
            f"F1 average: {avg_f1:.3f}\n"
            f"Strict (F1>=0.5): {exact_acc:.3f} ({num_exact} / {total})\n"
            f"Relaxed (F1>=0.3): {loose_acc:.3f} ({num_loose} / {total})"
        )
    else:
        print(
            f"Finished {total} examples from {args.dataset}/{split}.\n"
            f"Exact match accuracy: {exact_acc:.3f} ({num_exact} / {total})\n"
            f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})"
        )

    if args.log_compute and compute_examples > 0:
        mean_calls = compute_calls_total / compute_examples
        mean_tokens = compute_gen_tokens_total / compute_examples
        print(
            f"Compute summary (v1): mean generate calls={mean_calls:.2f}, "
            f"mean gen tokens={mean_tokens:.1f} over {compute_examples} examples."
        )


if __name__ == "__main__":
    main()
