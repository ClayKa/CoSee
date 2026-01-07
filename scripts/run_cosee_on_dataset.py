from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from cosee.agents import QwenAgent
from cosee.controller import CoSeeController
from cosee.data.datasets import load_toy_split, ROOT as DATA_ROOT
from cosee.metrics import compute_vqaonline_f1
from cosee.models.qwen_vl_wrapper import QwenVLClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoSee multi-agent on a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["slidevqa", "chartqapro", "vqaonline"],
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=50)
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
        help="Path to Qwen3-VL-4B-Instruct; defaults to COSEE_MODEL_PATH or ./models/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit path for the output JSONL results file.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-images-per-example", type=int, default=None)
    parser.add_argument(
        "--log-compute",
        action="store_true",
        help="If set, log per-example compute proxies (generate calls and token counts).",
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        default="two_qwen",
        choices=["single_qwen_board", "two_qwen", "two_agent", "three_qwen"],
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
        help="Shard id (0-based). Only examples with idx %% num_shards == shard_id are processed unless --run-all-shards.",
    )
    parser.add_argument(
        "--run-all-shards",
        action="store_true",
        help="If set, iterate over all shard IDs [0..num_shards-1] in this process, reusing the same model.",
    )
    return parser.parse_args()


def default_split_for(dataset: str) -> str:
    if dataset == "slidevqa":
        return "train"
    return "test"


def resolve_model_path(path_arg: Optional[str]) -> str:
    if path_arg:
        return path_arg
    return os.environ.get("COSEE_MODEL_PATH", "./models/Qwen3-VL-4B-Instruct")


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_chartqapro_answerable_from_record(rec: Dict[str, Any]) -> bool:
    gold_norm = (rec.get("gold_norm") or "").strip().lower()
    raw_answer = (rec.get("gold_answer") or rec.get("answer") or "").strip().lower()
    return not (gold_norm == "unanswerable" or raw_answer == "unanswerable")


def build_input_text_for_example(ex, max_context_chars: int = 1500) -> str:
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


def read_only_ids(path: str) -> List[str]:
    ids: List[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def collect_unique_clients(agents: List[QwenAgent]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    for agent in agents:
        client = getattr(agent, "qwen_client", None)
        if client is None:
            continue
        key = id(client)
        if key not in mapping:
            mapping[key] = {"client": client, "names": [agent.name]}
        else:
            mapping[key]["names"].append(agent.name)
    return mapping


def reset_compute_for_agents(agents: List[QwenAgent]) -> None:
    for entry in collect_unique_clients(agents).values():
        entry["client"].reset_compute()


def get_compute_for_agents(agents: List[QwenAgent]) -> Dict[str, Any]:
    breakdown: List[Dict[str, Any]] = []
    total_calls = 0
    total_tokens = 0
    for entry in collect_unique_clients(agents).values():
        compute = entry["client"].get_compute()
        name = ",".join(entry["names"])
        breakdown.append(
            {
                "name": name,
                "calls": compute["calls"],
                "gen_tokens_total": compute["gen_tokens_total"],
            }
        )
        total_calls += compute["calls"]
        total_tokens += compute["gen_tokens_total"]
    return {
        "calls": total_calls,
        "gen_tokens_total": total_tokens,
        "breakdown": breakdown,
    }


def build_controller(
    qwen_client: QwenVLClient,
    max_steps: int,
    config_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> CoSeeController:
    if config_name == "two_agent":
        config_name = "two_qwen"
    ROLE_PROMPT_SINGLE = (
        "You are a careful multimodal reasoner. You can add short observations to the shared board first, "
        "then give a single concise final answer to the question. Use the board to keep track of important findings "
        "from the images and avoid repeating yourself."
    )
    ROLE_PROMPT_SCANNER = (
        "You are a scanning agent. Quickly skim the images and write 1–2 concise observations that may be useful "
        "for answering the question. Do not answer the question. Focus on key text, numbers, and visual structure."
    )
    ROLE_PROMPT_DETAIL = (
        "You are a detail-reading agent. Focus on reading fine-grained text, numbers, labels, and legends that are "
        "directly relevant to the question. Add precise observations to the shared board. Do not answer the question directly."
    )
    ROLE_PROMPT_CROSSCHECKER = (
        "You are a cross-checking agent. Read the shared board notes and the images, then give a single concise final answer "
        "to the question. Use the board as your evidence; do not restate all notes, just answer."
    )

    def qwen_agent(name: str, prompt: str, allow_final: bool, final_step: int = 1) -> QwenAgent:
        return QwenAgent(
            name=name,
            role_prompt=prompt,
            qwen_client=qwen_client,
            allow_final_answer=allow_final,
            final_answer_step=final_step,
            default_gen_kwargs={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )

    if config_name == "single_qwen_board":
        agent = qwen_agent("QwenSingle", ROLE_PROMPT_SINGLE, allow_final=True, final_step=1)
        return CoSeeController(agents=[agent], max_steps=max_steps)

    if config_name == "two_qwen":
        scanner = qwen_agent("QwenScanner", ROLE_PROMPT_SCANNER, allow_final=False)
        cross_checker = qwen_agent("QwenCrossChecker", ROLE_PROMPT_CROSSCHECKER, allow_final=True, final_step=0)
        return CoSeeController(agents=[scanner, cross_checker], max_steps=max_steps)

    if config_name == "three_qwen":
        scanner = qwen_agent("QwenScanner", ROLE_PROMPT_SCANNER, allow_final=False)
        detail = qwen_agent("QwenDetailReader", ROLE_PROMPT_DETAIL, allow_final=False)
        cross_checker = qwen_agent("QwenCrossChecker", ROLE_PROMPT_CROSSCHECKER, allow_final=True, final_step=1)
        return CoSeeController(agents=[scanner, detail, cross_checker], max_steps=max_steps)

    raise ValueError(f"Unknown agent configuration: {config_name}")


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model_path)
    split = args.split or default_split_for(args.dataset)

    default_output = Path("results") / f"cosee_{args.agent_config}_{args.dataset}_{split}.jsonl"
    output_path = Path(args.results_path) if args.results_path else (Path(args.output) if args.output else default_output)
    resume_path = Path(args.resume_from) if args.resume_from else output_path

    if args.resume_from and not Path(args.resume_from).exists():
        print(f"[ERROR] Resume file not found: {args.resume_from}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.shard_id >= args.num_shards:
        print(f"[ERROR] shard-id {args.shard_id} is out of range for num-shards {args.num_shards}")
        return

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
                if (
                    args.dataset == "vqaonline"
                    and "score_f1" in rec
                    and isinstance(rec["score_f1"], (int, float))
                ):
                    total_f1_done += rec["score_f1"]
                if args.dataset == "chartqapro" and is_chartqapro_answerable_from_record(rec):
                    answerable_total_done += 1
                    if rec.get("correct_exact"):
                        answerable_correct_exact_done += 1
                    if rec.get("correct_loose"):
                        answerable_correct_loose_done += 1

        print(
            f"[RESUME] Found {n_done} existing results in {resume_path}. "
            f"{len(existing_ids)} unique example ids."
        )

    examples = load_toy_split(
        dataset=args.dataset,
        split=split,
        max_examples=None if args.only_ids else args.max_examples,
    )
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
    if args.shard_id < 0 or args.shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id={args.shard_id} for num_shards={num_shards}")

    shard_ids = list(range(num_shards)) if args.run_all_shards else [args.shard_id]
    print(
        f"Sharding: num_shards={num_shards}, running shard ids {shard_ids}. "
        f"Total loaded examples: {len(examples)}"
    )

    n_total_planned = len(examples)

    qwen_client = QwenVLClient(
        model_path=model_path,
        device=args.device,
        dtype="auto",
    )

    total = n_done
    num_exact = num_exact_done
    num_loose = num_loose_done
    f1_sum = total_f1_done
    answerable_total = answerable_total_done
    answerable_correct_exact = answerable_correct_exact_done
    answerable_correct_loose = answerable_correct_loose_done
    total_examples = len(examples)
    total_processed = n_done
    compute_examples = 0
    compute_calls_total = 0
    compute_gen_tokens_total = 0

    mode = "a" if output_path.exists() and resume_path else "w"
    fout = output_path.open(mode, encoding="utf-8")
    if resume_path and resume_path.exists() and output_path != resume_path and mode == "w":
        with resume_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)

    controller = build_controller(
        qwen_client=qwen_client,
        max_steps=args.max_steps,
        config_name=args.agent_config,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for shard_id in shard_ids:
        print(f"[INFO] Running shard {shard_id}/{num_shards - 1}")
        for idx, ex in enumerate(examples):
            if num_shards > 1 and idx % num_shards != shard_id:
                continue
            if ex.id in existing_ids:
                continue

            image_paths = ex.image_paths
            if args.max_images_per_example is not None:
                image_paths = image_paths[: args.max_images_per_example]

            images = [Image.open(DATA_ROOT / p).convert("RGB") for p in image_paths]

            question_input = build_input_text_for_example(ex)
            if args.log_compute:
                reset_compute_for_agents(controller.agents)
            final_answer, final_board = controller.run(
                images=images,
                question=question_input,
            )

            board_summary = final_board.to_text(
                max_cells_per_page=8,
                max_total_chars=1500,
            )

            gold_answer = ex.answer
            gold_norm = normalize_answer(gold_answer)
            pred_norm = normalize_answer(final_answer or "")

            if args.dataset == "vqaonline":
                score_f1 = compute_vqaonline_f1(gold_answer, final_answer or "")
                correct_exact = score_f1 >= 0.5
                correct_loose = score_f1 >= 0.3
                f1_sum += score_f1
            else:
                correct_exact = pred_norm == gold_norm
                correct_loose = correct_exact

            total += 1
            num_exact += int(correct_exact)
            num_loose += int(correct_loose)
            if args.dataset == "chartqapro":
                gold_raw_str = str(gold_answer).strip().lower()
                if gold_raw_str != "unanswerable":
                    answerable_total += 1
                    answerable_correct_exact += int(correct_exact)
                    answerable_correct_loose += int(correct_loose)

            total_processed += 1
            if total_processed % 5 == 0:
                print(
                    f"[{total_processed}/{total_examples}] id={ex.id} dataset={args.dataset}/{split} agent_config={args.agent_config}"
                )

            record = {
                "id": ex.id,
                "dataset": ex.dataset,
                "split": ex.split,
                "question": ex.question,
                "gold_answer": gold_answer,
                "pred_answer": final_answer,
                "gold_norm": gold_norm,
                "pred_norm": pred_norm,
                "correct_exact": bool(correct_exact),
                "correct_loose": bool(correct_loose),
                **({"score_f1": score_f1} if args.dataset == "vqaonline" else {}),
                "agent_config": args.agent_config,
                "max_steps": args.max_steps,
                "num_images": len(image_paths),
                "board_summary": board_summary,
                "meta": ex.meta or {},
            }
            if args.log_compute:
                compute = get_compute_for_agents(controller.agents)
                record["num_generate_calls"] = compute["calls"]
                record["gen_tokens_total"] = compute["gen_tokens_total"]
                record["compute_version"] = "v1"
                record["compute_breakdown"] = compute["breakdown"]
                compute_calls_total += compute["calls"]
                compute_gen_tokens_total += compute["gen_tokens_total"]
                compute_examples += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_ids.add(ex.id)

    fout.close()

    print(
        f"Finished {total} examples from {args.dataset}/{split} with config={args.agent_config}."
    )
    if total == 0:
        print("[WARN] No examples were evaluated (total==0).")
        return

    exact_acc = num_exact / total
    loose_acc = num_loose / total
    if args.dataset == "vqaonline":
        avg_f1 = f1_sum / total
        print(f"F1 average: {avg_f1:.3f}")
        print(f"Strict (F1>=0.5): {exact_acc:.3f} ({num_exact} / {total})")
        print(f"Relaxed (F1>=0.3): {loose_acc:.3f} ({num_loose} / {total})")
    elif args.dataset == "chartqapro":
        print(f"Exact match accuracy: {exact_acc:.3f} ({num_exact} / {total})")
        print(f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})")
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
        print(f"Exact match accuracy: {exact_acc:.3f} ({num_exact} / {total})")
        print(f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})")

    if args.log_compute and compute_examples > 0:
        mean_calls = compute_calls_total / compute_examples
        mean_tokens = compute_gen_tokens_total / compute_examples
        print(
            f"Compute summary (v1): mean generate calls={mean_calls:.2f}, "
            f"mean gen tokens={mean_tokens:.1f} over {compute_examples} examples."
        )


if __name__ == "__main__":
    main()
