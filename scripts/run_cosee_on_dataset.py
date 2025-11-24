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
        "--model-path",
        type=str,
        default=None,
        help="Path to Qwen3-VL-4B-Instruct; defaults to COSEE_MODEL_PATH or ./models/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-images-per-example", type=int, default=None)
    parser.add_argument(
        "--agent-config",
        type=str,
        default="two_qwen",
        choices=["single_qwen_board", "two_qwen", "three_qwen"],
    )
    parser.add_argument("--output", type=str, default=None)
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


def build_controller(
    qwen_client: QwenVLClient,
    max_steps: int,
    config_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> CoSeeController:
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

    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"cosee_{args.agent_config}_{args.dataset}_{split}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_toy_split(
        dataset=args.dataset,
        split=split,
        max_examples=args.max_examples,
    )
    print(f"Loaded {len(examples)} examples from {args.dataset}/{split}")

    qwen_client = QwenVLClient(
        model_path=model_path,
        device=args.device,
        dtype="auto",
    )

    total = 0
    num_exact = 0
    num_loose = 0
    results: List[Dict[str, Any]] = []

    for idx, ex in enumerate(examples):
        image_paths = ex.image_paths
        if args.max_images_per_example is not None:
            image_paths = image_paths[: args.max_images_per_example]

        images = [Image.open(DATA_ROOT / p).convert("RGB") for p in image_paths]

        controller = build_controller(
            qwen_client=qwen_client,
            max_steps=args.max_steps,
            config_name=args.agent_config,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        final_answer, final_board = controller.run(
            images=images,
            question=ex.question,
        )

        board_summary = final_board.to_text(
            max_cells_per_page=8,
            max_total_chars=1500,
        )

        gold_answer = ex.answer
        gold_norm = normalize_answer(gold_answer)
        pred_norm = normalize_answer(final_answer or "")

        correct_exact = pred_norm == gold_norm
        correct_loose = correct_exact

        total += 1
        num_exact += int(correct_exact)
        num_loose += int(correct_loose)

        print(
            f"[{idx+1}/{len(examples)}] id={ex.id} exact={int(correct_exact)} loose={int(correct_loose)}",
            flush=True,
        )

        results.append(
            {
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
                "agent_config": args.agent_config,
                "max_steps": args.max_steps,
                "num_images": len(image_paths),
                "board_summary": board_summary,
                "meta": ex.meta or {},
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        f"Finished {total} examples from {args.dataset}/{split} with config={args.agent_config}."
    )
    if total > 0:
        exact_acc = num_exact / total
        loose_acc = num_loose / total
        print(f"Exact match accuracy: {exact_acc:.3f} ({num_exact} / {total})")
        print(f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})")


if __name__ == "__main__":
    main()
