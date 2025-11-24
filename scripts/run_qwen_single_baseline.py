from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import torch

from cosee.data.datasets import load_toy_split
from cosee.models.qwen_vl_wrapper import QwenVLClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-model Qwen baseline on toy datasets.")
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
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
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


def load_images(image_paths: List[str]) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for p in image_paths:
        path = Path(p)
        img = Image.open(path).convert("RGB")
        imgs.append(img)
    return imgs


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model_path)
    split = args.split or default_split_for(args.dataset)
    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"baseline_qwen_single_{args.dataset}_{split}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    examples = load_toy_split(
        dataset=args.dataset,
        split=split,
        max_examples=args.max_examples,
    )
    print(f"Loaded {len(examples)} examples from {args.dataset}/{split}")

    client = QwenVLClient(
        model_path=model_path,
        device=args.device,
        dtype="auto",
    )

    role_prompt = (
        "You are a careful multimodal assistant that answers questions about one or more images. "
        "Answer concisely, without explanation."
    )

    total = 0
    num_exact = 0
    num_loose = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, ex in enumerate(examples):
            images = load_images(ex.image_paths)
            question_text = (
                f"Question: {ex.question}\n"
                "Please answer with a short phrase or number, without explanation."
            )

            pred_text = client.generate(
                images=images,
                question=question_text,
                board_text=None,
                role_prompt=role_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            ).strip()

            gold_norm = normalize_answer(ex.answer)
            pred_norm = normalize_answer(pred_text)

            correct_exact = pred_norm == gold_norm
            correct_loose = bool(gold_norm) and (gold_norm in pred_norm or pred_norm in gold_norm)

            total += 1
            num_exact += int(correct_exact)
            num_loose += int(correct_loose)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(examples):
                print(
                    f"[{idx+1}/{len(examples)}] id={ex.id} exact={int(correct_exact)} loose={int(correct_loose)}"
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
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    exact_acc = num_exact / total if total else 0.0
    loose_acc = num_loose / total if total else 0.0
    print(
        f"Finished {total} examples from {args.dataset}/{split}.\n"
        f"Exact match accuracy: {exact_acc:.3f} ({num_exact} / {total})\n"
        f"Loose match accuracy: {loose_acc:.3f} ({num_loose} / {total})"
    )


if __name__ == "__main__":
    main()
