from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a formal SlideVQA subset.")
    parser.add_argument("--split", type=str, default="train", help="HF split to load.")
    parser.add_argument(
        "--num-decks", type=int, default=200, help="Number of decks to sample."
    )
    parser.add_argument(
        "--questions-per-deck",
        type=int,
        default=-1,
        help="If >0, sample up to this many QA per deck; if <=0, use all.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/slidevqa",
        help="Output root directory for images/annotations.",
    )
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> Tuple[Path, Path]:
    img_root = out_dir / "images"
    ann_root = out_dir / "annotations"
    img_root.mkdir(parents=True, exist_ok=True)
    ann_root.mkdir(parents=True, exist_ok=True)
    return img_root, ann_root


def save_pages_for_deck(deck_name: str, example, img_root: Path, saved_pages: set) -> List[str]:
    """Save images for all pages in a deck if not already saved."""
    rel_paths: List[str] = []
    for page_idx in range(1, 21):
        field_name = f"page_{page_idx}"
        img = example.get(field_name)
        if img is None:
            continue
        rel_path = f"data/slidevqa/images/{deck_name}/page_{page_idx:02d}.png"
        abs_path = ROOT / rel_path
        page_key = (deck_name, page_idx)
        if page_key not in saved_pages:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(abs_path)
            saved_pages.add(page_key)
        rel_paths.append(rel_path)
    return rel_paths


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    img_root, ann_root = ensure_dirs(out_dir)

    ds = load_dataset(
        "NTT-hil-insight/SlideVQA",
        split=args.split,
        cache_dir=str(ROOT / "data" / "hf_cache" / "slidevqa"),
    )
    page_cols = [f"page_{i}" for i in range(1, 21)]
    ds_meta = ds.remove_columns(page_cols)

    decks: Dict[str, List[int]] = {}
    deck_names_column = ds["deck_name"]
    for idx, deck_name in enumerate(deck_names_column):
        decks.setdefault(deck_name, []).append(idx)
    print(f"Found {len(decks)} decks in split '{args.split}'.")

    deck_names = sorted(decks.keys())
    rng.shuffle(deck_names)
    selected_decks = deck_names[: args.num_decks]

    saved_pages: set = set()
    saved_page_paths_per_deck: Dict[str, List[str]] = {}
    exported = 0
    ann_path = ann_root / "slidevqa_200deck_allq.jsonl"

    total_qas = 0
    for deck_name in selected_decks:
        deck_indices = decks[deck_name]
        if args.questions_per_deck > 0 and len(deck_indices) > args.questions_per_deck:
            total_qas += args.questions_per_deck
        else:
            total_qas += len(deck_indices)

    print(f"Selected {len(selected_decks)} decks, planning to export ~{total_qas} QA pairs.")
    pbar = tqdm(total=total_qas, desc="Exporting SlideVQA examples")
    with ann_path.open("w", encoding="utf-8") as fout:
        for deck_name in selected_decks:
            qa_indices = decks[deck_name]
            if args.questions_per_deck > 0 and len(qa_indices) > args.questions_per_deck:
                qa_indices = rng.sample(qa_indices, args.questions_per_deck)

            if deck_name not in saved_page_paths_per_deck:
                first_qa = ds[qa_indices[0]]
                saved_paths = save_pages_for_deck(deck_name, first_qa, img_root, saved_pages)
                saved_page_paths_per_deck[deck_name] = saved_paths

            for qa_idx in qa_indices:
                qa = ds_meta[qa_idx]
                pages = saved_page_paths_per_deck.get(deck_name, [])
                if not pages:
                    continue

                record = {
                    "id": f"slidevqa_{args.split}_{exported:06d}",
                    "dataset": "slidevqa",
                    "split": args.split,
                    "image_paths": pages,
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "meta": {
                        "deck_name": qa.get("deck_name"),
                        "deck_url": qa.get("deck_url", ""),
                        "qa_id": qa.get("qa_id", -1),
                        "arithmetic_expression": qa.get("arithmetic_expression", "None"),
                        "evidence_pages": qa.get("evidence_pages", []),
                    },
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1
                pbar.update(1)
    pbar.close()

    print(f"Wrote {exported} examples to {ann_path}")
    print(f"Saved {len(saved_pages)} unique pages under {img_root}")


if __name__ == "__main__":
    main()
