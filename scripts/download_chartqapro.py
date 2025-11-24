from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

# Repo root: scripts/ is directly under root
ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = ROOT / "data" / "hf_cache" / "chartqapro"


def main() -> None:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using HF cache dir: {HF_CACHE_DIR}")

    ds_dict = load_dataset(
        "ahmed-masry/ChartQAPro",
        cache_dir=str(HF_CACHE_DIR),
    )

    print("Available splits:", list(ds_dict.keys()))

    split_priority = ["test", "validation", "dev", "train"]
    chosen_split = None
    for s in split_priority:
        if s in ds_dict:
            chosen_split = s
            break
    if chosen_split is None:
        chosen_split = next(iter(ds_dict.keys()))

    ds = ds_dict[chosen_split]
    print(f"Loaded split='{chosen_split}' with {len(ds)} examples.")
    print("Columns:", ds.column_names)


if __name__ == "__main__":
    main()
