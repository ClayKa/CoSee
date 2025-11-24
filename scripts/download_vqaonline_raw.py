from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "ChongyanChen/VQAonline"
RAW_DIR = ROOT / "data" / "hf_raw" / "vqaonline"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        cache_dir=str(RAW_DIR),
        allow_patterns=["train.json", "val.json", "test.json"],
    )

    print(f"Snapshot path: {snapshot_path}")
    for fname in ["train.json", "val.json", "test.json"]:
        fpath = Path(snapshot_path) / fname
        status = "OK" if fpath.exists() else "MISSING"
        print(f"  {fname}: {status}")


if __name__ == "__main__":
    main()
