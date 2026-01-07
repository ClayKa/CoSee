#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Cost-normalized analysis (no reruns).
- Reads JSONL files under results/final_raw/
- Computes GenTokens(final) using Qwen tokenizer on pred_answer
- Writes:
  1) token_cost_report.csv  (per-run token stats)
  2) vqaonline_f1_by_len_bins.csv (per-run, per-length-bin F1 stats)
  3) vqaonline_f1_vs_len.pdf/.png (figure for main paper)
Optionally writes simple LaTeX tables.

Usage:
  python scripts/phase1_cost_analysis.py \
    --final-raw-dir results/final_raw \
    --model-path ./models/Qwen3-VL-4B-Instruct \
    --out-dir analysis \
    --fig-dir figs

Notes:
- This script uses GenTokens(final) = #tokens in pred_answer only.
- For VQAonline it additionally analyzes F1 vs output length bins.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# matplotlib is optional but recommended for the main-paper figure
import matplotlib.pyplot as plt

from transformers import AutoTokenizer


# -----------------------------
# Helpers: method/dataset labels
# -----------------------------
def infer_method_from_filename(name: str) -> str:
    n = name.lower()
    if n.startswith("baseline_qwen_single_"):
        return "baseline_single"
    if n.startswith("single_qwen_board_"):
        return "single_board"
    if n.startswith("cosee_two_qwen_"):
        return "cosee_two"
    if n.startswith("cosee_three_qwen_"):
        return "cosee_three"
    # fallback
    return "unknown"


def infer_dataset_from_record_or_filename(rec: Dict[str, Any], fname: str) -> str:
    if isinstance(rec.get("dataset"), str) and rec["dataset"].strip():
        return rec["dataset"].strip().lower()
    fn = fname.lower()
    for key in ["slidevqa", "chartqapro", "vqaonline"]:
        if key in fn:
            return key
    return "unknown"


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def percentile_int(arr: np.ndarray, p: float) -> int:
    if arr.size == 0:
        return 0
    return int(np.percentile(arr, p))


# -----------------------------
# Core stats
# -----------------------------
@dataclass
class TokenStats:
    n: int
    mean: float
    p50: int
    p90: int
    p95: int
    maxv: int


def compute_token_stats(lengths: List[int]) -> TokenStats:
    a = np.array(lengths, dtype=int)
    if a.size == 0:
        return TokenStats(n=0, mean=0.0, p50=0, p90=0, p95=0, maxv=0)
    return TokenStats(
        n=int(a.size),
        mean=float(a.mean()),
        p50=percentile_int(a, 50),
        p90=percentile_int(a, 90),
        p95=percentile_int(a, 95),
        maxv=int(a.max()),
    )


# -----------------------------
# VQAonline bins
# -----------------------------
BIN_EDGES = [0, 16, 32, 64, 128, 256, 10**9]
BIN_LABELS = ["0-16", "17-32", "33-64", "65-128", "129-256", "257+"]

def bin_index(tok_len: int) -> int:
    # tok_len >= 0
    for i in range(len(BIN_EDGES) - 1):
        lo = BIN_EDGES[i]
        hi = BIN_EDGES[i + 1]
        if i == 0:
            # include 0..16
            if lo <= tok_len <= hi:
                return i
        else:
            # 17..32 etc
            if lo < tok_len <= hi:
                return i
    return len(BIN_LABELS) - 1


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-raw-dir", type=str, default="results/final_raw",
                    help="Directory containing final raw JSONL outputs.")
    ap.add_argument("--model-path", type=str, default="./models/Qwen3-VL-4B-Instruct",
                    help="Tokenizer/model path for Qwen3-VL-4B-Instruct.")
    ap.add_argument("--out-dir", type=str, default="analysis",
                    help="Directory to write CSV (and optional TeX) outputs.")
    ap.add_argument("--fig-dir", type=str, default="figs",
                    help="Directory to write figures (PDF/PNG).")
    ap.add_argument("--write-tex", action="store_true",
                    help="Also write simple LaTeX tables for Overleaf.")
    args = ap.parse_args()

    final_raw_dir = Path(args.final_raw_dir)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not final_raw_dir.exists():
        raise FileNotFoundError(f"final_raw_dir not found: {final_raw_dir.resolve()}")

    jsonl_files = sorted([p for p in final_raw_dir.glob("*.jsonl") if p.is_file()])
    if not jsonl_files:
        raise RuntimeError(f"No .jsonl files found under: {final_raw_dir.resolve()}")

    print(f"[INFO] Found {len(jsonl_files)} JSONL files under {final_raw_dir}")

    print(f"[INFO] Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Per-run token stats
    token_cost_rows: List[Dict[str, Any]] = []

    # VQAonline bin stats: dict[(method, runfile)] -> bin accumulators
    # accum: {bin_label: {"count": int, "sum_f1": float, "sum_strict": int, "sum_relaxed": int}}
    vqa_bin_acc: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}

    for fp in jsonl_files:
        method = infer_method_from_filename(fp.name)
        # read all records
        lengths: List[int] = []
        datasets_seen = set()
        n_missing = 0

        # vqaonline per-file accum
        key = (method, fp.name)

        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ds = infer_dataset_from_record_or_filename(rec, fp.name)
                datasets_seen.add(ds)

                pred = rec.get("pred_answer", "")
                if not isinstance(pred, str) or not pred.strip():
                    n_missing += 1
                    continue

                tok_len = len(tokenizer.encode(pred, add_special_tokens=False))
                lengths.append(tok_len)

                # VQAonline binning
                if ds == "vqaonline":
                    f1 = safe_float(rec.get("score_f1"))
                    if f1 is None:
                        # if missing, skip bin stats
                        continue
                    strict = 1 if f1 >= 0.5 else 0
                    relaxed = 1 if f1 >= 0.3 else 0

                    bi = bin_index(tok_len)
                    bl = BIN_LABELS[bi]

                    if key not in vqa_bin_acc:
                        vqa_bin_acc[key] = {lbl: {"count": 0.0, "sum_f1": 0.0, "sum_strict": 0.0, "sum_relaxed": 0.0}
                                            for lbl in BIN_LABELS}
                    vqa_bin_acc[key][bl]["count"] += 1.0
                    vqa_bin_acc[key][bl]["sum_f1"] += float(f1)
                    vqa_bin_acc[key][bl]["sum_strict"] += float(strict)
                    vqa_bin_acc[key][bl]["sum_relaxed"] += float(relaxed)

        stats = compute_token_stats(lengths)
        # determine dataset label for the run
        ds_label = "unknown"
        if len(datasets_seen) == 1:
            ds_label = next(iter(datasets_seen))
        else:
            # fallback to filename inference
            ds_label = infer_dataset_from_record_or_filename({}, fp.name)

        token_cost_rows.append({
            "run_file": fp.name,
            "dataset": ds_label,
            "method": method,
            "n": stats.n,
            "missing_pred_answer": n_missing,
            "gen_tokens_final_mean": round(stats.mean, 3),
            "gen_tokens_final_p50": stats.p50,
            "gen_tokens_final_p90": stats.p90,
            "gen_tokens_final_p95": stats.p95,
            "gen_tokens_final_max": stats.maxv,
        })

        print(f"[OK] {fp.name:55s} dataset={ds_label:9s} method={method:13s} "
              f"n={stats.n:4d} mean={stats.mean:.1f} p90={stats.p90} max={stats.maxv} missing={n_missing}")

    # Write token_cost_report.csv
    token_csv = out_dir / "token_cost_report.csv"
    with token_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(token_cost_rows[0].keys()))
        writer.writeheader()
        writer.writerows(token_cost_rows)
    print(f"[WRITE] {token_csv}")

    # Write vqaonline_f1_by_len_bins.csv
    vqa_rows: List[Dict[str, Any]] = []
    for (method, runfile), bins in vqa_bin_acc.items():
        for bl in BIN_LABELS:
            c = int(bins[bl]["count"])
            if c == 0:
                mean_f1 = 0.0
                strict_rate = 0.0
                relaxed_rate = 0.0
            else:
                mean_f1 = bins[bl]["sum_f1"] / c
                strict_rate = bins[bl]["sum_strict"] / c
                relaxed_rate = bins[bl]["sum_relaxed"] / c
            vqa_rows.append({
                "run_file": runfile,
                "method": method,
                "len_bin": bl,
                "count": c,
                "mean_f1": round(float(mean_f1), 6),
                "strict_rate(F1>=0.5)": round(float(strict_rate), 6),
                "relaxed_rate(F1>=0.3)": round(float(relaxed_rate), 6),
            })

    if vqa_rows:
        vqa_csv = out_dir / "vqaonline_f1_by_len_bins.csv"
        with vqa_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(vqa_rows[0].keys()))
            writer.writeheader()
            writer.writerows(vqa_rows)
        print(f"[WRITE] {vqa_csv}")
    else:
        vqa_csv = None
        print("[WARN] No VQAonline rows found for binning (missing dataset=VQAonline or missing score_f1).")

    # Plot: VQAonline mean F1 vs length bins (for main paper)
    if vqa_rows:
        # pick representative runs: baseline_single, single_board, cosee_two (ignore cosee_three unless present)
        # If multiple runs per method exist, we use the first one encountered.
        # This keeps the main-paper figure simple.
        method_priority = ["baseline_single", "single_board", "cosee_two", "cosee_three"]
        chosen_run_by_method: Dict[str, str] = {}
        for r in vqa_rows:
            m = r["method"]
            if m not in chosen_run_by_method:
                chosen_run_by_method[m] = r["run_file"]

        # build series
        x = np.arange(len(BIN_LABELS))
        plt.figure()
        for m in method_priority:
            if m not in chosen_run_by_method:
                continue
            rf = chosen_run_by_method[m]
            series = [rr for rr in vqa_rows if rr["method"] == m and rr["run_file"] == rf]
            series_map = {rr["len_bin"]: rr for rr in series}
            y = [series_map[bl]["mean_f1"] for bl in BIN_LABELS]
            plt.plot(x, y, marker="o", label=m)

        plt.xticks(x, BIN_LABELS, rotation=0)
        plt.xlabel("Final answer length (GenTokens(final) bins)")
        plt.ylabel("Mean token-F1 (VQAonline)")
        plt.title("VQAonline: F1 vs final answer length")
        plt.legend()
        plt.tight_layout()

        fig_pdf = fig_dir / "vqaonline_f1_vs_len.pdf"
        fig_png = fig_dir / "vqaonline_f1_vs_len.png"
        plt.savefig(fig_pdf)
        plt.savefig(fig_png, dpi=200)
        plt.close()
        print(f"[WRITE] {fig_pdf}")
        print(f"[WRITE] {fig_png}")

    # Optional: write simple LaTeX tables (for Overleaf)
    if args.write_tex:
        # Token cost table (simple)
        tex_path = out_dir / "tab_token_cost_auto.tex"
        # group rows by dataset for nicer layout
        rows_sorted = sorted(token_cost_rows, key=lambda r: (r["dataset"], r["method"], r["run_file"]))
        with tex_path.open("w", encoding="utf-8") as f:
            f.write("% Auto-generated by scripts/phase1_cost_analysis.py\n")
            f.write("\\begin{table}[t]\n\\centering\n")
            f.write("\\small\n")
            f.write("\\caption{Final-answer token length statistics (GenTokens(final)).}\n")
            f.write("\\label{tab:token_cost}\n")
            f.write("\\begin{tabular}{l l r r r r}\n")
            f.write("\\toprule\n")
            f.write("Dataset & Method & Mean & P50 & P90 & Max \\\\\n")
            f.write("\\midrule\n")
            for r in rows_sorted:
                f.write(f"{r['dataset']} & {r['method']} & {r['gen_tokens_final_mean']} & "
                        f"{r['gen_tokens_final_p50']} & {r['gen_tokens_final_p90']} & {r['gen_tokens_final_max']} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        print(f"[WRITE] {tex_path}")

        # VQA bin table (simple, long; recommend appendix)
        if vqa_rows:
            tex_path2 = out_dir / "tab_vqaonline_f1_by_len_bins_auto.tex"
            with tex_path2.open("w", encoding="utf-8") as f:
                f.write("% Auto-generated by scripts/phase1_cost_analysis.py\n")
                f.write("\\begin{table*}[t]\n\\centering\n")
                f.write("\\small\n")
                f.write("\\caption{VQAonline performance by final-answer length bins.}\n")
                f.write("\\label{tab:vqa_len_bins}\n")
                f.write("\\begin{tabular}{l l l r r r r}\n")
                f.write("\\toprule\n")
                f.write("Run & Method & Bin & Count & Mean F1 & Strict & Relaxed \\\\\n")
                f.write("\\midrule\n")
                for rr in sorted(vqa_rows, key=lambda x: (x["method"], x["run_file"], x["len_bin"])):
                    f.write(f"{rr['run_file']} & {rr['method']} & {rr['len_bin']} & {rr['count']} & "
                            f"{rr['mean_f1']} & {rr['strict_rate(F1>=0.5)']} & {rr['relaxed_rate(F1>=0.3)']} \\\\\n")
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table*}\n")
            print(f"[WRITE] {tex_path2}")

    print("[DONE] Phase 1 analysis complete.")


if __name__ == "__main__":
    main()
