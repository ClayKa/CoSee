# CoSee

CoSee is a research prototype for studying shared-state collaboration in resource-constrained visual agents. The code implements a small multimodal collaboration loop where role-specialized agents write evidence notes to a shared Board before producing an answer.

The repository is aligned with the paper direction, **Diagnosing Failure Modes of Shared-State Collaboration in Resource-Constrained Visual Agents**, and focuses on auditable intermediate state, bounded generation calls, and reproducible JSONL experiment logs.

## Current Status

Implemented:

- Core Board data structure in `cosee/board.py`
- Agent and Action abstractions in `cosee/agents.py`
- Sequential CoSee controller in `cosee/controller.py`
- Normalized JSONL dataset loader in `cosee/data/datasets.py`
- Lazy-loading Qwen VL wrapper in `cosee/models/qwen_vl_wrapper.py`
- Baseline, single-board, multi-agent, export, and aggregation scripts under `scripts/`

Not included in the repository:

- Dataset files under `data/`
- Local Qwen model weights under `models/`
- Published experiment result files under `results/`

## Installation

```bash
git clone https://github.com/ClayKa/CoSee.git
cd CoSee
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The code uses Python 3.10+ syntax.

## Quick Smoke Test

This command does not require model weights or datasets:

```bash
python -m scripts.run_dummy_cosee
```

Expected behavior: it runs the controller with dummy agents and prints a final answer plus a Board summary.

## Data Layout

All experiment scripts expect normalized JSONL annotations:

```text
data/
  slidevqa/
    images/
    annotations/slidevqa.jsonl
  chartqapro/
    images/
    annotations/chartqapro.jsonl
  vqaonline/
    images/
    annotations/vqaonline.jsonl
```

Each JSONL row follows this schema:

```json
{
  "id": "chartqapro_test_000001",
  "dataset": "chartqapro",
  "split": "test",
  "image_paths": ["data/chartqapro/images/example.png"],
  "question": "What is the highest value?",
  "answer": "42",
  "meta": {}
}
```

Export helpers:

```bash
python -m scripts.export_chartqapro_toy
python -m scripts.export_slidevqa_toy
python -m scripts.export_vqaonline_toy
```

Some exporters download from Hugging Face and require network access plus the `datasets` or `huggingface_hub` packages.

## Model Setup

Set `COSEE_MODEL_PATH` to a local Qwen VL checkpoint directory:

```bash
export COSEE_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
```

You can also pass `--model-path` directly to the run scripts.

## Running Experiments

Single Qwen baseline:

```bash
python -m scripts.run_qwen_single_baseline \
  --dataset chartqapro \
  --split test \
  --max-examples 50 \
  --device cuda \
  --log-compute
```

Two-agent CoSee:

```bash
python -m scripts.run_cosee_on_dataset \
  --dataset chartqapro \
  --split test \
  --agent-config two_qwen \
  --max-steps 3 \
  --max-examples 50 \
  --device cuda \
  --log-compute
```

Single-agent Board variant:

```bash
python -m scripts.run_single_qwen_board \
  --dataset slidevqa \
  --split train \
  --max-examples 50 \
  --device cuda \
  --log-compute
```

Aggregate JSONL results:

```bash
python -m scripts.aggregate_results \
  --mode cosee \
  --dataset chartqapro \
  --inputs results/cosee_two_qwen_chartqapro_test.jsonl
```

## Notes For Reproducibility

- `--log-compute` records generation call counts and generated token counts.
- `--num-shards`, `--shard-id`, and `--run-all-shards` support long runs on larger splits.
- `--resume-from` skips already processed example IDs.
- `data/`, `models/`, and `results/` are intentionally ignored by git.

## License

No license file is currently included. Add one before public redistribution if needed.
