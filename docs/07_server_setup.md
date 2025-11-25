--

# 07_server_setup.md — CoSee Server Setup & Migration Guide

This document describes how to set up and validate a GPU server environment for the CoSee project.
The goal of Phase 1 is:

> Reproduce the local CoSee environment on a 4090 server and confirm that GPU-based demos and toy experiments run end-to-end.

---

## 1. Assumptions and Target Layout

### 1.1 Hardware and OS assumptions

We assume a machine with roughly the following specs:

* 1× NVIDIA RTX 4090 (24 GB VRAM)
* ≥ 20 CPU cores
* ≥ 90 GB RAM
* OS: Linux (Ubuntu or similar)
* You have SSH access and sudo privileges (or can ask the admin to install system packages).

### 1.2 Directory layout (server side)

We standardize the following layout to keep scripts and docs consistent:

* Project root:

  ```text
  ~/projects/CoSee
  ```

* Data root (mounted data disk):

  ```text
  /data/cosee
  ```

* Under `/data/cosee`:

  ```text
  /data/cosee/
    data/        # datasets and derived files
    models/      # local model weights (e.g., Qwen3-VL-4B-Instruct)
    hf_cache/    # Hugging Face datasets / models cache
  ```

If your cloud provider uses a different mount point, adjust the paths in this document and in your environment variables accordingly.

---

## 2. System Preparation

> This section is usually done once per machine.

### 2.1 Mount and size the data disk

1. Ensure there is a data disk with at least 200 GB available.
2. Mount it to `/data` (or another agreed-upon mount point).
3. Create the CoSee subdirectories:

   ```bash
   sudo mkdir -p /data/cosee/{data,models,hf_cache}
   sudo chown -R "$USER":"$USER" /data/cosee
   ```

### 2.2 Install basic tools

Install commonly used CLI tools:

```bash
sudo apt-get update
sudo apt-get install -y git tmux screen htop unzip wget curl
```

Optional but recommended:

* Set up your preferred shell (zsh or bash) and add useful aliases (e.g., `cconda`, `cproj`) in `~/.bashrc` or `~/.zshrc`.

Example aliases:

```bash
echo 'alias cproj="cd ~/projects/CoSee"' >> ~/.bashrc
echo 'alias cconda="conda activate cosee"' >> ~/.bashrc
```

Reload your shell:

```bash
source ~/.bashrc
```

---

## 3. Conda Environment and Python Dependencies

### 3.1 Ensure Conda is installed

If Conda (or Miniconda) is not installed, install Miniconda using the official installer from Anaconda or Miniconda (this step is provider-specific and may already be done by your platform).

After installation, make sure `conda` is on your PATH:

```bash
conda --version
```

### 3.2 Create the CoSee Conda environment

From any directory:

```bash
conda create -n cosee python=3.10 -y
conda activate cosee
```

### 3.3 Clone the CoSee repository

Create the project root and clone:

```bash
mkdir -p ~/projects
cd ~/projects
git clone <YOUR_COSEE_REPO_URL> CoSee
cd CoSee
```

The tree should look roughly like:

```text
CoSee/
  cosee/
  scripts/
  docs/
  data/          # may be initially empty on server
  models/        # may be initially empty on server
  requirements.txt
  ...
```

### 3.4 Install Python dependencies

With the `cosee` environment activated and inside `~/projects/CoSee`:

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not contain a GPU build of PyTorch, install a CUDA-enabled wheel explicitly (matching the server CUDA version), e.g.:

```bash
# Example; adjust version and CUDA tag to your environment
pip install "torch==2.3.0+cu121" --index-url https://download.pytorch.org/whl/cu121
```

Check that PyTorch sees the GPU:

```bash
python - << 'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
EOF
```

---

## 4. Environment Variables and Caches

We want all Hugging Face downloads to go into `/data/cosee/hf_cache`, and models to live under `/data/cosee/models`.

Add the following to your shell config (e.g., `~/.bashrc`):

```bash
# CoSee paths
export COSEE_ROOT="$HOME/projects/CoSee"
export COSEE_DATA_ROOT="/data/cosee/data"
export COSEE_MODELS_ROOT="/data/cosee/models"
export HF_HOME="/data/cosee/hf_cache"  # HF cache dir

# (Optional) Shortcuts
alias cproj="cd $COSEE_ROOT"
alias cconda="conda activate cosee"
```

Reload:

```bash
source ~/.bashrc
```

Make sure these directories exist:

```bash
mkdir -p "$COSEE_DATA_ROOT" "$COSEE_MODELS_ROOT" "$HF_HOME"
```

If your code currently uses relative paths (e.g., `./data`, `./models`), you can either:

* Keep the current behavior and symlink server paths into the repo, or
* Update your config to use `COSEE_DATA_ROOT` and `COSEE_MODELS_ROOT`.

Example symlinks from the project root:

```bash
cd "$COSEE_ROOT"
ln -s /data/cosee/data    data
ln -s /data/cosee/models  models
```

---

## 5. Models and Datasets on the Server

### 5.1 Copy or download Qwen3-VL-4B-Instruct

Option A — copy from your local machine using `scp`:

On your local machine:

```bash
# From local CoSee root
scp -r models/Qwen3-VL-4B-Instruct \
    <username>@<server_ip>:/data/cosee/models/
```

On the server, you should then see:

```text
/data/cosee/models/Qwen3-VL-4B-Instruct/...
```

Option B — re-download on the server via Hugging Face (if the scripts support it):

* Configure your HF token:

  ```bash
  huggingface-cli login
  ```

* Run your existing model download script (if you have one), or use `transformers` to load the model once and let it cache into `$HF_HOME`, then optionally move it into `/data/cosee/models`.

### 5.2 Copy or reconstruct datasets

You can either:

* Copy the entire `data/` directory from your local machine, or
* Re-run the dataset download/export scripts on the server.

Option A — copy `data/`:

On local:

```bash
# From local CoSee root
scp -r data \
    <username>@<server_ip>:/data/cosee/
```

On server:

```bash
ls /data/cosee/data
# Expect: slidevqa/, chartqapro/, vqaonline/, hf_cache/ or similar
```

Option B — re-run dataset scripts (recommended if they’re stable):

On server:

```bash
cd "$COSEE_ROOT"
conda activate cosee

# Example commands; adjust to your actual script names
python -m scripts.download_slidevqa_train
python -m scripts.export_slidevqa_toy

python -m scripts.download_chartqapro
python -m scripts.export_chartqapro_toy

python -m scripts.download_vqaonline_raw
python -m scripts.export_vqaonline_toy
```

Then verify paths and samples:

```bash
python -m scripts.tmp_check_data_paths
python -m scripts.inspect_dataset_examples
```

If these scripts succeed and show a few valid examples for each dataset, the data layer is good.

---

## 6. Smoke Tests: GPU Qwen and CoSee Pipeline

Once the environment, models, and data are in place, run three levels of tests.

### 6.1 Qwen client demo (single image, GPU)

From the project root:

```bash
cd "$COSEE_ROOT"
conda activate cosee

python -m scripts.dev_qwen_client_demo --device cuda
```

Check:

* The script runs without errors.
* The model uses the GPU (`nvidia-smi` should show activity).
* The output is a reasonable caption or answer for the demo image.

If you don’t have a `--device` flag, ensure the script uses CUDA when available (e.g., `device="cuda"` in `QwenVLClient`).

### 6.2 QwenAgent + Board demo

Run the agent demo:

```bash
python -m scripts.run_qwen_agent_demo --device cuda
```

Expected behavior:

* The script loads the Qwen model on GPU.
* It constructs a small Board and uses `QwenAgent` to write at least one note.
* You see printed output showing the agent’s note and any debug information.

### 6.3 CoSee controller demo (Qwen + Dummy)

Run the combined demo:

```bash
python -m scripts.run_cosee_qwen_plus_dummy --device cuda
```

Expected behavior:

* The script initializes a Board and two agents (e.g., QwenScanner + DummyCrossChecker).
* The `CoSeeController` runs a short episode.
* The script prints:

  * The final answer (even if placeholder).
  * A serialized or summarized Board trace.

If all three demos run successfully on GPU, the core CoSee framework is operational on the server.

---

## 7. Dataset-Level Toy Experiments (Phase 1 Completion Check)

To fully consider Phase 1 done, we want to verify that **the same scripts used on your local machine also run on the server with GPU**, at least on very small toy splits.

### 7.1 Single-model baseline (small N)

From project root:

```bash
cd "$COSEE_ROOT"
conda activate cosee

# Adjust N to small numbers for smoke testing, e.g., 4 or 10
python -m scripts.run_qwen_single_baseline \
  --dataset chartqapro \
  --split test \
  --max-examples 4 \
  --device cuda
```

Check:

* The script finishes without error.
* It writes a JSONL result file under `results/` (exact path depends on your script).
* Accuracy fields are present (exact / loose, or similar).

You can repeat the same for SlideVQA and VQAonline once ChartQAPro works.

### 7.2 CoSee multi-agent runner (small N)

Run a tiny CoSee experiment:

```bash
python -m scripts.run_cosee_on_dataset \
  --dataset chartqapro \
  --split test \
  --max-examples 4 \
  --agent-config two_qwen \
  --max-steps 3 \
  --device cuda
```

Check:

* The script finishes successfully.
* It writes a `results/...` file containing:

  * Input metadata;
  * Predictions;
  * `board_summary` or equivalent trace field.
* GPU memory usage and runtime per sample are reasonable (you can eyeball via `nvidia-smi`).

If both baseline and CoSee runners work on small N for at least one dataset, the environment is ready for the Phase 2 toy experiments.

---

## 8. What Counts as “Phase 1 Done”?

You can mark Phase 1 (server migration) as complete when all of the following hold:

* [ ] `/data/cosee` is mounted and writable, with `data/`, `models/`, `hf_cache/` subdirectories.
* [ ] `conda activate cosee` works and `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
* [ ] The CoSee repo is cloned under `~/projects/CoSee` and Python dependencies are installed.
* [ ] `HF_HOME` and CoSee path environment variables are set (and/or symlinks from project root to `/data/cosee/...` exist).
* [ ] Qwen3-VL-4B-Instruct weights are available under `/data/cosee/models` (or a known path).
* [ ] The three smoke tests all pass on GPU:

  * `dev_qwen_client_demo`
  * `run_qwen_agent_demo`
  * `run_cosee_qwen_plus_dummy`
* [ ] A tiny baseline run and a tiny CoSee run on at least one dataset (e.g., ChartQAPro) finish successfully and write JSONL outputs under `results/`.

Once all boxes are checked, you can safely tell any assistant (e.g., Codex or another ChatGPT session):

> “Follow `docs/07_server_setup.md` to reproduce the environment, then move on to Phase 2 toy-scale experiments.”

This keeps the development and collaboration experience consistent with what you have locally, while making GPU experiments and future scaling straightforward.
