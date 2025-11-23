# 04 – Development Environment Setup

This document describes how to set up the development and runtime environment for **CoSee** on:

- A local **macOS** machine (for coding, small demos, and smoke tests),
- A remote **GPU server** (for full experiments).

The goals are:

- Use a **reproducible** Python environment across local and server,
- Load **Qwen3-VL-4B-Instruct** from a **local directory** (no online fetch in experiments),
- Keep all steps explicit so they can be followed by a single developer.

---

## 1. Assumptions and prerequisites

We assume the following:

- You have **git** installed and can clone the CoSee repository.
- You have **Python 3.10+** installed on both local and server.
- You have either:
  - **conda** (Anaconda / Miniconda), or
  - Native `python3` + `venv` available.
- On the GPU server:
  - At least one modern GPU (e.g., RTX 4090, 24 GB),
  - A working CUDA + driver setup compatible with the PyTorch version you will install.

The project root is assumed to be:

```text
/…/CoSee/
````

with subdirectories like `cosee/`, `docs/`, `scripts/`, `models/`, etc.

---

## 2. Python environment: local development (macOS)

### 2.1 Create and activate a virtual environment

You can use either `venv` or `conda`. Choose one and stick to it.

#### Option A: `venv` (recommended if Python is already properly installed)

From the `CoSee` root:

```bash
cd /path/to/CoSee

# Check Python version
python3 --version

# Create virtual environment
python3 -m venv .venv

# Activate (macOS / Linux)
source .venv/bin/activate

# Later, to deactivate
deactivate
```

You should see a prompt like:

```text
(.venv) user@MacBook-Pro CoSee %
```

#### Option B: `conda`

```bash
cd /path/to/CoSee

# Create environment named "cosee"
conda create -n cosee python=3.10

# Activate
conda activate cosee

# Later, to deactivate
conda deactivate
```

In the rest of this document, we refer to the active environment generically as **the CoSee environment**.

---

### 2.2 Install local dependencies (CPU-friendly)

On macOS you only need enough dependencies to:

* Load Qwen3-VL-4B-Instruct,
* Run small CPU demos,
* Develop and test the CoSee code.

From the activated CoSee environment:

```bash
# Upgrade pip
pip install --upgrade pip

# Core libraries
pip install "transformers>=4.57.0"
pip install accelerate safetensors

# Vision and data handling
pip install pillow opencv-python datasets numpy tqdm

# Model management (optional but recommended)
pip install huggingface_hub

# PyTorch CPU build (sufficient for local testing)
pip install "torch==2.3.0" torchvision
```

You can verify imports with:

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

---

## 3. Python environment: GPU server

On the GPU server, we create a similar environment, but install a **GPU-enabled** PyTorch.

### 3.1 Create and activate the environment

Using `conda` (typical on servers):

```bash
cd /path/to/CoSee

conda create -n cosee python=3.10
conda activate cosee
```

Or using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

First install core libraries (same as local):

```bash
pip install --upgrade pip

pip install "transformers>=4.57.0"
pip install accelerate safetensors

pip install pillow opencv-python datasets numpy tqdm
pip install huggingface_hub
```

Then install **GPU-enabled PyTorch**. The exact command depends on your CUDA and driver setup; consult the official PyTorch installation instructions and use the command they provide for your OS and CUDA version, for example:

```bash
# Example only; replace with the command recommended by the PyTorch site.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you plan to experiment with quantization or more advanced optimizations, you may later add packages like `bitsandbytes` or `flash-attn`, but they are not required for a basic CoSee run.

Verify that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## 4. Model download and layout

CoSee assumes **Qwen3-VL-4B-Instruct** is available locally in a directory such as:

```text
CoSee/models/Qwen3-VL-4B-Instruct/
```

### 4.1 Local model download (macOS)

From the activated CoSee environment on your Mac:

```bash
cd /path/to/CoSee

# Create a models/ directory if it does not exist
mkdir -p models
```

Then download the model using your preferred method:

#### Option A: `huggingface_hub` CLI

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir ./models/Qwen3-VL-4B-Instruct \
    --local-dir-use-symlinks False
```

This should populate:

```text
models/Qwen3-VL-4B-Instruct/
  config.json
  generation_config.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  model.safetensors.index.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  preprocessor_config.json
  video_preprocessor_config.json
  chat_template.json
  README.md
  .gitattributes
```

You can verify loading with a minimal script (CPU only):

```python
# test_qwen_local.py
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_path = "./models/Qwen3-VL-4B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

print("Loading model (CPU)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, local_files_only=True, device_map="cpu", dtype="auto"
)

print("Loaded successfully.")
```

Run:

```bash
python test_qwen_local.py
```

If it prints `Loaded successfully.`, the local model is usable.

### 4.2 Transferring the model to the GPU server

Because network access to hosting platforms may be unstable, the recommended workflow is:

1. **Download once on your local machine** as above.

2. **Compress the model directory**, for example:

   ```bash
   cd /path/to/CoSee
   tar -czf Qwen3-VL-4B-Instruct.tar.gz -C models Qwen3-VL-4B-Instruct
   ```

3. **Upload the archive** to the server using `scp`, `rsync`, or an SFTP client (e.g., FileZilla).

4. On the server, choose a data directory (e.g., `/data/models`) and unpack:

   ```bash
   mkdir -p /data/models
   tar -xzf Qwen3-VL-4B-Instruct.tar.gz -C /data/models
   ```

   You should then have:

   ```text
   /data/models/Qwen3-VL-4B-Instruct/…
   ```

5. In the server-side CoSee configuration, set:

   * `model_path = "/data/models/Qwen3-VL-4B-Instruct"`

or define an environment variable for flexibility:

```bash
export COSEE_MODEL_PATH="/data/models/Qwen3-VL-4B-Instruct"
```

and have your code default to `os.environ.get("COSEE_MODEL_PATH", "./models/Qwen3-VL-4B-Instruct")`.

---

## 5. Project layout and configuration

A recommended project layout:

```text
CoSee/
  cosee/
    __init__.py
    board.py
    agents.py
    controller.py
    models/
      __init__.py
      qwen_vl_wrapper.py
    datasets/
      __init__.py
      mpdocvqa.py
      slidevqa.py
    eval/
      __init__.py
      metrics.py
      baselines.py
  scripts/
    dev_qwen_client_demo.py
    run_mpdocvqa_cosee.py
    run_mpdocvqa_baselines.py
  models/
    Qwen3-VL-4B-Instruct/        # local model copy (optional on server)
  docs/
    00_overview.md
    01_design_cosee.md
    02_api_spec.md
    03_experiments_plan.md
    04_dev_env_setup.md
    05_todo_and_sprints.md
  requirements.txt
  README.md
```

On the server, `models/` can be omitted if you store the model under `/data/models/` instead.

---

## 6. Initial smoke tests

Once the environment and model are in place, perform minimal smoke tests in both local and server environments.

### 6.1 Local: single-image demo

Create `scripts/dev_qwen_client_demo.py`:

```python
from PIL import Image
from cosee.models.qwen_vl_wrapper import QwenVLClient

def main():
    model_path = "./models/Qwen3-VL-4B-Instruct"
    client = QwenVLClient(model_path=model_path, device="cpu", dtype="auto")

    image = Image.open("test.jpg").convert("RGB")  # replace with a real image
    question = "Describe this image in one sentence."

    answer = client.generate(
        images=[image],
        question=question,
        board_text=None,
        role_prompt="You are a helpful multimodal assistant.",
        max_new_tokens=32,
    )

    print("MODEL OUTPUT:")
    print(answer)

if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/dev_qwen_client_demo.py
```

You should see a plausible description of the test image.

### 6.2 Server: GPU demo

On the server, after activating the CoSee environment:

```bash
cd /path/to/CoSee
python scripts/dev_qwen_client_demo.py
```

Make sure:

* `QwenVLClient` uses `model_path` pointing to `/data/models/Qwen3-VL-4B-Instruct`,
* `device` is `"auto"` or `"cuda:0"`.

You should see:

* GPU memory usage increase,
* Similar output content as on the local machine, but faster.

---

## 7. Environment files and reproducibility

To make environment reproduction easier:

### 7.1 `requirements.txt`

From your local environment (after installing dependencies):

```bash
pip freeze > requirements.txt
```

This captures exact versions. You may later manually trim it to only keep relevant packages.

On the server, you can then do:

```bash
pip install -r requirements.txt
```

(Adjust if you want a separate server-specific requirements file.)

### 7.2 Optional `environment.yml` (conda)

If using conda, you can export an environment file:

```bash
conda env export --name cosee > environment.yml
```

And recreate the environment elsewhere with:

```bash
conda env create -f environment.yml
```

---

## 8. Troubleshooting notes

* **Import errors (e.g., missing `torchvision`)**

  * Install missing libraries inside the active environment:

    ```bash
    pip install torchvision
    ```

* **Model not found / offline issues**

  * Ensure `model_path` is set correctly and `local_files_only=True` is used in model loading calls.
  * Verify the model directory structure is complete and was not truncated.

* **CUDA not available on server**

  * Check that you installed a GPU-enabled PyTorch matching the server’s CUDA version.
  * Verify `nvidia-smi` works and shows the GPU.
  * If necessary, reinstall PyTorch with a different CUDA wheel as recommended by the official instructions.

* **Token or memory errors**

  * Reduce `max_new_tokens`.
  * Reduce image resolution (if pre-processing is under your control).
  * Reduce the number of agents or collaboration steps in early experiments.

With these steps, both local and server environments should be ready for implementing and running CoSee, starting from small tests and scaling up to full DocVQA/SlideVQA experiments.