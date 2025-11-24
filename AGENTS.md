# Repository Guidelines

## Project Structure & Modules
- Core scripts live at `demo.py` and `test.py`; both load the local Qwen3-VL-4B-Instruct model and demonstrate minimal inference flows.
- Model weights and tokenizer assets reside in `models/Qwen3-VL-4B-Instruct/`; keep this path unchanged so loaders resolve `local_files_only=True`.
- Sample media assets go in the repo root (e.g., `images.jpeg`); add new fixtures with clear names like `sample_dog.jpg`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` (if present) or `pip install transformers pillow torch` to match the current scripts.
- `python demo.py` runs an end-to-end image + text prompt through the local model and prints the generated answer.
- `python test.py` verifies the model and processor load on CPU without running generation; use it as a quick sanity check after updates.

## Coding Style & Naming
- Python 3; prefer f-strings, type hints where practical, and explicit imports.
- Use 4-space indentation and keep lines under ~100 chars.
- Name new demo or utility scripts with action-oriented verbs, e.g., `export_metadata.py`, and co-locate assets they need.
- When adjusting model paths, keep them configurable via top-level constants to simplify swaps.

## Testing Guidelines
- Add lightweight smoke tests for loader changes (e.g., calling `AutoProcessor.from_pretrained` with `local_files_only=True`).
- If adding generation logic, include a minimal assertion on output shape or non-empty text rather than golden strings (model outputs vary).
- For new assets, document expected formats (RGB images, mp4 video) in comments near the load site.

## Commit & Pull Request Guidelines
- Write imperative, scoped commit messages: `load model config lazily`, `add cpu smoke test`.
- In PRs, include: purpose, key changes, how to run `python test.py` or other validation, and note any model/asset additions.
- Attach sample outputs or logs when touching generation paths; include file sizes for added assets.

## Security & Configuration
- Keep `local_files_only=True` when loading to avoid unintended network calls.
- Do not commit proprietary model weights beyond the checked-in artifacts; prefer `.gitattributes`/LFS for large files if needed.
- Avoid embedding secrets in scripts; use environment variables for any future credentials.