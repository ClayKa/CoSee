from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


class QwenVLClient:
    """
    Lazy-loading wrapper around a local Qwen VL model.

    The object can be constructed during argument parsing and dry runs without
    loading model weights. The model and processor are loaded on first generate().
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.processor: Any = None
        self.model: Any = None
        self._compute: Dict[str, int] = {"calls": 0, "gen_tokens_total": 0}

    def reset_compute(self) -> None:
        self._compute = {"calls": 0, "gen_tokens_total": 0}

    def get_compute(self) -> Dict[str, int]:
        return dict(self._compute)

    def build_messages(
        self,
        images: List[Image.Image],
        question: str,
        board_text: Optional[str] = None,
        role_prompt: Optional[str] = None,
        context: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for image in images:
            content.append({"type": "image", "image": image})

        text_parts: List[str] = []
        if role_prompt:
            text_parts.append(role_prompt.strip())
        if dataset:
            text_parts.append(f"Dataset: {dataset}")
        if context:
            text_parts.append(f"Context:\n{context.strip()}")
        text_parts.append(f"Question:\n{question.strip()}")
        if board_text:
            text_parts.append(f"Shared board summary:\n{board_text.strip()}")

        content.append({"type": "text", "text": "\n\n".join(part for part in text_parts if part)})
        return [{"role": "user", "content": content}]

    def _resolve_dtype(self) -> Any:
        if self.dtype == "auto":
            return "auto"
        import torch

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(str(self.dtype).lower(), self.dtype)

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        model_dir = Path(self.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model path not found: {model_dir}. "
                "Set COSEE_MODEL_PATH or pass --model-path to a local Qwen VL checkpoint."
            )

        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for QwenVLClient. Install requirements.txt first."
            ) from exc
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": self._resolve_dtype(),
        }
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        if self.device != "auto" and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def tokenize_prompt_length(self, messages: List[Dict[str, Any]]) -> int:
        self._ensure_loaded()
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return 0
        return int(input_ids.shape[-1])

    def generate(
        self,
        images: List[Image.Image],
        question: str,
        board_text: Optional[str] = None,
        role_prompt: Optional[str] = None,
        context: Optional[str] = None,
        dataset: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        return_full_text: bool = False,
        **gen_kwargs: Any,
    ) -> str:
        self._ensure_loaded()
        import torch

        messages = self.build_messages(
            images=images,
            question=question,
            board_text=board_text,
            role_prompt=role_prompt,
            context=context,
            dataset=dataset,
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        target_device = getattr(self.model, "device", None)
        if target_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(target_device)
        elif target_device is not None:
            inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        do_sample = temperature is not None and temperature > 0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            generation_kwargs["top_k"] = top_k
        generation_kwargs.update(gen_kwargs)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.get("input_ids")
        prompt_len = int(input_ids.shape[-1]) if input_ids is not None else 0
        generated_len = max(0, int(output_ids.shape[-1]) - prompt_len)
        self._compute["calls"] += 1
        self._compute["gen_tokens_total"] += generated_len

        if return_full_text:
            decoded_ids = output_ids
        else:
            decoded_ids = output_ids[:, prompt_len:] if prompt_len else output_ids
        text = self.processor.batch_decode(
            decoded_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        text = text.strip()
        return text

    def _extract_assistant_answer(self, text: str) -> str:
        if not text:
            return ""
        markers = [
            r"<\|im_start\|>assistant\s*",
            r"assistant\s*\n",
            r"Assistant:\s*",
            r"ASSISTANT:\s*",
        ]
        answer = text
        for pattern in markers:
            matches = list(re.finditer(pattern, answer))
            if matches:
                answer = answer[matches[-1].end() :]
        answer = answer.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        return answer.strip()
