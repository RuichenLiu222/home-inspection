from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


class SmolVLMRunner:
    """Lazy, deterministic wrapper around Hugging Face SmolVLM inference."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        max_new_tokens: int = 160,
        seed: int = 42,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> tuple[str, float]:
        import numpy as np
        import torch

        self.load()
        assert self.processor is not None and self.model is not None

        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - started

        input_length = inputs["input_ids"].shape[-1]
        if generated_ids.shape[-1] > input_length:
            generated_ids = generated_ids[:, input_length:]
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        if text.lower().startswith("assistant:"):
            text = text.split(":", 1)[1].strip()
        return text, elapsed
