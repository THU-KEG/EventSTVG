from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import torch
from .base import LocalModelAdapter, LoadedModel

class Qwen35Adapter(LocalModelAdapter):
    name = "qwen3.5-9b"
    model_path = Path("models/Qwen3.5-9B")
    # The formal API protocol intentionally leaves max_tokens unset (provider
    # default). Local generation needs an explicit safety ceiling; 4096 is a
    # generous equivalent for the longest observed Oracle JSON responses.
    max_new_tokens = 4096

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or self.model_path)
        self.loaded: LoadedModel | None = None
        self.load_seconds: float | None = None

    def load(self) -> LoadedModel:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path, local_files_only=True, device_map="auto", torch_dtype="auto"
        )
        self.loaded = LoadedModel(model, processor)
        self.load_seconds = time.perf_counter() - started
        return self.loaded

    def prepare_inputs(self, video_path: Path, prompt: str) -> dict[str, Any]:
        if self.loaded is None: raise RuntimeError("adapter is not loaded")
        if not video_path.is_file(): raise FileNotFoundError(video_path)
        messages = [{"role": "user", "content": [
            # Supplying the path lets Qwen's native processor select its own
            # FPS, frame limits, resizing, and source metadata.
            {"type": "video", "video": str(video_path)},
            {"type": "text", "text": prompt}
        ]}]
        batch = self.loaded.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True,
        )
        return {"batch": batch, "prompt_tokens": batch["input_ids"].shape[-1]}

    def generate(self, inputs: dict[str, Any]) -> Any:
        if self.loaded is None: raise RuntimeError("adapter is not loaded")
        batch = {k: (v.to(self.loaded.model.device) if hasattr(v, "to") else v)
                 for k, v in inputs["batch"].items()}
        with torch.inference_mode():
            return self.loaded.model.generate(
                **batch, max_new_tokens=self.max_new_tokens, do_sample=False,
            )

    def decode(self, generated: Any, inputs: dict[str, Any]) -> str:
        if self.loaded is None: raise RuntimeError("adapter is not loaded")
        prompt_tokens = inputs["prompt_tokens"]
        return self.loaded.processor.batch_decode(
            generated[:, prompt_tokens:], skip_special_tokens=True
        )[0].strip()
