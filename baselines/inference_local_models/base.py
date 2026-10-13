from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class LoadedModel:
    model: Any
    processor: Any

class LocalModelAdapter(ABC):
    name: str
    model_path: Path

    @abstractmethod
    def load(self) -> LoadedModel: ...
    @abstractmethod
    def prepare_inputs(self, video_path: Path, prompt: str) -> dict[str, Any]: ...
    @abstractmethod
    def generate(self, inputs: dict[str, Any]) -> Any: ...
    @abstractmethod
    def decode(self, generated: Any, inputs: dict[str, Any]) -> str: ...
    def unload(self) -> None:
        import gc
        self.loaded = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass
