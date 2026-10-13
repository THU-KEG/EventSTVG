from __future__ import annotations
from .base import LocalModelAdapter
from .qwen35 import Qwen35Adapter

_FACTORIES = {"qwen3.5-9b": Qwen35Adapter}

def register(name: str, factory: type[LocalModelAdapter]) -> None:
    _FACTORIES[name] = factory

def list_models() -> list[str]: return sorted(_FACTORIES)

def get_adapter(name: str, **kwargs) -> LocalModelAdapter:
    try: return _FACTORIES[name](**kwargs)
    except KeyError: raise ValueError(f"unknown local model {name!r}; available: {list_models()}")
