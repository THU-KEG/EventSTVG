"""Load protected YAML/env-file API profiles for local inference only."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Any

from dotenv import dotenv_values
import yaml

from inference_platform.api_pool import validate_api_profiles


@dataclass(frozen=True)
class LocalApiPoolProfile:
    name: str
    env_file: Path
    workers: int
    model: str
    backend: str
    base_url: str
    api_key: str


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_api_pool(path: Path, *, model: str) -> list[LocalApiPoolProfile]:
    path = path.expanduser().resolve()
    payload = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    unknown_top = set(payload) - {"model", "backend", "apis"}
    if unknown_top:
        raise ValueError(f"{path}: unknown fields: {sorted(unknown_top)}")
    pool_model = payload.get("model", model)
    if pool_model != model:
        raise ValueError(f"{path}: pool model {pool_model!r} does not match {model!r}")
    pool_backend = payload.get("backend", "openai_compatible")
    if pool_backend != "openai_compatible":
        raise ValueError(f"{path}: unsupported backend {pool_backend!r}")
    entries = payload.get("apis")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: apis must be a non-empty list")

    profiles: list[LocalApiPoolProfile] = []
    for index, raw in enumerate(entries):
        entry = _mapping(raw, f"{path}: apis[{index}]")
        unknown = set(entry) - {"name", "env_file", "workers", "model", "backend"}
        if unknown:
            raise ValueError(f"{path}: apis[{index}] unknown fields: {sorted(unknown)}")
        name = entry.get("name")
        workers = entry.get("workers", 1)
        entry_model = entry.get("model", pool_model)
        entry_backend = entry.get("backend", pool_backend)
        env_value = entry.get("env_file")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: apis[{index}].name must be non-empty")
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError(f"{path}: apis[{index}].workers must be positive")
        if entry_model != model or entry_backend != pool_backend:
            raise ValueError(f"{path}: API profile {name!r} is incompatible")
        if not isinstance(env_value, str) or not env_value:
            raise ValueError(f"{path}: apis[{index}].env_file must be a path string")
        env_file = Path(os.path.expandvars(env_value)).expanduser()
        if not env_file.is_absolute():
            env_file = path.parent / env_file
        env_file = env_file.resolve()
        mode = stat.S_IMODE(env_file.stat().st_mode)
        if mode & 0o077:
            raise ValueError(f"{env_file}: must be chmod 600 (current mode {mode:o})")
        values = dotenv_values(env_file)
        api_key = values.get("SCHOOL_API_KEY")
        base_url = values.get("SCHOOL_BASE_URL")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError(f"{env_file}: SCHOOL_API_KEY is required")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError(f"{env_file}: SCHOOL_BASE_URL is required")
        profiles.append(LocalApiPoolProfile(
            name=name.strip(), env_file=env_file, workers=workers, model=model,
            backend=pool_backend, base_url=base_url, api_key=api_key,
        ))
    return validate_api_profiles(profiles, model=model, source=str(path))  # type: ignore[return-value]


__all__ = ["LocalApiPoolProfile", "load_api_pool"]
