"""Validate ephemeral API profiles entered by the cloud collaborator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApiPoolProfile:
    name: str
    workers: int
    model: str
    backend: str
    base_url: str
    api_key: str


def validate_api_profiles(
    profiles: list[ApiPoolProfile], *, model: str, source: str,
) -> list[ApiPoolProfile]:
    if not profiles:
        raise ValueError(f"{source}: at least one API profile is required")
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)):
        raise ValueError(f"{source}: API profile names must be unique")
    if any(profile.model != model for profile in profiles):
        raise ValueError(f"{source}: all API profiles must use model alias {model!r}")
    if {profile.backend for profile in profiles} != {"openai_compatible"}:
        raise ValueError(f"{source}: all profiles must use the OpenAI-compatible backend")
    base_urls = {profile.base_url.rstrip("/") for profile in profiles}
    if "" in base_urls:
        raise ValueError(f"{source}: API profile base URLs must be non-empty")
    return profiles


def load_in_memory_api_pool(value: Any, *, model: str) -> list[ApiPoolProfile]:
    """Validate ephemeral UI profiles supplied through worker memory only."""
    if not isinstance(value, list) or not value:
        raise ValueError("in-memory API pool must be a non-empty list")
    profiles: list[ApiPoolProfile] = []
    allowed = {"name", "base_url", "api_key", "workers", "model", "backend"}
    for index, raw in enumerate(value):
        entry = _mapping(raw, f"in-memory API pool[{index}]")
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(f"in-memory API pool[{index}] unknown fields: {sorted(unknown)}")
        name = entry.get("name")
        base_url = entry.get("base_url")
        api_key = entry.get("api_key")
        workers = entry.get("workers", 1)
        entry_model = entry.get("model", model)
        backend = entry.get("backend", "openai_compatible")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"in-memory API pool[{index}].name must be non-empty")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(f"in-memory API pool[{index}].base_url must be non-empty")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError(f"in-memory API pool[{index}].api_key is required")
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError(f"in-memory API pool[{index}].workers must be positive")
        profiles.append(ApiPoolProfile(
            name=name.strip(), workers=workers,
            model=entry_model, backend=backend, base_url=base_url.strip(), api_key=api_key,
        ))
    return validate_api_profiles(profiles, model=model, source="in-memory API pool")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value
