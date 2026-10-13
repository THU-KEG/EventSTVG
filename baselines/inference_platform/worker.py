#!/usr/bin/env python3
"""Independent Role-STVG inference worker launched by the UI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from inference_platform.api_pool import load_in_memory_api_pool
from inference_platform.core import (
    RunConfig, RuntimeApiProfile, run_inference, run_inference_pool, write_json,
)
from inference_platform.provider import OpenAICompatibleAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.config.resolve().parent
    secrets: list[str] = []
    try:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        if isinstance(payload.get("run_config"), dict):
            config = RunConfig(**payload["run_config"])
            api_pool_source = payload.get("api_pool_source")
        else:
            # Backward compatibility with existing single-API worker configs.
            config = RunConfig(**payload)
            api_pool_source = None
        if api_pool_source == "environment":
            encoded_pool = os.environ.pop("ROLE_STVG_API_POOL_JSON", "")
            if not encoded_pool:
                raise ValueError("ephemeral API pool is missing from worker environment")
            profiles = load_in_memory_api_pool(json.loads(encoded_pool), model=config.model)
            encoded_pool = ""
        elif api_pool_source is None:
            profiles = []
        else:
            raise ValueError(f"unsupported API pool source: {api_pool_source!r}")
        if profiles:
            secrets.extend(profile.api_key for profile in profiles)
            runtime_profiles = [
                RuntimeApiProfile(
                    name=profile.name, api_key=profile.api_key,
                    base_url=profile.base_url, workers=profile.workers,
                    adapter_factory=(
                        lambda key=profile.api_key, url=profile.base_url:
                        OpenAICompatibleAdapter(api_key=key, base_url=url)
                    ),
                )
                for profile in profiles
            ]
            run_inference_pool(
                config, api_profiles=runtime_profiles,
                stop_path=run_dir / "stop.requested",
                metadata_path=run_dir / "run.json",
                summary_path=run_dir / "summary.json",
            )
        else:
            api_key = os.environ.get("ROLE_STVG_API_KEY", "")
            secrets.append(api_key)
            run_inference(
                config,
                api_key=api_key,
                adapter_factory=lambda: OpenAICompatibleAdapter(
                    api_key=api_key, base_url=config.api_base_url
                ),
                stop_path=run_dir / "stop.requested",
                metadata_path=run_dir / "run.json",
                summary_path=run_dir / "summary.json",
            )
    except Exception as exc:
        # Never include any pool or single-profile key in the worker log.
        message = str(exc)
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        metadata_path = run_dir / "run.json"
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        existing.update(
            status="failed", fatal_error_type=type(exc).__name__,
            fatal_error_message=message,
        )
        write_json(metadata_path, existing)
        print(f"{type(exc).__name__}: {message}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
