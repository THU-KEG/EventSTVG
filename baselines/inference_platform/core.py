"""Shared manifest, inference, resume, and monitoring logic."""

from __future__ import annotations

import concurrent.futures
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .provider import ProviderAdapter
from .resilience import CircuitBreaker, CircuitPolicy
from parser_policy import parse_response


PROTOCOL = "provider_defaults"
OPTIONAL_MODEL_PARAMETERS: list[str] = []


@dataclass(frozen=True)
class RunConfig:
    manifest_path: str
    video_root: str
    output_path: str
    model: str
    api_base_url: str
    workers: int = 1
    sample_start: int = 0
    sample_end: int | None = None
    sample_limit: int | None = None
    max_retries: int = 6
    retry_delays_seconds: tuple[float, ...] | list[float] = (5, 15, 30, 60, 120, 300)
    retry_jitter_fraction: float = 0.15
    circuit_window_seconds: float = 60.0
    circuit_profile_failure_threshold: int = 4
    circuit_gateway_failure_threshold: int = 8
    circuit_transient_failure_rate: float = 0.8
    circuit_initial_cooldown_seconds: float = 600.0
    circuit_repeated_cooldown_seconds: float = 900.0
    sample_ids: tuple[str, ...] | list[str] = ()
    rerun_sample_ids: tuple[str, ...] | list[str] = ()
    # Runtime-only resolution map for formal context videos. Frozen manifests
    # remain unchanged; when present, sample_id maps to an absolute window_path.
    video_path_map: dict[str, str] | None = None


@dataclass(frozen=True)
class RuntimeApiProfile:
    """One credential/backend profile contributing workers to a shared run."""

    name: str
    api_key: str
    base_url: str
    workers: int
    adapter_factory: Callable[[], ProviderAdapter]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            yield value


def normalize_manifest_record(record: dict[str, Any], path: Path, line_number: int) -> dict[str, Any]:
    """Normalize flat collaborator manifests and richer internal manifests.

    Ground-truth fields are intentionally ignored and never returned.
    """
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    sample_id = record.get("sample_id", metadata.get("sample_id"))
    setting = record.get("setting", metadata.get("setting"))
    prompt = record.get("prompt")
    video_path = record.get("video_path", metadata.get("video_path"))
    if not all(isinstance(value, str) and value for value in (sample_id, setting, prompt, video_path)):
        raise ValueError(
            f"{path}:{line_number}: sample_id, setting, prompt, and video_path must be strings"
        )
    prompt_metadata = {
        key: metadata.get(key)
        for key in (
            "prompt_spec_version", "prompt_spec_sha256", "record_schema_version",
            "generation_fingerprint", "source_manifest",
        )
        if metadata.get(key) is not None
    }
    return {
        "sample_id": sample_id,
        "setting": setting,
        "prompt": prompt,
        "video_path": video_path,
        "prompt_metadata": prompt_metadata,
    }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = [
        normalize_manifest_record(record, path, index)
        for index, record in enumerate(read_jsonl(path), 1)
    ]
    if not records:
        raise ValueError(f"{path}: manifest is empty")
    ids = [record["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample_id values")
    settings = {record["setting"] for record in records}
    if len(settings) != 1:
        raise ValueError(f"{path}: expected one setting, found {sorted(settings)}")
    return records


def discover_manifests(directory: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    if not directory.is_dir():
        return discovered
    for path in sorted(directory.glob("*.jsonl")):
        try:
            records = load_manifest(path)
        except (OSError, ValueError):
            continue
        setting = records[0]["setting"]
        if setting in discovered:
            continue
        discovered[setting] = path.resolve()
    return discovered


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_records(records: list[dict[str, Any]], config: RunConfig) -> list[dict[str, Any]]:
    if config.sample_start < 0:
        raise ValueError("sample_start must be non-negative")
    end = len(records) if config.sample_end is None else config.sample_end
    if end < config.sample_start:
        raise ValueError("sample_end must be greater than or equal to sample_start")
    selected = records[config.sample_start:end]
    if config.sample_ids:
        requested = set(config.sample_ids)
        available = {record["sample_id"] for record in records}
        unknown = requested - available
        if unknown:
            raise ValueError(f"unknown selected sample IDs: {sorted(unknown)}")
        selected = [record for record in selected if record["sample_id"] in requested]
    if config.sample_limit is not None:
        if config.sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        selected = selected[:config.sample_limit]
    rerun_unknown = set(config.rerun_sample_ids) - {record["sample_id"] for record in records}
    if rerun_unknown:
        raise ValueError(f"unknown rerun sample IDs: {sorted(rerun_unknown)}")
    rerun_excluded = set(config.rerun_sample_ids) - {
        record["sample_id"] for record in selected
    }
    if rerun_excluded:
        raise ValueError(f"rerun sample IDs excluded by current selection: {sorted(rerun_excluded)}")
    return selected


def resolve_video_path(video_root: Path, relative_path: str) -> Path:
    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise ValueError(f"manifest video_path must be relative: {relative_path}")
    root = video_root.resolve()
    resolved = (root / supplied).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest video_path escapes video root: {relative_path}") from exc
    return resolved


def successful_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        row["sample_id"]
        for row in read_jsonl(path)
        if row.get("status") == "ok" and isinstance(row.get("sample_id"), str)
    }


def validate_existing_results(path: Path, *, model: str, setting: str) -> None:
    """Refuse to mix another model, setting, or inference protocol on resume."""
    if not path.is_file():
        return
    for line_number, row in enumerate(read_jsonl(path), 1):
        if row.get("model") != model:
            raise ValueError(f"{path}:{line_number}: existing result model does not match")
        if row.get("setting") != setting:
            raise ValueError(f"{path}:{line_number}: existing result setting does not match")
        protocol = row.get("inference_protocol")
        legacy_provider_defaults = (
            protocol is None and row.get("inference_parameters") == {}
        )
        if protocol != PROTOCOL and not legacy_provider_defaults:
            raise ValueError(
                f"{path}:{line_number}: existing result is not from {PROTOCOL!r}"
            )


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


BALANCE_ERROR_TERMS = (
    "insufficient balance", "insufficient_balance", "payment required",
    "quota exhausted", "account quota", "billing", "balance exhausted",
    "balance not enough", "credit exhausted", "credit balance",
)

MODEL_ACCESS_DENIED_TERMS = (
    "team_model_access_denied", "team not allowed to access model",
)


def balance_exhausted_error(exc: Exception) -> bool:
    """Identify account funding failures before generic HTTP retry handling."""
    if _status_code(exc) == 402:
        return True
    messages: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    combined = " ".join(messages)
    return any(term in combined for term in BALANCE_ERROR_TERMS)


def balance_exhausted_result(result: dict[str, Any]) -> bool:
    diagnostics = result.get("transport_diagnostics") or []
    if any(item.get("http_status") == 402 for item in diagnostics if isinstance(item, dict)):
        return True
    message = str(result.get("error_message") or "").lower()
    return any(term in message for term in BALANCE_ERROR_TERMS)


def model_access_denied_error(exc: Exception) -> bool:
    """Identify a credential/profile that cannot use the requested model."""
    if _status_code(exc) != 403:
        return False
    messages: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    combined = " ".join(messages)
    return any(term in combined for term in MODEL_ACCESS_DENIED_TERMS)


def model_access_denied_result(result: dict[str, Any]) -> bool:
    diagnostics = result.get("transport_diagnostics") or []
    has_403 = any(
        item.get("http_status") == 403 for item in diagnostics if isinstance(item, dict)
    )
    message = str(result.get("error_message") or "").lower()
    return has_403 and any(term in message for term in MODEL_ACCESS_DENIED_TERMS)


def transient_error(exc: Exception) -> bool:
    if balance_exhausted_error(exc) or model_access_denied_error(exc):
        return False
    code = _status_code(exc)
    if code is not None:
        return code in {408, 429, 500, 502, 503, 504}
    name = type(exc).__name__.lower()
    return any(token in name for token in ("connection", "timeout", "ratelimit")) or isinstance(
        exc, (ConnectionError, TimeoutError)
    )


def exception_diagnostic(exc: Exception, api_key: str) -> dict[str, Any]:
    chain = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    root = chain[-1]
    status = _status_code(exc)
    name = type(root).__name__.lower()
    message = str(root).lower()
    if status == 429:
        category = "rate_limit"
    elif status in {500, 502, 503, 504}:
        category = "http_5xx"
    elif "dns" in name or "name resolution" in message or "nodename" in message:
        category = "dns"
    elif "ssl" in name or "tls" in name or "certificate" in message:
        category = "tls"
    elif "reset" in message:
        category = "connection_reset"
    elif "refused" in message:
        category = "connection_refused"
    elif "timeout" in name or "timed out" in message:
        category = "timeout"
    elif transient_error(exc):
        category = "connection"
    else:
        category = "permanent"
    return {
        "error_category": category,
        "http_status": status,
        "exception_class": type(exc).__name__,
        "root_exception_class": type(root).__name__,
        "root_exception_message": redact(str(root), api_key),
    }


def redact(text: str, secret: str) -> str:
    value = text.replace(secret, "[REDACTED]") if secret else text
    # Signed transport credentials must never enter durable diagnostics.
    return re.sub(r"(https://[^\s?]+)\?[^\s'\"<>]+", r"\1?[REDACTED_QUERY]", value)


def usage_totals(usage: dict[str, Any]) -> dict[str, int | float | None]:
    def integer(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int):
                return value
        return None

    reasoning = integer("reasoning_tokens")
    details = usage.get("completion_tokens_details", usage.get("output_tokens_details"))
    if reasoning is None and isinstance(details, dict):
        value = details.get("reasoning_tokens")
        reasoning = value if isinstance(value, int) else None
    return {
        "input_tokens": integer("prompt_tokens", "input_tokens"),
        "output_tokens": integer("completion_tokens", "output_tokens"),
        "total_tokens": integer("total_tokens"),
        "reasoning_tokens": reasoning,
        "cost_usd": usage.get("cost") if isinstance(usage.get("cost"), (int, float)) else None,
    }


def infer_one(
    record: dict[str, Any], config: RunConfig, adapter: ProviderAdapter, api_key: str,
    manifest_digest: str | None = None,
    api_profile: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    random_uniform: Callable[[float, float], float] = random.uniform,
    on_transient_attempt: Callable[[], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    retry_count = 0
    mapped_path = (config.video_path_map or {}).get(record["sample_id"])
    video_path = Path(mapped_path).resolve() if mapped_path else resolve_video_path(
        Path(config.video_root), record["video_path"])
    source_video = {"size_bytes": None, "duration_seconds": None}
    try:
        source_video["size_bytes"] = video_path.stat().st_size
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if probe.returncode == 0:
            source_video["duration_seconds"] = float(probe.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    base = {
        "sample_id": record["sample_id"],
        "setting": record["setting"],
        "model": config.model,
        "manifest_video_path": record["video_path"],
        "video_path": str(video_path),
        "source_video": source_video,
        "prompt_manifest": record["prompt_metadata"],
        "manifest_path": str(Path(config.manifest_path).resolve()),
        "manifest_sha256": manifest_digest,
        "inference_protocol": PROTOCOL,
        "optional_model_facing_parameters_sent": [],
    }
    if api_profile is not None:
        base["api_profile"] = api_profile
    diagnostics: list[dict[str, Any]] = []
    delays = tuple(float(value) for value in config.retry_delays_seconds)
    attempts = min(config.max_retries, len(delays)) + 1
    for attempt in range(attempts):
        try:
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            raw, usage = adapter.infer(config.model, video_path, record["prompt"])
            parsed = parse_response(raw, record["setting"])
            return {
                **base,
                "status": "ok",
                "raw_response": raw,
                "parsed_response": parsed,
                "prediction": parsed,
                "parse_status": "ok" if parsed is not None else "error",
                "parse_ok": parsed is not None,
                "latency_seconds": round(time.monotonic() - started, 6),
                "retry_count": retry_count,
                "api_usage": usage,
                "usage": usage_totals(usage),
                "error_type": None,
                "error_message": None,
                "error_retryable": None,
                "transport_diagnostics": diagnostics,
            }
        except Exception as exc:
            retryable = transient_error(exc)
            delay = None
            if attempt < attempts - 1 and retryable:
                base_delay = delays[attempt]
                jitter = max(0.0, float(config.retry_jitter_fraction))
                delay = max(0.0, base_delay * random_uniform(1.0 - jitter, 1.0 + jitter))
            diagnostics.append({
                **exception_diagnostic(exc, api_key),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "api_profile": api_profile,
                "attempt_number": attempt + 1,
                "retry_delay_seconds": round(delay, 3) if delay is not None else None,
            })
            if retryable and on_transient_attempt is not None:
                on_transient_attempt()
            if delay is not None:
                retry_count += 1
                sleep(delay)
                continue
            return {
                **base,
                "status": "error",
                "raw_response": None,
                "parsed_response": None,
                "prediction": None,
                "parse_status": "not_attempted",
                "parse_ok": False,
                "latency_seconds": round(time.monotonic() - started, 6),
                "retry_count": retry_count,
                "api_usage": {},
                "usage": usage_totals({}),
                "error_type": type(exc).__name__,
                "error_message": redact(str(exc), api_key),
                "error_retryable": retryable,
                "transport_diagnostics": diagnostics,
            }
    raise AssertionError("unreachable")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def summarize_results(
    output_path: Path, *, total_samples: int, started_at_epoch: float | None = None,
    sample_ids: set[str] | None = None, elapsed_offset_seconds: float = 0.0,
) -> dict[str, Any]:
    rows = list(read_jsonl(output_path)) if output_path.is_file() else []
    if sample_ids is not None:
        rows = [row for row in rows if row.get("sample_id") in sample_ids]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            grouped.setdefault(sample_id, []).append(row)
    # A sample remains complete once any successfully returned response is
    # durably saved. An explicit later rerun failure does not erase that fact.
    successes = [
        next(row for row in reversed(attempts) if row.get("status") == "ok")
        for attempts in grouped.values()
        if any(row.get("status") == "ok" for row in attempts)
    ]
    failures = [
        attempts[-1] for attempts in grouped.values()
        if not any(row.get("status") == "ok" for row in attempts)
        and attempts[-1].get("status") == "error"
    ]
    current = [attempts[-1] for attempts in grouped.values()]
    parse_failures = [row for row in successes if not row.get("parse_ok")]
    latencies = [float(row["latency_seconds"]) for row in current if isinstance(row.get("latency_seconds"), (int, float))]
    segment_elapsed = max(0.0, time.time() - started_at_epoch) if started_at_epoch else 0.0
    elapsed = max(0.0, elapsed_offset_seconds) + segment_elapsed
    # Benchmark progress is sample-based: only a successful saved response
    # completes a sample. Failed attempts remain resumable and must not inflate
    # completion, throughput, or reduce remaining work.
    completed = len(successes)
    throughput = completed / elapsed * 60 if elapsed > 0 else 0.0
    remaining = max(0, total_samples - completed)
    token_keys = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")
    tokens: dict[str, int | None] = {}
    for key in token_keys:
        values = [row.get("usage", {}).get(key) for row in rows]
        reported = [value for value in values if isinstance(value, int)]
        tokens[key] = sum(reported) if reported else None
    reported_costs = [
        row.get("usage", {}).get("cost_usd") for row in rows
        if isinstance(row.get("usage"), dict)
    ]
    reported_costs = [value for value in reported_costs if isinstance(value, (int, float))]
    errors = [
        {
            "sample_id": row.get("sample_id"),
            "api_profile": row.get("api_profile"),
            "error_type": row.get("error_type"),
            "retry_count": row.get("retry_count", 0),
            "error_message": row.get("error_message"),
            "error_retryable": row.get("error_retryable"),
        }
        for row in failures[-20:]
    ]
    return {
        "total_samples": total_samples,
        "completed": completed,
        "attempted_samples": len(current),
        "successful_requests": len(successes),
        "request_failures": len(failures),
        "request_failure_attempts": sum(row.get("status") == "error" for row in rows),
        "parsed_responses": len(successes) - len(parse_failures),
        "parse_failures": len(parse_failures),
        "retried": len({
            row.get("sample_id") for row in rows
            if isinstance(row.get("sample_id"), str) and int(row.get("retry_count", 0)) > 0
        }),
        "total_retries": sum(int(row.get("retry_count", 0)) for row in rows),
        "remaining": remaining,
        "completion_percentage": round(completed / total_samples * 100, 3) if total_samples else 100.0,
        "elapsed_seconds": round(elapsed, 3),
        "average_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "throughput_requests_per_minute": round(throughput, 3),
        "estimated_remaining_seconds": round(remaining / throughput * 60, 3) if throughput > 0 else None,
        "provider_reported_usage": tokens,
        "provider_reported_cost_usd": round(sum(reported_costs), 12) if reported_costs else None,
        "video_tokens": None,
        "recent_errors": errors,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_inference(
    config: RunConfig,
    *,
    api_key: str,
    adapter_factory: Callable[[], ProviderAdapter],
    stop_path: Path,
    metadata_path: Path,
    summary_path: Path,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    random_uniform: Callable[[float, float], float] = random.uniform,
) -> dict[str, Any]:
    """Run with one API profile; retained for cloud/UI compatibility."""
    profile = RuntimeApiProfile(
        name="default", api_key=api_key, base_url=config.api_base_url,
        workers=config.workers, adapter_factory=adapter_factory,
    )
    return run_inference_pool(
        config, api_profiles=[profile], stop_path=stop_path,
        metadata_path=metadata_path, summary_path=summary_path,
        record_api_profile=False,
        monotonic=monotonic, sleep=sleep, random_uniform=random_uniform,
    )


def run_inference_pool(
    config: RunConfig,
    *,
    api_profiles: list[RuntimeApiProfile],
    stop_path: Path,
    metadata_path: Path,
    summary_path: Path,
    record_api_profile: bool = True,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    random_uniform: Callable[[float, float], float] = random.uniform,
) -> dict[str, Any]:
    """Run fixed API worker slots against one queue and one result writer."""
    if not api_profiles:
        raise ValueError("at least one API profile is required")
    names = [profile.name for profile in api_profiles]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("API profile names must be non-empty and unique")
    if any(not profile.api_key for profile in api_profiles):
        raise ValueError("each API profile requires an API key")
    if any(profile.workers < 1 for profile in api_profiles):
        raise ValueError("API profile workers must be positive")
    normalized_urls = {profile.base_url.rstrip("/") for profile in api_profiles}
    if "" in normalized_urls:
        raise ValueError("API profile base URLs must be non-empty")
    total_workers = sum(profile.workers for profile in api_profiles)
    manifest_path = Path(config.manifest_path).resolve()
    output_path = Path(config.output_path).resolve()
    records = load_manifest(manifest_path)
    selected = select_records(records, config)
    selected_ids = {record["sample_id"] for record in selected}
    setting = records[0]["setting"]
    # Older provider-default rows used a null protocol marker together with an
    # explicit empty parameter dictionary. They remain valid completed history.
    validate_existing_results(output_path, model=config.model, setting=setting)
    segment_started_epoch = time.time()
    previous_metadata = {}
    previous_summary = {}
    try:
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    try:
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    previous_elapsed = previous_summary.get("elapsed_seconds", 0.0)
    if not isinstance(previous_elapsed, (int, float)):
        previous_elapsed = 0.0
    original_started_epoch = previous_metadata.get("started_at_epoch", segment_started_epoch)
    if not isinstance(original_started_epoch, (int, float)):
        original_started_epoch = segment_started_epoch
    digest = manifest_sha256(manifest_path)
    metadata = {
        "status": "running",
        "model": config.model,
        "setting": setting,
        "inference_protocol": PROTOCOL,
        "optional_model_facing_parameters_sent": OPTIONAL_MODEL_PARAMETERS,
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest,
        "video_root": str(Path(config.video_root).resolve()),
        "output_path": str(output_path),
        "workers": total_workers,
        "api_profiles": [
            {"name": profile.name, "base_url": profile.base_url, "workers": profile.workers}
            for profile in api_profiles
        ],
        "sample_start": config.sample_start,
        "sample_end": config.sample_end,
        "sample_limit": config.sample_limit,
        "sample_ids": list(config.sample_ids),
        "rerun_sample_ids": list(config.rerun_sample_ids),
        "selected_samples": len(selected),
        "api_base_url": next(iter(normalized_urls)) if len(normalized_urls) == 1 else None,
        "worker_pid": os.getpid(),
        "started_at_epoch": original_started_epoch,
        "segment_started_at_epoch": segment_started_epoch,
    }
    write_json(metadata_path, metadata)
    existing_attempts: dict[str, list[dict[str, Any]]] = {}
    if output_path.is_file():
        for row in read_jsonl(output_path):
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str):
                existing_attempts.setdefault(sample_id, []).append(row)
    forced = set(config.rerun_sample_ids)

    def should_run(record: dict[str, Any]) -> bool:
        sample_id = record["sample_id"]
        if sample_id in forced:
            return True
        attempts = existing_attempts.get(sample_id, [])
        if not attempts:
            return True
        # Validated unmarked legacy attempts are provenance only. They are never
        # selected by automatic resume; rerunning one requires --rerun-sample(s).
        if any(row.get("inference_protocol") is None for row in attempts):
            return False
        if any(row.get("status") == "ok" for row in attempts):
            return False
        latest = attempts[-1]
        # Funding failures are terminal within one run, but the sample remains
        # unfinished and must become eligible after accounts are replenished.
        if balance_exhausted_result(latest):
            return True
        if model_access_denied_result(latest):
            return True
        return latest.get("status") == "error" and latest.get("error_retryable") is True

    pending = deque(record for record in selected if should_run(record))
    stopped = False
    balance_auto_stopped = False
    disabled_profiles: set[str] = set()
    model_access_denied_profiles: set[str] = set()

    class WorkerSlot:
        def __init__(self, profile: RuntimeApiProfile):
            self.profile = profile
            self.adapter: ProviderAdapter | None = None

        def infer(self, record: dict[str, Any]) -> dict[str, Any]:
            # Each fixed slot owns one client, avoiding undocumented cross-thread
            # client sharing while preserving the configured per-profile capacity.
            adapter = self.adapter
            if adapter is None:
                try:
                    adapter = self.profile.adapter_factory()
                    self.adapter = adapter
                except Exception as exc:
                    adapter = FailedAdapter(exc)
                    self.adapter = adapter
            return infer_one(
                record, config, adapter, self.profile.api_key, digest,
                api_profile=self.profile.name if record_api_profile else None,
                sleep=sleep, random_uniform=random_uniform,
                on_transient_attempt=lambda: breaker.record(
                    self.profile.name, transient_failure=True, success=False
                ),
            )

    class FailedAdapter:
        def __init__(self, error: Exception):
            self.error = error

        def infer(self, model: str, video_path: Path, prompt: str):
            raise self.error

    slots = [
        WorkerSlot(profile)
        for profile in api_profiles
        for _ in range(profile.workers)
    ]

    policy = CircuitPolicy(
        window_seconds=config.circuit_window_seconds,
        profile_failure_threshold=config.circuit_profile_failure_threshold,
        gateway_failure_threshold=config.circuit_gateway_failure_threshold,
        transient_failure_rate=config.circuit_transient_failure_rate,
        initial_cooldown_seconds=config.circuit_initial_cooldown_seconds,
        repeated_cooldown_seconds=config.circuit_repeated_cooldown_seconds,
    )
    breaker = CircuitBreaker(
        {profile.name: profile.base_url.rstrip("/") for profile in api_profiles},
        policy=policy, monotonic=monotonic,
    )

    def next_record(slot: WorkerSlot) -> dict[str, Any] | None:
        nonlocal stopped
        if stop_path.exists():
            stopped = True
            return None
        if (slot.profile.name in disabled_profiles or not pending
                or not breaker.eligible(slot.profile.name)):
            return None
        breaker.mark_started(slot.profile.name)
        return pending.popleft()

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as pool:
        active: dict[concurrent.futures.Future[dict[str, Any]], tuple[WorkerSlot, dict[str, Any]]] = {}
        idle = list(slots)

        def schedule() -> None:
            for slot in list(idle):
                record = next_record(slot)
                if record is None:
                    continue
                idle.remove(slot)
                active[pool.submit(slot.infer, record)] = (slot, record)

        schedule()
        while active or (pending and not stopped):
            if not active:
                enabled = [
                    slot.profile.name for slot in idle
                    if slot.profile.name not in disabled_profiles
                ]
                if not enabled:
                    stopped = True
                    balance_auto_stopped = bool(disabled_profiles)
                    break
                delay = breaker.seconds_until_any_eligible(enabled)
                if delay > 0:
                    sleep(delay)
                schedule()
                continue
            finished, _ = concurrent.futures.wait(
                active, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in finished:
                slot, claimed = active.pop(future)
                idle.append(slot)
                result = future.result()
                # This coordinator is the only JSONL writer. Worker threads only
                # return records, so attempts cannot interleave on disk.
                append_jsonl(output_path, result)
                success = result.get("status") == "ok"
                transient = result.get("status") == "error" and result.get("error_retryable") is True
                balance_exhausted = (
                    result.get("status") == "error" and balance_exhausted_result(result)
                )
                model_access_denied = (
                    result.get("status") == "error" and model_access_denied_result(result)
                )
                if balance_exhausted:
                    disabled_profiles.add(slot.profile.name)
                if model_access_denied:
                    disabled_profiles.add(slot.profile.name)
                    model_access_denied_profiles.add(slot.profile.name)
                if success or not transient:
                    breaker.record(slot.profile.name, transient_failure=False, success=success)
                # A terminal transport attempt is provenance, not completion.
                # Keep that exact claimed record pending in this run; another
                # healthy profile may recover it while the failing profile cools.
                if transient and not stop_path.exists():
                    pending.appendleft(claimed)
                elif balance_exhausted and not stop_path.exists():
                    # The request never reached usable paid inference. Give the
                    # claimed sample to another funded profile, if one remains.
                    pending.appendleft(claimed)
                    if len(disabled_profiles) == len(api_profiles):
                        stopped = True
                        balance_auto_stopped = True
                elif model_access_denied and not stop_path.exists():
                    # Model authorization is profile-specific. Move this sample
                    # to another credential instead of failing it permanently.
                    pending.appendleft(claimed)
                    if len(disabled_profiles) == len(api_profiles):
                        stopped = True
                summary = summarize_results(
                    output_path, total_samples=len(selected), started_at_epoch=segment_started_epoch,
                    sample_ids=selected_ids,
                    elapsed_offset_seconds=float(previous_elapsed),
                )
                summary["circuit_breaker"] = breaker.snapshot()
                summary["disabled_profiles"] = sorted(disabled_profiles)
                summary["model_access_denied_profiles"] = sorted(model_access_denied_profiles)
                summary["balance_auto_stopped"] = balance_auto_stopped
                write_json(summary_path, summary)
            schedule()

    final = summarize_results(
        output_path, total_samples=len(selected), started_at_epoch=segment_started_epoch,
        sample_ids=selected_ids,
        elapsed_offset_seconds=float(previous_elapsed),
    )
    final["circuit_breaker"] = breaker.snapshot()
    final["disabled_profiles"] = sorted(disabled_profiles)
    final["model_access_denied_profiles"] = sorted(model_access_denied_profiles)
    final["balance_auto_stopped"] = balance_auto_stopped
    status = ("stopped_balance_exhausted" if balance_auto_stopped else
              "stopped" if stopped else
              ("complete_with_failures" if final["request_failures"] else "complete"))
    metadata.update(status=status, finished_at_epoch=time.time())
    final.update(status=status, model=config.model, setting=setting, inference_protocol=PROTOCOL,
                 output_path=str(output_path))
    write_json(metadata_path, metadata)
    write_json(summary_path, final)
    return final


def config_without_secrets(config: RunConfig) -> dict[str, Any]:
    return asdict(config)
