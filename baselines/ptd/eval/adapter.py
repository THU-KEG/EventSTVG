#!/usr/bin/env python3
"""Deterministic offline adapter from native PTD tubes to Role-STVG rows.

This module is deliberately model-free.  It only reads committed PTD role
records and never writes to the inference or frozen-policy directories.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from bisect import bisect_left
from pathlib import Path
from typing import Any, Iterable


ADAPTER_ID = "ptd_64f_role_stvg_adapter_v1.1"
EPS = 1e-9


class AdapterError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_sampling(record: dict[str, Any]) -> tuple[list[int], list[float], float, int, float]:
    indices = record.get("selected_source_frame_indices")
    timestamps = record.get("selected_timestamps")
    fps = record.get("source_fps")
    frame_count = record.get("source_frame_count")
    duration = record.get("source_duration")
    if not isinstance(indices, list) or not indices:
        raise AdapterError("selected_source_frame_indices must be a nonempty list")
    if not isinstance(timestamps, list) or len(timestamps) != len(indices):
        raise AdapterError("selected timestamps and indices must have identical nonzero length")
    if record.get("input_frame_count") != len(indices):
        raise AdapterError("input_frame_count does not equal stored sampling length")
    if not _finite(fps) or float(fps) <= 0 or not isinstance(frame_count, int) or frame_count <= 0:
        raise AdapterError("invalid source FPS/frame count")
    if not _finite(duration) or float(duration) <= 0:
        raise AdapterError("invalid source duration")
    if any(not isinstance(x, int) or isinstance(x, bool) for x in indices):
        raise AdapterError("source frame indices must be integers")
    if any(not _finite(x) for x in timestamps):
        raise AdapterError("sample timestamps must be finite")
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise AdapterError("source frame indices must be strictly increasing/distinct")
    times = [float(x) for x in timestamps]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise AdapterError("sample timestamps must be strictly increasing")
    if indices[0] < 0 or indices[-1] >= frame_count:
        raise AdapterError("sample frame index outside source video")
    if times[0] < -EPS or times[-1] > float(duration) + EPS:
        raise AdapterError("sample timestamp outside source duration")
    # Formal records derive timestamps as source_frame_index/source_fps.  This
    # assertion guards against accidentally pairing a token with another grid.
    tolerance = max(1e-6, 0.01 / float(fps))
    if any(abs(t - i / float(fps)) > tolerance for i, t in zip(indices, times)):
        raise AdapterError("stored timestamp is inconsistent with source frame index/FPS")
    return indices, times, float(fps), frame_count, float(duration)


def temporal_boundaries(record: dict[str, Any]) -> list[float]:
    """Return N+1 official PTD cell boundaries for N sampled frames.

    PTD evaluates token intervals as inclusive sampled indices [a,b], or the
    half-open discrete interval [a,b+1].  Its Charades converter places cuts
    between source-frame cells: (f_left + f_right + 1)/(2*fps), with video
    edges 0 and source_frame_count/fps.
    """
    indices, _times, fps, source_frames, _duration = validate_sampling(record)
    boundaries = [0.0]
    boundaries.extend(
        (indices[left] + indices[left + 1] + 1) / (2.0 * fps)
        for left in range(len(indices) - 1)
    )
    boundaries.append(source_frames / fps)
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise AdapterError("derived temporal cells are not strictly increasing")
    return boundaries


def build_time_token_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indices, timestamps, _fps, _count, _duration = validate_sampling(record)
    boundaries = temporal_boundaries(record)
    return {
        f"<t{position}>": {
            "sampled_position": position,
            "source_frame_index": indices[position - 1],
            "timestamp_sec": timestamps[position - 1],
            "cell_support_sec": [boundaries[position - 1], boundaries[position]],
        }
        for position in range(1, len(indices) + 1)
    }


def convert_interval(record: dict[str, Any]) -> tuple[list[int] | None, list[float] | None]:
    interval = record.get("parsed_native_time_interval")
    n = len(record.get("selected_source_frame_indices") or [])
    if (
        not isinstance(interval, list) or len(interval) != 2
        or any(not isinstance(x, int) or isinstance(x, bool) for x in interval)
    ):
        return None, None
    start, end = interval
    if start < 1 or end < start or end > n:
        return list(interval), None
    boundaries = temporal_boundaries(record)
    # One-based inclusive <ta>..<tb> becomes continuous [B[a-1], B[b]].
    return [start, end], [boundaries[start - 1], boundaries[end]]


def canonicalize_ptd_box(value: object) -> list[float] | None:
    """Apply official PTD clamp+axis-sort; reject nonfinite/zero-area boxes."""
    if not isinstance(value, list) or len(value) != 4 or any(not _finite(x) for x in value):
        return None
    x1, y1, x2, y2 = [max(0.0, min(float(x), 1000.0)) for x in value]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _native_tube(record: dict[str, Any], token_map: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, bool]:
    entries = record.get("parsed_native_bbox_tokens")
    if not isinstance(entries, list):
        return [], 0, 0, False
    tokens = [entry.get("time_token") for entry in entries if isinstance(entry, dict)]
    duplicate_tokens = len(tokens) != len(set(tokens))
    tube: list[dict[str, Any]] = []
    valid_count = invalid_count = 0
    for entry in entries:
        token = entry.get("time_token") if isinstance(entry, dict) else None
        original_box = entry.get("bbox_tokens") if isinstance(entry, dict) else None
        mapping = token_map.get(f"<t{token}>") if isinstance(token, int) else None
        canonical = canonicalize_ptd_box(original_box) if mapping is not None and not duplicate_tokens else None
        valid = canonical is not None
        valid_count += int(valid)
        invalid_count += int(not valid)
        tube.append({
            "time_token": f"<t{token}>" if isinstance(token, int) else None,
            "sampled_position": mapping.get("sampled_position") if mapping else None,
            "source_frame_index": mapping.get("source_frame_index") if mapping else None,
            "timestamp_sec": mapping.get("timestamp_sec") if mapping else None,
            "cell_support_sec": mapping.get("cell_support_sec") if mapping else None,
            "native_bbox": original_box,
            "canonical_bbox": canonical,
            "valid": valid,
        })
    tube.sort(key=lambda item: item["sampled_position"] if item["sampled_position"] is not None else math.inf)
    return tube, valid_count, invalid_count, duplicate_tokens


def interpolate_box(tube: list[dict[str, Any]], time_sec: float) -> list[float] | None:
    """Interpolate at time_sec without extrapolating valid PTD boxes."""
    valid = [item for item in tube if item["valid"]]
    if not valid:
        return None
    if len(valid) == 1:
        left, right = valid[0]["cell_support_sec"]
        return list(valid[0]["canonical_bbox"]) if left - EPS <= time_sec <= right + EPS else None
    times = [float(item["timestamp_sec"]) for item in valid]
    offset = bisect_left(times, time_sec)
    if offset < len(times) and abs(times[offset] - time_sec) <= EPS:
        return list(valid[offset]["canonical_bbox"])
    if offset == 0 or offset == len(times):
        return None
    before, after = valid[offset - 1], valid[offset]
    t0, t1 = float(before["timestamp_sec"]), float(after["timestamp_sec"])
    alpha = (time_sec - t0) / (t1 - t0)
    return [
        float(a) + alpha * (float(b) - float(a))
        for a, b in zip(before["canonical_bbox"], after["canonical_bbox"])
    ]


def adapt_role_record(record: dict[str, Any], source_record_path: Path) -> dict[str, Any]:
    """Create the required auditable per-role adapted record."""
    token_map: dict[str, dict[str, Any]] = {}
    tube: list[dict[str, Any]] = []
    valid_count = invalid_count = 0
    duplicate_tokens = False
    interval_tokens: list[int] | None = None
    interval_sec: list[float] | None = None
    errors: list[str] = []
    try:
        token_map = build_time_token_map(record)
        interval_tokens, interval_sec = convert_interval(record)
        if interval_sec is None:
            errors.append("invalid_or_missing_native_interval")
        tube, valid_count, invalid_count, duplicate_tokens = _native_tube(record, token_map)
        if duplicate_tokens:
            errors.append("duplicate_native_bbox_time_token")
        if valid_count == 0:
            errors.append("no_valid_native_bbox")
    except AdapterError as exc:
        errors.append(str(exc))

    original_status = record.get("status")
    if original_status != "OK":
        adapter_status = "SOURCE_NOT_OK"
    elif not token_map:
        adapter_status = "INVALID_SAMPLING_METADATA"
    elif interval_sec is None:
        adapter_status = "INVALID_NATIVE_INTERVAL"
    elif duplicate_tokens:
        adapter_status = "MALFORMED_DUPLICATE_BOX_TOKEN"
    elif valid_count == 0:
        adapter_status = "OK_NO_VALID_BOX"
    else:
        adapter_status = "OK"

    return {
        "adapter_id": ADAPTER_ID,
        "source_record_path": str(source_record_path.resolve()),
        "source_record_sha256": sha256_file(source_record_path),
        "raw_native_output_reference": record.get("raw_native_output_path") or str(source_record_path.resolve()),
        "sample_id": record.get("sample_id"),
        "setting": record.get("setting"),
        "event_type": record.get("event_type"),
        "role_key": record.get("role_key"),
        "source_video": record.get("source_video"),
        "source_duration": record.get("source_duration"),
        "source_fps": record.get("source_fps"),
        "source_frame_count": record.get("source_frame_count"),
        "input_frame_count": record.get("input_frame_count"),
        "selected_source_frame_indices": record.get("selected_source_frame_indices"),
        "selected_timestamps": record.get("selected_timestamps"),
        "native_time_interval_tokens": interval_tokens,
        "native_bbox_tokens": record.get("parsed_native_bbox_tokens"),
        "time_token_map": token_map,
        "predicted_interval_sec": interval_sec,
        "native_tube_timestamped": tube,
        "adapted_role_stvg_tube": [],
        "invalid_bbox_count": invalid_count,
        "valid_bbox_count": valid_count,
        "original_status": original_status,
        "adapter_status": adapter_status,
        "adapter_errors": errors,
    }


def role_rows(adapted: dict[str, Any], evaluation_times: Iterable[float]) -> list[dict[str, Any]]:
    role = adapted.get("role_key")
    interval = adapted.get("predicted_interval_sec")
    tube = adapted.get("native_tube_timestamped") or []
    scorable = adapted.get("adapter_status") in {"OK", "OK_NO_VALID_BOX"}
    times = tuple(float(value) for value in evaluation_times)
    valid = [item for item in tube if item.get("valid")]
    singleton_projection_offset = None
    if scorable and len(valid) == 1:
        source_duration = adapted.get("source_duration")
        candidates = [
            (offset, timestamp) for offset, timestamp in enumerate(times)
            if _finite(timestamp)
            and _finite(source_duration)
            and -EPS <= timestamp <= float(source_duration) + EPS
        ]
        if candidates:
            native_timestamp = float(valid[0]["timestamp_sec"])
            # Timestamp is the secondary key, so an exact-distance tie selects
            # the earlier evaluator timestamp deterministically.
            singleton_projection_offset = min(
                candidates, key=lambda item: (abs(item[1] - native_timestamp), item[1])
            )[0]
    rows = []
    for offset, timestamp in enumerate(times):
        inside = bool(interval and interval[0] - EPS <= timestamp <= interval[1] + EPS)
        if singleton_projection_offset is not None:
            # A singleton has no pair from which interpolation is possible. It
            # receives exactly one discrete representative on the destination
            # grid; the continuous PTD interval itself is left unchanged.
            box = list(valid[0]["canonical_bbox"]) if offset == singleton_projection_offset else None
        else:
            box = interpolate_box(tube, timestamp) if scorable and inside else None
        rows.append({"time_sec": timestamp, role: box})
    return rows


def merge_role_rows(
    role_records: list[dict[str, Any]], evaluation_times: Iterable[float]
) -> list[dict[str, Any]]:
    times = tuple(float(x) for x in evaluation_times)
    rows = [{"time_sec": timestamp} for timestamp in times]
    for adapted in sorted(role_records, key=lambda item: str(item.get("role_key"))):
        # Failed role calls are deliberately absent (not explicit null), matching
        # the frozen evaluator's missing-role failure policy.
        if adapted.get("adapter_status") not in {"OK", "OK_NO_VALID_BOX"}:
            adapted["adapted_role_stvg_tube"] = []
            continue
        generated = role_rows(adapted, times)
        adapted["adapted_role_stvg_tube"] = generated
        role = adapted["role_key"]
        for target, source in zip(rows, generated):
            target[role] = source[role]
    return rows


def union_interval(role_records: Iterable[dict[str, Any]]) -> tuple[float, float] | None:
    intervals = [
        item["predicted_interval_sec"] for item in role_records
        if item.get("adapter_status") in {"OK", "OK_NO_VALID_BOX"}
        and isinstance(item.get("predicted_interval_sec"), list)
    ]
    return (min(x[0] for x in intervals), max(x[1] for x in intervals)) if intervals else None
