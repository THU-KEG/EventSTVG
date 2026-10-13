from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from .models import TargetClip


def probe_frame_rate(path: Path, ffprobe: str = "ffprobe") -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    value = json.loads(result.stdout)["streams"][0]["avg_frame_rate"]
    frame_rate = float(Fraction(value))
    if frame_rate <= 0:
        raise ValueError(f"invalid frame rate for {path}: {value}")
    return frame_rate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corrections(path: Path) -> tuple[dict[str, dict], dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record["sample_id"]
            if sample_id in records:
                raise ValueError(f"duplicate correction for {sample_id} at line {line_number}")
            records[sample_id] = record
    provenance = {
        "timestamp_correction_version": "timestamp-correction-v1",
        "correction_manifest_path": str(path.resolve()),
        "correction_manifest_sha256": sha256_file(path),
    }
    return records, provenance


def load_manual_audit_passes(path: Path) -> tuple[set[str], dict]:
    """Load explicit per-sample manual passes used for source-boundary clamps."""
    passed: set[str] = set()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record["sample_id"]
            if sample_id in seen:
                raise ValueError(
                    f"duplicate manual-audit sample {sample_id} at line {line_number}"
                )
            seen.add(sample_id)
            if record.get("result") == "pass":
                passed.add(sample_id)
    return passed, {
        "manual_audit_manifest_path": str(path.resolve()),
        "manual_audit_manifest_sha256": sha256_file(path),
        "manual_audit_record_count": len(seen),
        "manual_audit_pass_count": len(passed),
    }


def apply_corrections(
    targets: list[TargetClip],
    inventory: list[TargetClip],
    removed: list[dict],
    path: Path,
    allow_probable: bool = False,
    source_boundary_tolerance_sec: float = 0.0,
    source_boundary_audit_pass_manifest: Path | None = None,
    source_boundary_tolerance_frames: int = 2,
    ffprobe: str = "ffprobe",
) -> tuple[list[TargetClip], list[TargetClip], list[dict], dict]:
    records, provenance = load_corrections(path)
    allowed = {"verified"} | ({"probable"} if allow_probable else set())
    audit_passes: set[str] = set()
    audit_provenance: dict = {}
    if source_boundary_audit_pass_manifest is not None:
        audit_passes, audit_provenance = load_manual_audit_passes(
            source_boundary_audit_pass_manifest
        )
    corrected_inventory: list[TargetClip] = []
    excluded_ids: set[str] = set()
    new_removed = list(removed)
    for item in inventory:
        record = records.get(item.sample_id)
        if record is None:
            reason, status = "timestamp_correction_missing", "missing"
        else:
            status = record.get("status")
            reason = (None if status in allowed else
                      f"timestamp_correction_status_{status}")
        if reason:
            excluded_ids.add(item.sample_id)
            new_removed.append({
                "sample_id": item.sample_id, "game": item.game, "half": item.half,
                "event_type": item.event_type, "reason": reason,
                "timestamp_recovery_status": status,
            })
            continue
        raw_start = float(record["corrected_game_ts_start"])
        end = float(record["corrected_game_ts_end"])
        if not end > raw_start:
            raise ValueError(f"invalid corrected interval for {item.sample_id}")
        expected = float(record["local_t_e"]) - float(record["local_t_s"])
        if abs((end - raw_start) - expected) > 1e-6:
            raise ValueError(f"corrected duration mismatch for {item.sample_id}")
        source_frame_rate = None
        frame_tolerance_sec = None
        is_clamp_candidate = (
            source_boundary_tolerance_sec > 0.0
            and raw_start < 0.0
            and end > 0.0
            and status == "verified"
            and item.sample_id in audit_passes
        )
        if is_clamp_candidate:
            source_frame_rate = probe_frame_rate(item.source_half_video, ffprobe)
            frame_tolerance_sec = source_boundary_tolerance_frames / source_frame_rate
        effective_tolerance = (
            min(source_boundary_tolerance_sec, frame_tolerance_sec)
            if frame_tolerance_sec is not None else 0.0
        )
        boundary_clamped = (
            is_clamp_candidate
            and -effective_tolerance - 1e-9 <= raw_start < 0.0
        )
        start = 0.0 if boundary_clamped else raw_start
        clamp_amount = -raw_start if boundary_clamped else 0.0
        correction = {
            **provenance,
            "recovery_status": status,
            "original_stored_game_ts_start": float(record["stored_game_ts_start"]),
            "original_stored_game_ts_end": float(record["stored_game_ts_end"]),
            "corrected_game_ts_start": start,
            "corrected_game_ts_end": end,
            "raw_corrected_game_ts_start": raw_start,
            "source_boundary_clamped": boundary_clamped,
            "source_boundary_tolerance_sec": source_boundary_tolerance_sec,
            "source_boundary_tolerance_frames": source_boundary_tolerance_frames,
            "source_frame_rate_fps": source_frame_rate,
            "source_frame_tolerance_sec": frame_tolerance_sec,
            "source_boundary_clamp_amount_sec": clamp_amount,
            "source_boundary_clamp_reason": (
                "verified_manual_pass_within_two_source_frames"
                if boundary_clamped else None
            ),
            "timestamp_offset_sec": float(record["offset_sec"]),
            "recovery_method": record.get("recovery_method"),
            "recovery_confidence": record.get("confidence"),
        }
        corrected_inventory.append(replace(
            item, clip_start=start, clip_end=end,
            source_timestamp_in_trim=start <= item.target_event_time <= end,
            timestamp_correction=correction,
        ))
    by_id = {item.sample_id: item for item in corrected_inventory}
    corrected_targets = [by_id[item.sample_id] for item in targets
                         if item.sample_id in by_id]
    expected_ids = {item.sample_id for item in inventory}
    extra_ids = set(records) - expected_ids
    provenance.update({
        "allow_probable_corrections": allow_probable,
        "correction_record_count": len(records),
        "correction_extra_record_count": len(extra_ids),
        "correction_excluded_count": len(excluded_ids),
        "source_boundary_tolerance_sec": source_boundary_tolerance_sec,
        "source_boundary_tolerance_frames": source_boundary_tolerance_frames,
        "source_boundary_clamped_count": sum(
            item.timestamp_correction.get("source_boundary_clamped", False)
            for item in corrected_inventory
            if item.timestamp_correction is not None
        ),
        **audit_provenance,
    })
    return corrected_targets, corrected_inventory, new_removed, provenance
