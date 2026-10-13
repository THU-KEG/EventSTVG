from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from .events import EventIndex
from .models import EventOccurrence, TargetClip
from .sources import probe_duration
from .windows import (
    EPSILON,
    contained_same_type_trims,
    generate_window,
    stable_seed,
)


CONTEXTS = (
    "oracle", "context_1min", "context_3min", "context_5min", "context_10min",
)
GENERATION_POLICY_VERSION = 7
SUPPORTED_POLICY_VERSIONS = (7, 8)
OCCURRENCE_COUNTING_POLICY = (
    "qualified_soccernet_target_or_timestamp_or_contained_trim_v1"
)


def target_clip_within_half(target: TargetClip, half_duration: float) -> bool:
    return (
        target.clip_start >= 0.0
        and target.clip_end > target.clip_start
        and target.clip_end <= half_duration
    )


def _source_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _generation_provenance(
    target: TargetClip,
    context_type: str,
    start: float,
    end: float,
    global_seed: int,
    safety_margin_sec: float,
    policy_version: int = GENERATION_POLICY_VERSION,
) -> dict:
    sample_seed = stable_seed(global_seed, target.sample_id, context_type)
    configuration = {
        "policy_version": policy_version,
        "context_type": context_type,
        "context_start": round(start, 6),
        "context_end": round(end, 6),
        "expected_duration": round(end - start, 6),
        "global_seed": global_seed,
        "sample_seed": sample_seed,
        "safety_margin_sec": safety_margin_sec,
        "occurrence_counting_policy": OCCURRENCE_COUNTING_POLICY,
        "source_video": _source_signature(
            target.source_clip
            if context_type == "oracle" else target.source_half_video
        ),
    }
    if target.timestamp_correction is not None:
        configuration["timestamp_correction"] = target.timestamp_correction
    fingerprint = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**configuration, "generation_fingerprint": fingerprint}


def _same_type_trim_count(
    target: TargetClip,
    inventory: list[TargetClip],
    start: float,
    end: float,
) -> int:
    return sum(
        trim.game == target.game
        and trim.half == target.half
        and trim.event_type == target.event_type
        and start <= trim.clip_start + EPSILON
        and trim.clip_end <= end + EPSILON
        for trim in inventory
    )


def _sample_metadata(
    target: TargetClip,
    context_type: str,
    start: float,
    end: float,
    contained_trims: tuple[TargetClip, ...],
    order: int,
    checks: dict[str, bool],
    output_video: Path,
    trim_inventory: list[TargetClip],
    global_seed: int,
    safety_margin_sec: float,
    qualified_occurrences: tuple[EventOccurrence, ...] | None = None,
    policy_version: int = GENERATION_POLICY_VERSION,
) -> dict:
    provenance = _generation_provenance(
        target, context_type, start, end, global_seed, safety_margin_sec,
        policy_version,
    )
    timestamp = target.target_event_time
    trim_occurrence_ids = [
        trim.occurrence.source_occurrence_id for trim in contained_trims
    ]
    if qualified_occurrences is None:
        qualified_occurrences = tuple(trim.occurrence for trim in contained_trims)
    qualified_occurrence_ids = [
        occurrence.source_occurrence_id for occurrence in qualified_occurrences
    ]
    metadata = {
        "sample_id": target.sample_id,
        "game": target.game,
        "half": target.half,
        "context_type": context_type,
        "context_policy_version": policy_version,
        "video_path": str(output_video),
        "source_half_video": str(target.source_half_video),
        "source_clip": str(target.source_clip),
        "role_annotation_path": (
            str(target.role_annotation_path)
            if target.role_annotation_path is not None else None
        ),
        "trim_annotation_path": str(target.trim_annotation_path),
        "target_role_stvg_type": target.event_type,
        "matched_source_sequence_type": target.sequence_event_type,
        "matched_source_occurrence_id":
            target.occurrence.source_occurrence_id,
        "matched_source_timestamp": timestamp,
        "source_timestamp_inside_context":
            start - EPSILON <= timestamp <= end + EPSILON,
        "target_trim_source_start": target.clip_start,
        "target_trim_source_end": target.clip_end,
        "target_trim_start_rel": target.clip_start - start,
        "target_trim_end_rel": target.clip_end - start,
        "context_start": start,
        "context_end": end,
        "context_duration": end - start,
        "contained_same_type_trim_ids": [
            trim.sample_id for trim in contained_trims
        ],
        "contained_same_type_occurrence_ids": trim_occurrence_ids,
        "qualified_source_occurrence_ids": qualified_occurrence_ids,
        "occurrence_counting_policy": OCCURRENCE_COUNTING_POLICY,
        "same_type_trim_count": _same_type_trim_count(
            target, trim_inventory, start, end
        ),
        "same_source_occurrence_count": len(qualified_occurrence_ids),
        "same_event_count": len(qualified_occurrence_ids),
        "num_same_event_distractors": len(qualified_occurrence_ids) - 1,
        "target_event_order": order,
        "num_prior_same_events": order - 1,
        "requires_disambiguation": len(qualified_occurrence_ids) > 1,
        "safety_margin_sec": (
            safety_margin_sec if context_type == "context_1min" else 0.0
        ),
        "global_seed": global_seed,
        "sample_seed": provenance["sample_seed"],
        "generation_fingerprint": provenance["generation_fingerprint"],
        "source_video_signature": provenance["source_video"],
        "window_validity_checks": checks,
        # Compatibility aliases. Count/order follow the qualified SoccerNet
        # union; contained_* fields continue to describe retained trims.
        "event_type": target.event_type,
        "sequence_event_type": target.sequence_event_type,
        "source_occurrence_id": target.occurrence.source_occurrence_id,
        "target_event_time_source": "SoccerNet Labels-v2.json (indexing only)",
        "target_event_time_abs": timestamp,
        "target_clip_start_abs": target.clip_start,
        "target_clip_end_abs": target.clip_end,
        "context_start_abs": start,
        "context_end_abs": end,
        "target_clip_start_rel": target.clip_start - start,
        "target_clip_end_rel": target.clip_end - start,
        "target_event_time_rel": timestamp - start,
    }
    if target.timestamp_correction is not None:
        metadata["timestamp_correction"] = target.timestamp_correction
    return metadata


def _run_ffmpeg(source: Path, destination: Path, start: float, duration: float):
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-avoid_negative_ts", "make_zero", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def _provenance_path(video_path: Path) -> Path:
    return video_path.with_suffix(video_path.suffix + ".provenance.json")


def _write_provenance(path: Path, provenance: dict) -> None:
    _provenance_path(path).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_existing_video(
    video_path: Path,
    expected: dict,
    ffprobe: str,
    duration_tolerance_sec: float = 0.25,
) -> None:
    provenance_path = _provenance_path(video_path)
    if not provenance_path.is_file():
        raise RuntimeError(
            f"Existing context video has no provenance: {video_path}. "
            "Use --overwrite or a clean output directory."
        )
    try:
        actual = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read provenance for existing video {video_path}: {exc}"
        ) from exc
    if actual != expected:
        raise RuntimeError(
            f"Existing context video provenance mismatch: {video_path}. "
            "Use --overwrite or a clean output directory."
        )
    actual_duration = probe_duration(video_path, ffprobe)
    if abs(actual_duration - expected["expected_duration"]) > duration_tolerance_sec:
        raise RuntimeError(
            f"Existing context video duration mismatch for {video_path}: "
            f"expected {expected['expected_duration']}, got {actual_duration}"
        )


def _stats(retained: list[dict], include_clean_repeated: bool = True) -> dict:
    stats = {
        "total_samples": len(retained),
        "event_category_distribution": dict(sorted(Counter(
            item["target_role_stvg_type"] for item in retained
        ).items())),
        "source_timestamp_inside_context_distribution": dict(sorted(Counter(
            str(item["source_timestamp_inside_context"]).lower()
            for item in retained
        ).items())),
    }
    if include_clean_repeated:
        stats.update({
            "same_event_count_distribution": dict(sorted(Counter(
                str(item["same_event_count"]) for item in retained
            ).items(), key=lambda pair: int(pair[0]))),
            "target_event_order_distribution": dict(sorted(Counter(
                str(item["target_event_order"]) for item in retained
            ).items(), key=lambda pair: int(pair[0]))),
            "num_prior_same_events_distribution": dict(sorted(Counter(
                str(item["num_prior_same_events"]) for item in retained
            ).items(), key=lambda pair: int(pair[0]))),
            "num_same_event_distractors_distribution": dict(sorted(Counter(
                str(item["num_same_event_distractors"]) for item in retained
            ).items(), key=lambda pair: int(pair[0]))),
            "requires_disambiguation_distribution": dict(sorted(Counter(
                str(item["requires_disambiguation"]).lower() for item in retained
            ).items())),
        })
        stats["clean_repeated_distribution"] = {
            "clean": sum(item["same_event_count"] == 1 for item in retained),
            "repeated": sum(item["same_event_count"] > 1 for item in retained),
        }
    return stats


def _removed_type_distribution(removed: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(
        item.get("event_type", "unknown") for item in removed
    ).items()))


def generate_all(
    targets: list[TargetClip],
    trim_inventory: list[TargetClip],
    discovery_removed: list[dict],
    event_index: EventIndex,
    output_root: Path,
    seed: int,
    dry_run: bool,
    overwrite: bool,
    ffprobe: str,
    safety_margin_sec: float = 10.0,
    manifests_only: bool = False,
    correction_provenance: dict | None = None,
    existing_context_video_root: Path | None = None,
    policy_version: int = GENERATION_POLICY_VERSION,
    manual_exclusion_provenance: dict | None = None,
    contexts: tuple[str, ...] = CONTEXTS,
) -> dict:
    durations: dict[Path, float] = {}
    all_summary = {}
    unknown_contexts = set(contexts) - set(CONTEXTS)
    if unknown_contexts:
        raise ValueError(f"unsupported contexts: {sorted(unknown_contexts)}")
    for context_type in contexts:
        dataset_dir = output_root / context_type
        retained: list[dict] = []
        removed = list(discovery_removed)
        for target in targets:
            sequence = event_index.sequence(
                target.game, target.half, target.sequence_event_type
            )
            try:
                if target.source_half_video not in durations:
                    durations[target.source_half_video] = probe_duration(
                        target.source_half_video, ffprobe
                    )
                half_duration = durations[target.source_half_video]
            except Exception as exc:
                removed.append({
                    "sample_id": target.sample_id,
                    "event_type": target.event_type,
                    "reason": "half_duration_probe_failed",
                    "detail": str(exc),
                })
                continue
            if not target_clip_within_half(target, half_duration):
                removed.append({
                    "sample_id": target.sample_id,
                    "game": target.game,
                    "half": target.half,
                    "event_type": target.event_type,
                    "context_type": context_type,
                    "reason": "target_trim_outside_half_video",
                    "target_trim_source_start": target.clip_start,
                    "target_trim_source_end": target.clip_end,
                    "half_duration": half_duration,
                    "window_validity_checks": {
                        "valid_target_trim_interval":
                            target.clip_end > target.clip_start,
                        "target_trim_starts_inside_half":
                            target.clip_start >= 0.0,
                        "target_trim_ends_inside_half":
                            target.clip_end <= half_duration,
                    },
                })
                continue

            if context_type == "oracle":
                start, end = target.clip_start, target.clip_end
                try:
                    contained = contained_same_type_trims(
                        target, trim_inventory, start, end
                    )
                except ValueError as exc:
                    removed.append({
                        "sample_id": target.sample_id,
                        "event_type": target.event_type,
                        "context_type": context_type,
                        "reason": "target_trim_missing_from_inventory",
                        "detail": str(exc),
                    })
                    continue
                order = (1 if policy_version >= 8 else next(
                    i + 1 for i, trim in enumerate(contained)
                    if trim.occurrence.source_occurrence_id
                    == target.occurrence.source_occurrence_id
                ))
                checks = {
                    "contains_target_trim": True,
                    "within_half": True,
                    "target_trim_present": True,
                    "valid_source_occurrence_match": True,
                    "oracle_contains_target_occurrence": len(contained) >= 1,
                    "oracle_target_event_order_valid":
                        1 <= order <= len(contained),
                }
                oracle_metadata = _sample_metadata(
                    target, context_type, start, end, contained, order, checks,
                    target.source_clip, trim_inventory, seed, safety_margin_sec,
                    qualified_occurrences=(target.occurrence,)
                    if policy_version >= 8 else None,
                    policy_version=policy_version,
                )
                if policy_version >= 8:
                    for field in (
                        "qualified_source_occurrence_ids",
                        "contained_same_type_occurrence_ids",
                        "contained_same_type_trim_ids",
                        "same_event_count", "target_event_order",
                        "num_prior_same_events", "num_same_event_distractors",
                        "requires_disambiguation", "safety_margin_sec",
                    ):
                        oracle_metadata.pop(field, None)
                    oracle_metadata["target_source_occurrence_id"] = (
                        target.occurrence.source_occurrence_id
                    )
                retained.append(oracle_metadata)
                continue

            result = generate_window(
                target, trim_inventory, sequence, context_type, half_duration,
                seed, safety_margin_sec, policy_version,
            )
            if not result.retained:
                removed.append({
                    "sample_id": target.sample_id,
                    "game": target.game,
                    "half": target.half,
                    "event_type": target.event_type,
                    "sequence_event_type": target.sequence_event_type,
                    "context_type": context_type,
                    "reason": result.reason,
                    "window_validity_checks": result.checks,
                })
                continue
            relative_video = Path(target.sample_id).with_suffix(".mp4")
            video_dataset_dir = (
                existing_context_video_root / context_type
                if existing_context_video_root is not None else dataset_dir
            )
            output_video = video_dataset_dir / "videos" / relative_video
            metadata = _sample_metadata(
                target, context_type, result.start, result.end,
                result.contained_same_type_trims, result.target_order,
                result.checks, output_video.resolve(), trim_inventory, seed,
                safety_margin_sec, result.contained_same_type_occurrences,
                policy_version,
            )
            if policy_version >= 8:
                metadata.pop("safety_margin_sec", None)
            retained.append(metadata)
            if not dry_run and not manifests_only:
                provenance = _generation_provenance(
                    target, context_type, result.start, result.end, seed,
                    safety_margin_sec, policy_version,
                )
                if output_video.exists() and not overwrite:
                    _validate_existing_video(output_video, provenance, ffprobe)
                else:
                    _run_ffmpeg(
                        target.source_half_video, output_video, result.start,
                        result.end - result.start,
                    )
                    actual_duration = probe_duration(output_video, ffprobe)
                    if abs(actual_duration - provenance["expected_duration"]) > 0.25:
                        raise RuntimeError(
                            f"Generated video duration mismatch for {output_video}: "
                            f"expected {provenance['expected_duration']}, "
                            f"got {actual_duration}"
                        )
                    _write_provenance(output_video, provenance)

        summary = {
            "context_type": context_type,
            "dry_run": dry_run,
            "manifests_only": manifests_only,
            "global_seed": seed,
            "context_policy_version": policy_version,
            "occurrence_counting_policy": OCCURRENCE_COUNTING_POLICY,
            "safety_margin_sec":
                safety_margin_sec if policy_version < 8 and context_type == "context_1min" else 0.0,
            **_stats(retained, include_clean_repeated=(
                context_type != "oracle" or policy_version < 8
            )),
            "removed_samples": len(removed),
            "removed_event_category_distribution":
                _removed_type_distribution(removed),
            "removal_reason_distribution": dict(sorted(Counter(
                item["reason"] for item in removed
            ).items())),
            "all_window_validity_checks_passed": all(
                all(item["window_validity_checks"].values())
                for item in retained
            ),
        }
        if policy_version >= 8:
            summary.pop("safety_margin_sec", None)
            if context_type == "oracle":
                for field in (
                    "same_event_count_distribution",
                    "target_event_order_distribution",
                    "num_prior_same_events_distribution",
                    "num_same_event_distractors_distribution",
                    "requires_disambiguation_distribution",
                ):
                    summary.pop(field, None)
        if context_type == "oracle":
            summary.pop("clean_repeated_distribution", None)
        if correction_provenance is not None:
            summary["timestamp_correction"] = correction_provenance
        if manual_exclusion_provenance is not None:
            summary["manual_exclusions"] = manual_exclusion_provenance
        all_summary[context_type] = summary
        dataset_dir.mkdir(parents=True, exist_ok=True)
        configuration = {
            "context_policy_version": policy_version,
            "occurrence_counting_policy": OCCURRENCE_COUNTING_POLICY,
            "existing_context_video_root": (
                str(existing_context_video_root.resolve())
                if existing_context_video_root is not None else None
            ),
            "global_seed": seed,
            "trim_inventory_size": len(trim_inventory),
            "eligible_target_count": len(targets),
            "timestamp_correction": correction_provenance,
            "manual_exclusions": manual_exclusion_provenance,
        }
        if policy_version < 8:
            configuration["safety_margin_sec"] = (
                safety_margin_sec if context_type == "context_1min" else 0.0
            )
        (dataset_dir / "manifest.json").write_text(json.dumps({
            "configuration": configuration,
            "summary": summary,
            "retained": [] if dry_run else retained,
            "removed": removed,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (dataset_dir / "retained.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            if not dry_run:
                for item in retained:
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
        with (dataset_dir / "removed.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for item in removed:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    return all_summary
