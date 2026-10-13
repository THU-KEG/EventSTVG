from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .events import (
    STEM_RE,
    EventIndex,
    normalized_event_type,
    occurrences_from_labels,
    sequence_event_type,
)
from .models import TargetClip
from .quality_control import invalid_sample_reason


@dataclass(frozen=True)
class Roots:
    trimmed_clips: Path
    role_annotations: Path
    trim_annotations: Path
    soccernet_data: Path


def validate_roots(roots: Roots) -> None:
    missing = [str(path) for path in roots.__dict__.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing input directories: " + ", ".join(missing))


def build_event_index(roots: Roots) -> EventIndex:
    occurrences = []
    for labels_path in sorted(roots.soccernet_data.rglob("Labels-v2.json")):
        game = labels_path.parent.relative_to(roots.soccernet_data).as_posix()
        occurrences.extend(occurrences_from_labels(labels_path, game))
    return EventIndex(occurrences)


def _trim_path(roots: Roots, relative_parent: Path, stem: str) -> Path:
    return roots.trim_annotations / relative_parent / f"{stem}_trim.json"


def discover_trim_inventory(
    roots: Roots, event_index: EventIndex, match_tolerance: float,
    manual_exclusions: dict[str, dict] | None = None,
) -> tuple[list[TargetClip], list[TargetClip], list[dict]]:
    targets: list[TargetClip] = []
    inventory: list[TargetClip] = []
    removed: list[dict] = []
    for clip in sorted(roots.trimmed_clips.rglob("*.mp4")):
        relative = clip.relative_to(roots.trimmed_clips)
        match = STEM_RE.fullmatch(clip.stem)
        base = {"sample_id": relative.with_suffix("").as_posix(),
                "source_clip": str(clip.resolve())}
        if match is None or len(relative.parts) < 4:
            removed.append({**base, "reason": "unparseable_clip_path"})
            continue
        game = relative.parent.as_posix()
        half = int(match.group("half"))
        event_position = int(match.group("position")) / 1000.0
        event_type = normalized_event_type(
            match.group("label"), match.group("window")
        )
        sequence_type = sequence_event_type(event_type)
        base["event_type"] = event_type
        base["sequence_event_type"] = sequence_type
        problem = (
            roots.role_annotations / relative.parent /
            f"{clip.stem}_problem.json"
        )
        if problem.is_file():
            removed.append({
                **base,
                "reason": "problem_flag_present",
                "problem_flag": str(problem.resolve()),
                "quality_control_exclusion": True,
            })
            continue
        manual_exclusion = (manual_exclusions or {}).get(base["sample_id"])
        if manual_exclusion is not None:
            removed.append({
                **base,
                "reason": manual_exclusion["reason"],
                "quality_control_exclusion": True,
                "manual_review_exclusion": True,
                "review_group_id": manual_exclusion["review_group_id"],
                "kept_sample_id": manual_exclusion["kept_sample_id"],
            })
            continue
        quality_reason = invalid_sample_reason(base["sample_id"])
        if quality_reason is not None:
            removed.append({
                **base,
                "reason": quality_reason,
                "quality_control_exclusion": True,
            })
            continue
        role = roots.role_annotations / relative.parent / f"{clip.stem}_annotation.json"
        trim = _trim_path(roots, relative.parent, clip.stem)
        labels = roots.soccernet_data / relative.parent / "Labels-v2.json"
        half_video = roots.soccernet_data / relative.parent / f"{half}_720p.mp4"
        required = (
            ("trim_annotation_not_available", trim),
            ("soccernet_labels_not_available", labels),
            ("source_half_video_not_available", half_video),
        )
        missing = [
            {"kind": reason, "path": str(path)}
            for reason, path in required if not path.is_file()
        ]
        if missing:
            reason = missing[0]["kind"] if len(missing) == 1 else "multiple_inputs_not_available"
            removed.append({**base, "reason": reason, "missing": missing})
            continue
        try:
            trim_data = json.loads(trim.read_text(encoding="utf-8"))
            clip_start = float(trim_data["game_ts_start"])
            clip_end = float(trim_data["game_ts_end"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            removed.append({**base, "reason": "invalid_trim_annotation",
                            "detail": str(exc)})
            continue
        sequence = event_index.sequence(game, half, sequence_type)
        if not sequence:
            removed.append({**base, "reason": "no_occurrence_sequence",
                            "event_type": event_type})
            continue
        matches = event_index.match(
            game, half, sequence_type, int(match.group("position")),
            match_tolerance,
        )
        if not matches:
            removed.append({
                **base, "reason": "unmatched_stage1_to_soccernet",
                "source_position_ms": int(match.group("position")),
            })
            continue
        if len(matches) > 1:
            removed.append({
                **base, "reason": "ambiguous_stage1_to_soccernet_match",
                "source_position_ms": int(match.group("position")),
                "candidate_source_occurrence_ids": [
                    item.source_occurrence_id for item in matches
                ],
            })
            continue
        occurrence = matches[0]
        target_event_time = occurrence.soccernet_event_time_abs
        source_timestamp_in_trim = (
            clip_start <= target_event_time <= clip_end
        )
        item = TargetClip(
            sample_id=relative.with_suffix("").as_posix(),
            game=game, half=half, source_clip=clip.resolve(),
            role_annotation_path=role.resolve() if role.is_file() else None,
            trim_annotation_path=trim.resolve(),
            source_half_video=half_video.resolve(),
            filename_label=match.group("label"), event_type=event_type,
            sequence_event_type=sequence_type,
            event_position=event_position, target_event_time=target_event_time,
            clip_start=clip_start, clip_end=clip_end,
            occurrence=occurrence,
            source_timestamp_in_trim=source_timestamp_in_trim,
        )
        inventory.append(item)
        if role.is_file():
            targets.append(item)
        else:
            removed.append({
                **base,
                "reason": "role_annotation_not_available",
                "missing": [{
                    "kind": "role_annotation_not_available",
                    "path": str(role),
                }],
            })
    return targets, inventory, removed


def discover_targets(
    roots: Roots, event_index: EventIndex, match_tolerance: float
) -> tuple[list[TargetClip], list[dict]]:
    """Compatibility wrapper returning only eligible annotated targets."""
    targets, _inventory, removed = discover_trim_inventory(
        roots, event_index, match_tolerance
    )
    return targets, removed


def probe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    command = [
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return float(result.stdout.strip())


def source_coverage(roots: Roots, event_index: EventIndex,
                    match_tolerance: float) -> dict:
    matched_ids: set[str] = set()
    unmatched = 0
    ambiguous = 0
    paired: dict[str, dict[str, str]] = {}
    for clip in sorted(roots.trimmed_clips.rglob("*.mp4")):
        relative = clip.relative_to(roots.trimmed_clips)
        match = STEM_RE.fullmatch(clip.stem)
        if match is None:
            continue
        target_type = normalized_event_type(
            match.group("label"), match.group("window")
        )
        sequence_type = sequence_event_type(target_type)
        game = relative.parent.as_posix()
        candidates = event_index.match(
            game, int(match.group("half")), sequence_type,
            int(match.group("position")), match_tolerance,
        )
        if not candidates:
            unmatched += 1
            continue
        if len(candidates) > 1:
            ambiguous += 1
            continue
        source_id = candidates[0].source_occurrence_id
        matched_ids.add(source_id)
        window = match.group("window")
        if window:
            base = re.sub(r"_w[12]$", "", relative.with_suffix("").as_posix())
            paired.setdefault(base, {})[window] = source_id
    matched_pairs = sum(
        set(windows) == {"w1", "w2"}
        and windows["w1"] == windows["w2"]
        for windows in paired.values()
    )
    conflicting_pairs = sum(
        set(windows) == {"w1", "w2"}
        and windows["w1"] != windows["w2"]
        for windows in paired.values()
    )
    return {
        "total_relevant_labels_v2_occurrences": len(event_index.occurrences),
        "stage1_clips_matched_to_soccernet": len(matched_ids),
        "matched_paired_w1_w2_groups": matched_pairs,
        "conflicting_paired_w1_w2_groups": conflicting_pairs,
        "unmatched_stage1_clips": unmatched,
        "ambiguous_stage1_matches": ambiguous,
        "labels_v2_occurrences_without_stage1_clips":
            len(event_index.occurrences) - len(matched_ids),
    }
