from __future__ import annotations

import hashlib
import random

from .models import EventOccurrence, TargetClip, WindowResult


EPSILON = 1e-6


def stable_seed(global_seed: int, sample_id: str, context_type: str) -> int:
    digest = hashlib.sha256(
        f"{global_seed}|{sample_id}|{context_type}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def base_interval(target: TargetClip, duration: float, half_duration: float):
    """Starts whose window contains the complete Role-STVG target trim."""
    return (
        max(0.0, target.clip_end - duration),
        min(target.clip_start, half_duration - duration),
    )


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    valid = sorted((lo, hi) for lo, hi in intervals if lo <= hi)
    merged: list[list[float]] = []
    for lo, hi in valid:
        if not merged or lo > merged[-1][1] + EPSILON:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return [(lo, hi) for lo, hi in merged]


def choose_interval(intervals: list[tuple[float, float]], seed: int) -> float | None:
    """Select uniformly over the merged union's continuous length."""
    merged = _merge_intervals(intervals)
    if not merged:
        return None
    rng = random.Random(seed)
    lengths = [max(0.0, hi - lo) for lo, hi in merged]
    total = sum(lengths)
    if total <= EPSILON:
        return merged[rng.randrange(len(merged))][0]
    draw = rng.random() * total
    for (lo, hi), length in zip(merged, lengths):
        if draw <= length:
            return lo + draw
        draw -= length
    return merged[-1][1]


def _subtract_closed(
    intervals: list[tuple[float, float]], forbidden: tuple[float, float]
) -> list[tuple[float, float]]:
    """Remove a closed forbidden range, retaining a tiny numeric separation."""
    bad_lo, bad_hi = forbidden
    result: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if bad_hi < lo or bad_lo > hi:
            result.append((lo, hi))
            continue
        left_hi = min(hi, bad_lo - EPSILON)
        right_lo = max(lo, bad_hi + EPSILON)
        if lo <= left_hi:
            result.append((lo, left_hi))
        if right_lo <= hi:
            result.append((right_lo, hi))
    return _merge_intervals(result)


def _subtract_open(
    intervals: list[tuple[float, float]],
    forbidden: tuple[float, float],
) -> list[tuple[float, float]]:
    bad_lo, bad_hi = forbidden
    result: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if bad_hi <= lo or bad_lo >= hi:
            result.append((lo, hi))
            continue
        if lo <= bad_lo:
            result.append((lo, min(hi, bad_lo)))
        if bad_hi <= hi:
            result.append((max(lo, bad_hi), hi))
    return _merge_intervals(result)


def _subtract_trim_overlap(
    intervals: list[tuple[float, float]],
    trim: TargetClip,
    duration: float,
) -> list[tuple[float, float]]:
    # W=[s,s+D] overlaps T=[a,b] iff s < b and s+D > a. Touching at
    # exactly one boundary is allowed, so the forbidden start range is open.
    return _subtract_open(
        intervals, (trim.clip_start - duration, trim.clip_end)
    )


def _exclude_partial_trim_overlap(
    intervals: list[tuple[float, float]],
    trim: TargetClip,
    duration: float,
) -> list[tuple[float, float]]:
    """Keep starts for which trim is fully contained or fully disjoint."""
    # Left-boundary partial overlap: s < a < s+D < b.
    intervals = _subtract_open(
        intervals,
        (trim.clip_start - duration, trim.clip_end - duration),
    )
    # Right-boundary partial overlap: a < s < b < s+D.
    return _subtract_open(
        intervals, (trim.clip_start, trim.clip_end)
    )


def _fully_contained(trim: TargetClip, start: float, end: float) -> bool:
    return (
        start <= trim.clip_start + EPSILON
        and trim.clip_end <= end + EPSILON
    )


def _overlaps(trim: TargetClip, start: float, end: float) -> bool:
    return trim.clip_start < end and trim.clip_end > start


def contained_same_type_trims(
    target: TargetClip,
    trim_inventory: list[TargetClip],
    start: float,
    end: float,
) -> tuple[TargetClip, ...]:
    """Fully contained same-Role-STVG-type trims, deduplicated by occurrence."""
    by_occurrence: dict[str, TargetClip] = {}
    for trim in trim_inventory:
        if (
            trim.game == target.game
            and trim.half == target.half
            and trim.event_type == target.event_type
            and _fully_contained(trim, start, end)
        ):
            occurrence_id = trim.occurrence.source_occurrence_id
            existing = by_occurrence.get(occurrence_id)
            # Prefer the actual target when duplicate trims map to its occurrence.
            if existing is None or trim.sample_id == target.sample_id:
                by_occurrence[occurrence_id] = trim
    if target.occurrence.source_occurrence_id not in by_occurrence:
        raise ValueError("target trim is not represented in its contained inventory")
    return tuple(sorted(
        by_occurrence.values(),
        key=lambda item: (
            item.occurrence.soccernet_event_time_abs,
            item.occurrence.source_occurrence_id,
        ),
    ))


def qualified_context_occurrences(
    target: TargetClip,
    sequence: list[EventOccurrence],
    contained_trims: tuple[TargetClip, ...],
    start: float,
    end: float,
) -> tuple[EventOccurrence, ...]:
    """Return the exact-query occurrence union used for count and ordinal.

    The target is always included when its trim is contained. Other exact-query
    occurrences enter through either their canonical SoccerNet timestamp or a
    fully contained retained trim. The caller supplies the already qualified
    SoccerNet sequence, so foul and other compound query scopes are preserved.
    """
    by_id: dict[str, EventOccurrence] = {
        target.occurrence.source_occurrence_id: target.occurrence,
    }
    for occurrence in sequence:
        if start - EPSILON <= occurrence.soccernet_event_time_abs <= end + EPSILON:
            by_id[occurrence.source_occurrence_id] = occurrence
    for trim in contained_trims:
        by_id[trim.occurrence.source_occurrence_id] = trim.occurrence
    return tuple(sorted(
        by_id.values(),
        key=lambda item: (
            item.soccernet_event_time_abs,
            item.source_occurrence_id,
        ),
    ))


def _result_for_start(
    target: TargetClip,
    trim_inventory: list[TargetClip],
    sequence: list[EventOccurrence],
    context_type: str,
    start: float,
    duration: float,
    half_duration: float,
    policy_version: int = 7,
) -> WindowResult:
    end = start + duration
    trims = contained_same_type_trims(
        target, trim_inventory, start, end
    )
    occurrences = qualified_context_occurrences(
        target, sequence, trims, start, end
    )
    target_order = next(
        i + 1 for i, occurrence in enumerate(occurrences)
        if occurrence.source_occurrence_id
        == target.occurrence.source_occurrence_id
    )
    count = len(occurrences)
    same_type_boundary_complete = all(
        trim.event_type != target.event_type
        or not _overlaps(trim, start, end)
        or _fully_contained(trim, start, end)
        for trim in trim_inventory
    )
    checks = {
        "base_feasible": True,
        "context_constraint_feasible": True,
        "contains_target_trim": _fully_contained(target, start, end),
        "within_half": start >= -EPSILON and end <= half_duration + EPSILON,
        "target_trim_present": any(
            trim.sample_id == target.sample_id for trim in trims
        ),
        "same_event_count_valid": (count >= 1 if policy_version >= 8 else
            (count == 1 if context_type == "context_1min" else count >= 1)),
        "five_min_trim_backed_distractor_valid": (True if policy_version >= 8 else
            (len(trims) >= 2 if context_type == "context_5min" else True)),
        "target_event_order_valid": 1 <= target_order <= count,
        "same_type_boundary_complete": (True if policy_version >= 8 else
                                         same_type_boundary_complete),
    }
    if not all(checks.values()):
        return WindowResult(
            False, "post_selection_validation_failed", checks=checks
        )
    return WindowResult(
        True,
        None,
        start,
        end,
        trims,
        occurrences,
        target_order,
        checks,
    )


def generate_window(
    target: TargetClip,
    trim_inventory: list[TargetClip],
    sequence: list[EventOccurrence],
    context_type: str,
    half_duration: float,
    global_seed: int,
    safety_margin_sec: float = 10.0,
    policy_version: int = 7,
) -> WindowResult:
    durations = {
        "context_1min": 60.0,
        "context_3min": 180.0,
        "context_5min": 300.0,
        "context_10min": 600.0,
    }
    duration = durations[context_type]
    lower, upper = base_interval(target, duration, half_duration)
    if lower > upper + EPSILON:
        return WindowResult(False, "target_trim_not_feasible_in_context", checks={
            "base_feasible": False,
            "contains_target_trim": False,
            "within_half": False,
        })

    source_index = next(
        (i for i, occurrence in enumerate(sequence)
         if occurrence.source_occurrence_id
         == target.occurrence.source_occurrence_id),
        None,
    )
    if source_index is None:
        return WindowResult(False, "target_occurrence_not_in_source_sequence")

    game_half_trims = [
        trim for trim in trim_inventory
        if trim.game == target.game and trim.half == target.half
    ]
    intervals: list[tuple[float, float]] = [(lower, upper)]
    empty_reason: str | None = None

    if policy_version >= 8:
        pass
    elif context_type == "context_1min":
        for other in game_half_trims:
            if (
                other.sample_id != target.sample_id
                and other.event_type == target.event_type
            ):
                intervals = _subtract_trim_overlap(intervals, other, duration)
        for occurrence in sequence:
            if (
                occurrence.source_occurrence_id
                != target.occurrence.source_occurrence_id
            ):
                timestamp = occurrence.soccernet_event_time_abs
                intervals = _subtract_closed(
                    intervals,
                    (
                        timestamp - duration - safety_margin_sec,
                        timestamp + safety_margin_sec,
                    ),
                )
    elif context_type == "context_5min":
        candidates: list[tuple[float, float]] = []
        for other in game_half_trims:
            if (
                other.sample_id != target.sample_id
                and other.event_type == target.event_type
                and other.occurrence.source_occurrence_id
                != target.occurrence.source_occurrence_id
                and other.sequence_event_type == target.sequence_event_type
            ):
                candidates.append((
                    max(lower, other.clip_end - duration),
                    min(upper, other.clip_start),
                ))
        intervals = _merge_intervals(candidates)
        if not intervals:
            empty_reason = "no_fully_contained_same_type_distractor"
        else:
            for other in game_half_trims:
                if (
                    other.sample_id != target.sample_id
                    and other.event_type == target.event_type
                ):
                    intervals = _exclude_partial_trim_overlap(
                        intervals, other, duration
                    )
            if not intervals:
                empty_reason = (
                    "partial_same_type_trim_at_5min_context_boundary"
                )
    elif context_type == "context_10min":
        for other in game_half_trims:
            if (
                other.sample_id != target.sample_id
                and other.event_type == target.event_type
            ):
                intervals = _exclude_partial_trim_overlap(
                    intervals, other, duration
                )
        if not intervals:
            empty_reason = "partial_same_type_trim_at_10min_context_boundary"

    start = choose_interval(
        intervals, stable_seed(global_seed, target.sample_id, context_type)
    )
    if start is None:
        reason = empty_reason or (
            "no_one_minute_start_satisfies_trim_and_sequence_exclusions"
        )
        return WindowResult(False, reason, checks={
            "base_feasible": True,
            "context_constraint_feasible": False,
        })
    return _result_for_start(
        target, game_half_trims, sequence, context_type, start, duration,
        half_duration, policy_version
    )
