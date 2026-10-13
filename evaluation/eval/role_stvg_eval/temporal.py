"""Continuous and canonical-grid temporal operations."""

from __future__ import annotations

import math
from typing import Iterable


GRID_HZ = 2
GRID_STEP_SECONDS = 0.5
TOLERANCE = 1e-6


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def interval(value: object) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    start, end = value.get("start_sec"), value.get("end_sec")
    if not finite_number(start) or not finite_number(end):
        return None
    start, end = float(start), float(end)
    return (start, end) if end >= start else None


def temporal_iou(
    predicted: tuple[float, float] | None,
    ground_truth: tuple[float, float],
) -> float:
    if predicted is None:
        return 0.0
    p0, p1 = predicted
    g0, g1 = ground_truth
    intersection = max(0.0, min(p1, g1) - max(p0, g0))
    union = max(p1, g1) - min(p0, g0)
    return intersection / union if union > 0.0 else 0.0


def nominal_index(time_sec: object) -> int | None:
    """Nearest nominal 2 Hz index, with exact half ties rounded upward.

    Python's built-in round uses ties-to-even, which can map two adjacent
    0.5-second timestamps to the same index when a window starts at an exact
    quarter second.
    """
    if not finite_number(time_sec):
        return None
    return math.floor(float(time_sec) * GRID_HZ + 0.5)


def interval_support(window: tuple[float, float] | None) -> set[int]:
    """Return the 0.5-second support anchored at the interval start.

    Benchmark GT starts can be non-half-second values. The first point maps to
    its nearest nominal 2 Hz index and subsequent points advance by one index.
    """
    if window is None:
        return set()
    start, end = window
    if end < start:
        return set()
    count = math.floor((end - start + TOLERANCE) / GRID_STEP_SECONDS) + 1
    first = nominal_index(start)
    return {first + offset for offset in range(count)} if first is not None else set()


def generated_timestamps(
    window: tuple[float, float] | None,
) -> tuple[float, ...]:
    """Generate the start-anchored 0.5-second sequence for a valid window."""
    if window is None:
        return ()
    start, end = window
    if end < start:
        return ()
    count = math.floor((end - start + TOLERANCE) / GRID_STEP_SECONDS) + 1
    return tuple(start + offset * GRID_STEP_SECONDS for offset in range(count))


def timestamps_interval(values: Iterable[object]) -> tuple[float, float] | None:
    times = [float(value) for value in values if finite_number(value)]
    return (min(times), max(times)) if times else None
