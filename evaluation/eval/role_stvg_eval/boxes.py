"""Bounding-box validation and IoU using the frozen Oracle convention."""

from __future__ import annotations

import math
from typing import Any


def valid_box(value: object) -> bool:
    """Frozen downstream validity for normalized Role-STVG boxes."""
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and math.isfinite(float(x))
            and 0.0 <= float(x) <= 1000.0
            for x in value
        )
        and value[2] > value[0]
        and value[3] > value[1]
    )


def valid_normalized_box(value: object) -> bool:
    """Compatibility alias for the single strict scoring convention."""
    return valid_box(value)


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def as_float_box(value: Any) -> list[float] | None:
    return [float(x) for x in value] if valid_box(value) else None
