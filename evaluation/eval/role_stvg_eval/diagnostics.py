"""Prediction diagnostics independent from score computation."""

from __future__ import annotations

from typing import Any, Iterable

from .boxes import valid_normalized_box
from .temporal import finite_number, nominal_index


def rows_for_setting(prediction: object, setting: str) -> object:
    if setting == "oracle":
        return prediction
    return prediction.get("role_localization") if isinstance(prediction, dict) else None


def inspect_rows(rows: object, required_roles: Iterable[str]) -> dict[str, Any]:
    required = set(required_roles)
    observed: set[str] = set()
    invalid_boxes = 0
    invalid_timestamps = 0
    indices: list[int] = []
    if not isinstance(rows, list):
        return {
            "missing_roles": sorted(required), "extra_roles": [],
            "invalid_boxes": 0, "invalid_timestamps": 0,
        }
    for row in rows:
        if not isinstance(row, dict):
            invalid_timestamps += 1
            continue
        observed.update(key for key in row if key != "time_sec")
        time = row.get("time_sec")
        index = nominal_index(time)
        if index is None:
            invalid_timestamps += 1
        else:
            indices.append(index)
        for key, value in row.items():
            if key == "time_sec" or value is None:
                continue
            if not valid_normalized_box(value):
                invalid_boxes += 1
    invalid_timestamps += len(indices) - len(set(indices))
    if len(indices) > 1:
        invalid_timestamps += sum(
            current != previous + 1 for previous, current in zip(indices, indices[1:])
        )
    return {
        "missing_roles": sorted(required - observed),
        "extra_roles": sorted(observed - required),
        "invalid_boxes": invalid_boxes,
        "invalid_timestamps": invalid_timestamps,
    }

