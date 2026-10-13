"""Component-level validation and canonical Role-STVG scoring views."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from .boxes import as_float_box
from .temporal import TOLERANCE, finite_number, generated_timestamps, nominal_index


RoleStateKind = Literal["box", "null", "missing", "invalid"]


@dataclass(frozen=True)
class RoleState:
    kind: RoleStateKind
    box: tuple[float, float, float, float] | None = None


MISSING = RoleState("missing")
INVALID = RoleState("invalid")
NULL = RoleState("null")


@dataclass(frozen=True)
class CanonicalRows:
    support: tuple[int, ...]
    times: tuple[float, ...]
    states: dict[int, dict[str, RoleState]]
    container_state: Literal["valid", "empty", "missing", "invalid"]
    diagnostics: dict[str, int]

    def state(self, timestamp: int, role: str) -> RoleState:
        return self.states.get(timestamp, {}).get(role, MISSING)


@dataclass(frozen=True)
class ValidatedWindow:
    value: tuple[float, float] | None
    state: Literal[
        "valid", "missing", "not_object", "missing_endpoint",
        "invalid_endpoint", "reversed", "outside_video",
    ]
    extra_fields: tuple[str, ...] = ()


def validate_context_window(
    prediction: object, input_duration: object,
) -> ValidatedWindow:
    if not isinstance(prediction, dict) or "temporal_window" not in prediction:
        return ValidatedWindow(None, "missing")
    value = prediction.get("temporal_window")
    if not isinstance(value, dict):
        return ValidatedWindow(None, "not_object")
    extras = tuple(sorted(set(value) - {"start_sec", "end_sec"}))
    if "start_sec" not in value or "end_sec" not in value:
        return ValidatedWindow(None, "missing_endpoint", extras)
    start, end = value.get("start_sec"), value.get("end_sec")
    if not finite_number(start) or not finite_number(end):
        return ValidatedWindow(None, "invalid_endpoint", extras)
    start, end = float(start), float(end)
    if start > end:
        return ValidatedWindow(None, "reversed", extras)
    if not finite_number(input_duration) or start < 0.0 or end > float(input_duration):
        return ValidatedWindow(None, "outside_video", extras)
    return ValidatedWindow((start, end), "valid", extras)


def context_support(window: tuple[float, float] | None) -> tuple[tuple[float, ...], tuple[int, ...]]:
    times = generated_timestamps(window)
    indices = tuple(nominal_index(time) for time in times)
    if any(index is None for index in indices):
        return (), ()
    return times, tuple(index for index in indices if index is not None)


def _role_state(row: dict, role: str) -> RoleState:
    if role not in row:
        return MISSING
    value = row[role]
    if value is None:
        return NULL
    box = as_float_box(value)
    return RoleState("box", tuple(box)) if box is not None else INVALID


def canonicalize_rows(
    rows: object,
    expected_times: Iterable[float],
    support: Iterable[int],
    required_roles: Iterable[str],
) -> CanonicalRows:
    """Build a repair-free view on one exact timestamp support."""
    times = tuple(float(value) for value in expected_times)
    keys = tuple(int(value) for value in support)
    roles = tuple(required_roles)
    if len(times) != len(keys) or len(set(keys)) != len(keys):
        raise ValueError("canonical support must have one unique key per timestamp")

    diagnostics = defaultdict(int)
    states = {key: {role: MISSING for role in roles} for key in keys}
    if rows is None:
        return CanonicalRows(keys, times, states, "missing", dict(diagnostics))
    if not isinstance(rows, list):
        diagnostics["invalid_localization_container"] += 1
        return CanonicalRows(keys, times, states, "invalid", dict(diagnostics))
    if not rows:
        diagnostics["empty_localization"] += 1
        return CanonicalRows(keys, times, states, "empty", dict(diagnostics))

    matches: dict[int, list[dict]] = defaultdict(list)
    finite_sequence: list[float] = []
    allowed_fields = {"time_sec", *roles}
    for row in rows:
        if not isinstance(row, dict):
            diagnostics["invalid_rows"] += 1
            continue
        extras = set(row) - allowed_fields
        diagnostics["extra_fields"] += len(extras)
        if "time_sec" not in row:
            diagnostics["missing_timestamps"] += 1
            continue
        timestamp = row.get("time_sec")
        if not finite_number(timestamp):
            diagnostics["invalid_timestamps"] += 1
            continue
        timestamp = float(timestamp)
        finite_sequence.append(timestamp)
        candidates = [
            index for index, expected in enumerate(times)
            if abs(timestamp - expected) <= TOLERANCE
        ]
        if len(candidates) != 1:
            diagnostics["ignored_extra_timestamps"] += 1
            continue
        matches[candidates[0]].append(row)

    diagnostics["out_of_order_rows"] += sum(
        current < previous
        for previous, current in zip(finite_sequence, finite_sequence[1:])
    )
    for offset, key in enumerate(keys):
        candidates = matches.get(offset, [])
        if not candidates:
            diagnostics["missing_expected_timestamps"] += 1
            continue
        if len(candidates) > 1:
            diagnostics["duplicate_timestamps"] += 1
            states[key] = {role: INVALID for role in roles}
            continue
        row = candidates[0]
        for role in roles:
            value = _role_state(row, role)
            states[key][role] = value
            diagnostics[f"role_{value.kind}"] += 1

    return CanonicalRows(keys, times, states, "valid", dict(diagnostics))
