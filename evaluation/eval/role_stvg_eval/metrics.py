"""Pure Role-STVG metrics over validated, canonical scoring views."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from .boxes import box_iou
from .temporal import finite_number, nominal_index, temporal_iou
from .validation import CanonicalRows, canonicalize_rows, context_support


def visibility_aware_role_score(
    gt: CanonicalRows,
    pred: CanonicalRows,
    role: str,
    domain: Iterable[int],
    *,
    gt_visible_anywhere: bool | None = None,
) -> tuple[float, float]:
    """Return (role score, IoU numerator) under ``role_stvg_eval_v1.0``."""
    timestamps = tuple(dict.fromkeys(domain))
    if not timestamps:
        return 0.0, 0.0

    if gt_visible_anywhere is None:
        gt_visible_anywhere = any(
            gt.state(timestamp, role).kind == "box" for timestamp in gt.support
        )
    if not gt_visible_anywhere:
        all_explicit_null = all(
            pred.state(timestamp, role).kind == "null" for timestamp in timestamps
        )
        return (1.0 if all_explicit_null else 0.0), 0.0

    gt_boxes = {
        timestamp for timestamp in timestamps
        if gt.state(timestamp, role).kind == "box"
    }
    pred_boxes = {
        timestamp for timestamp in timestamps
        if pred.state(timestamp, role).kind == "box"
    }
    pred_errors = {
        timestamp for timestamp in timestamps
        if pred.state(timestamp, role).kind in {"missing", "invalid"}
    }
    numerator = sum(
        box_iou(
            list(gt.state(timestamp, role).box or ()),
            list(pred.state(timestamp, role).box or ()),
        )
        for timestamp in gt_boxes & pred_boxes
    )
    denominator = len(gt_boxes | pred_boxes | pred_errors)
    return (numerator / denominator if denominator else 0.0), numerator


def nullable_classic_role_score(
    gt: CanonicalRows,
    pred: CanonicalRows,
    role: str,
    gt_support: Iterable[int],
    pred_support: Iterable[int],
) -> tuple[float, float]:
    """Return nullable-classic joint vIoU and its spatial numerator."""
    gt_timestamps = set(gt_support)
    pred_timestamps = set(pred_support)
    intersection = gt_timestamps & pred_timestamps
    union = gt_timestamps | pred_timestamps
    numerator = sum(
        box_iou(
            list(gt.state(timestamp, role).box or ()),
            list(pred.state(timestamp, role).box or ()),
        )
        for timestamp in intersection
        if (
            gt.state(timestamp, role).kind == "box"
            and pred.state(timestamp, role).kind == "box"
        )
    )
    shared_nulls = sum(
        gt.state(timestamp, role).kind == "null"
        and pred.state(timestamp, role).kind == "null"
        for timestamp in intersection
    )
    denominator = len(union) - shared_nulls
    return (numerator / denominator if denominator else 1.0), numerator


def oracle_role_scores(
    gt: CanonicalRows,
    pred: CanonicalRows,
    required_roles: Iterable[str],
) -> dict[str, float]:
    domain = gt.support
    return {
        role: nullable_classic_role_score(
            gt, pred, role, domain, domain,
        )[0]
        for role in required_roles
    }


@dataclass(frozen=True)
class ContextRoleScore:
    role: str
    viou: float
    viou_int: float
    spatial_iou_sum: float


@dataclass(frozen=True)
class ContextScore:
    tiou: float
    grid_tiou: float
    viou: float
    viou_int: float
    temporal_overlap_frames: int
    temporal_union_frames: int
    roles: tuple[ContextRoleScore, ...]


def canonical_context_scores(
    gt_window: tuple[float, float],
    pred_window: tuple[float, float] | None,
    gt: CanonicalRows,
    pred: CanonicalRows,
    required_roles: Iterable[str],
    *,
    localization_scorable: bool = True,
) -> ContextScore:
    """Score context predictions after temporal and row validation."""
    roles = tuple(required_roles)
    gt_support = set(gt.support)
    pred_support = set(pred.support) if pred_window is not None else set()
    intersection = gt_support & pred_support
    union = gt_support | pred_support
    grid_tiou = len(intersection) / len(union) if union else 0.0

    role_results = []
    for role in roles:
        gt_visible = any(
            gt.state(timestamp, role).kind == "box" for timestamp in gt.support
        )
        if localization_scorable:
            viou, numerator = nullable_classic_role_score(
                gt, pred, role, gt_support, pred_support,
            )
            viou_int, _ = visibility_aware_role_score(
                gt, pred, role, intersection, gt_visible_anywhere=gt_visible,
            )
        else:
            viou = viou_int = numerator = 0.0
        role_results.append(ContextRoleScore(
            role=role,
            viou=viou,
            viou_int=viou_int,
            spatial_iou_sum=numerator,
        ))

    return ContextScore(
        tiou=temporal_iou(pred_window, gt_window),
        grid_tiou=grid_tiou,
        viou=mean(item.viou for item in role_results) if role_results else 0.0,
        viou_int=(
            mean(item.viou_int for item in role_results)
            if role_results and intersection else 0.0
        ),
        temporal_overlap_frames=len(intersection),
        temporal_union_frames=len(union),
        roles=tuple(role_results),
    )


def explicit_gt_support(rows: object) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Read exact GT timestamps without regenerating an interval terminal."""
    if not isinstance(rows, list):
        raise ValueError("GT role_localization must be a list")
    times: list[float] = []
    support: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or not finite_number(row.get("time_sec")):
            raise ValueError("GT localization row has invalid time_sec")
        timestamp = float(row["time_sec"])
        index = nominal_index(timestamp)
        if index is None:
            raise ValueError("GT localization row has invalid time_sec")
        times.append(timestamp)
        support.append(index)
    if len(support) != len(set(support)):
        raise ValueError("GT localization support is not unique")
    return tuple(times), tuple(support)


def context_scores(
    gt_window: tuple[float, float],
    pred_window: tuple[float, float] | None,
    gt_rows: object,
    pred_rows: object,
    required_roles: Iterable[str],
) -> ContextScore:
    """Convenience wrapper for validated Role-STVG context scoring."""
    roles = tuple(required_roles)
    gt_times, gt_support = explicit_gt_support(gt_rows)
    pred_times, pred_support = context_support(pred_window)
    gt = canonicalize_rows(gt_rows, gt_times, gt_support, roles)
    pred = canonicalize_rows(pred_rows, pred_times, pred_support, roles)
    return canonical_context_scores(
        gt_window, pred_window, gt, pred, roles,
        localization_scorable=pred.container_state == "valid",
    )
