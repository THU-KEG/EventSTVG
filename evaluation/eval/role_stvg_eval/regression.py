"""Dataset-wide invariants for the frozen downstream evaluation policy."""

from __future__ import annotations

from typing import Any

from .metrics import explicit_gt_support, nullable_classic_role_score
from .validation import canonicalize_rows
from .version import EVALUATOR_ID


def policy_identity_regression(
    gt_records: list[dict[str, Any]],
    setting: str,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Check GT identity and the visible-GT/null-prediction invariant."""
    gt_self_failures: list[dict[str, str]] = []
    null_everywhere_failures: list[dict[str, str]] = []
    checked_roles = visible_roles = 0

    for record in gt_records:
        metadata = record["metadata"]
        sample_id = metadata["sample_id"]
        roles = tuple(metadata["active_roles"])
        rows = (
            record["answer"] if setting == "oracle"
            else record["answer"]["role_localization"]
        )
        times, support = explicit_gt_support(rows)
        gt_view = canonicalize_rows(rows, times, support, roles)
        null_rows = [
            {"time_sec": timestamp, **{role: None for role in roles}}
            for timestamp in times
        ]
        null_view = canonicalize_rows(null_rows, times, support, roles)

        for role in roles:
            checked_roles += 1
            self_score, _ = nullable_classic_role_score(
                gt_view, gt_view, role, support, support,
            )
            if abs(self_score - 1.0) > tolerance:
                gt_self_failures.append({"sample_id": sample_id, "role": role})

            has_visible_gt = any(
                gt_view.state(timestamp, role).kind == "box"
                for timestamp in support
            )
            if has_visible_gt:
                visible_roles += 1
                null_score, _ = nullable_classic_role_score(
                    gt_view, null_view, role, support, support,
                )
                if abs(null_score) > tolerance:
                    null_everywhere_failures.append({
                        "sample_id": sample_id, "role": role,
                    })

    return {
        "policy": EVALUATOR_ID,
        "setting": setting,
        "tolerance": tolerance,
        "checked_sample_roles": checked_roles,
        "visible_gt_sample_roles": visible_roles,
        "gt_self_failures": gt_self_failures,
        "predict_null_everywhere_failures": null_everywhere_failures,
        "passed": not gt_self_failures and not null_everywhere_failures,
    }
