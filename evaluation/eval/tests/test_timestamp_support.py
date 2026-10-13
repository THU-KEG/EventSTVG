import pytest

from role_stvg_eval.metrics import context_scores, explicit_gt_support
from role_stvg_eval.validation import canonicalize_rows
from role_stvg_eval.metrics import oracle_role_scores


BOX = [0, 0, 10, 10]


def test_oracle_support_is_exactly_answer_timestamps():
    gt_rows = [
        {"time_sec": 0.0, "A": BOX},
        {"time_sec": 0.5, "A": BOX},
    ]
    pred_rows = [
        {"time_sec": 0.0, "A": BOX},
        {"time_sec": 0.25, "A": BOX},
    ]
    times, support = explicit_gt_support(gt_rows)
    gt = canonicalize_rows(gt_rows, times, support, ["A"])
    pred = canonicalize_rows(pred_rows, times, support, ["A"])
    assert oracle_role_scores(gt, pred, ["A"])["A"] == 0.5
    assert pred.diagnostics["ignored_extra_timestamps"] == 1


def test_context_gt_support_does_not_gain_phantom_terminal():
    gt_rows = [
        {"time_sec": 0.0, "A": BOX},
        {"time_sec": 0.5, "A": BOX},
        {"time_sec": 1.0, "A": BOX},
    ]
    # The GT end lies after 1.5, but explicit GT localization stops at 1.0.
    # Predicted support is still generated from the predicted interval.
    score = context_scores((0, 1.508), (0, 1.508), gt_rows, gt_rows, ["A"])
    assert score.temporal_overlap_frames == 3
    assert score.temporal_union_frames == 4
    assert score.grid_tiou == 0.75
    assert score.viou == pytest.approx(0.75)
