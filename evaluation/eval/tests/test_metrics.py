import pytest

from role_stvg_eval.metrics import context_scores, oracle_role_scores
from role_stvg_eval.validation import canonicalize_rows


BOX = [0, 0, 10, 10]


def rows(times, role="A", box=BOX):
    return [{"time_sec": time, role: box} for time in times]


def test_perfect_prediction():
    gt = rows([0.0, 0.5, 1.0])
    score = context_scores((0, 1), (0, 1), gt, gt, ["A"])
    assert score.tiou == score.grid_tiou == score.viou == score.viou_int == 1.0


def test_temporal_over_prediction_with_null_rows_is_penalized():
    gt_times = [index * 0.5 for index in range(3)]
    pred = rows(gt_times) + [{"time_sec": 1.5, "A": None}]
    score = context_scores((0, 1), (0, 1.5), rows(gt_times), pred, ["A"])
    assert score.tiou == pytest.approx(2 / 3)
    assert score.grid_tiou == 0.75
    assert score.viou == 0.75
    assert score.viou_int == 1.0


def test_missing_predicted_timestamp_is_in_error_support():
    gt = rows([0.0, 0.5, 1.0])
    pred = rows([0.0, 0.5])
    score = context_scores((0, 1), (0, 1), gt, pred, ["A"])
    assert score.viou == score.viou_int == pytest.approx(2 / 3)


def test_malformed_row_invalidates_only_that_row():
    gt = rows([0.0, 0.5])
    pred = [{"time_sec": 0.0, "A": BOX}, "malformed-row"]
    score = context_scores((0, 0.5), (0, 0.5), gt, pred, ["A"])
    assert score.tiou == 1.0
    assert score.viou == score.viou_int == 0.5


def test_partial_temporal_overlap():
    gt = rows([0.0, 0.5, 1.0])
    pred = rows([0.5, 1.0, 1.5])
    score = context_scores((0, 1), (0.5, 1.5), gt, pred, ["A"])
    assert score.grid_tiou == 0.5
    assert score.viou == 0.5
    assert score.viou_int == 1.0


def test_temporal_under_prediction_is_penalized():
    gt = rows([0.0, 0.5, 1.0])
    pred = rows([0.5, 1.0])
    score = context_scores((0, 1), (0.5, 1), gt, pred, ["A"])
    assert score.viou == pytest.approx(2 / 3)
    assert score.viou_int == 1.0


def test_spatial_only_error():
    gt = rows([0.0, 0.5], box=[0, 0, 7, 1])
    pred = rows([0.0, 0.5], box=[3, 0, 10, 1])
    score = context_scores((0, 0.5), (0, 0.5), gt, pred, ["A"])
    assert score.tiou == 1.0
    assert score.viou == score.viou_int == pytest.approx(0.4)


def test_no_temporal_overlap():
    score = context_scores(
        (0, 1), (2, 3), rows([0, 0.5, 1]), rows([2, 2.5, 3]), ["A"],
    )
    assert score.tiou == score.viou == score.viou_int == 0.0


def test_null_does_not_create_free_credit_and_gt_self_is_one():
    gt = [
        {"time_sec": 0.0, "A": BOX},
        {"time_sec": 0.5, "A": None},
        {"time_sec": 1.0, "A": BOX},
    ]
    self_score = context_scores((0, 1), (0, 1), gt, gt, ["A"])
    null_score = context_scores(
        (0, 1), (0, 1), gt,
        [{"time_sec": t, "A": None} for t in (0, 0.5, 1)], ["A"],
    )
    assert self_score.viou == self_score.viou_int == 1.0
    assert null_score.viou == null_score.viou_int == 0.0


def test_gt_null_pred_null_does_not_enlarge_joint_denominator():
    gt = [
        {"time_sec": 0.0, "A": BOX},
        {"time_sec": 0.5, "A": None},
    ]
    score = context_scores((0, 0.5), (0, 0.5), gt, gt, ["A"])
    assert score.viou == score.viou_int == 1.0


def test_gt_null_pred_box_is_penalized():
    gt = [{"time_sec": 0.0, "A": None}]
    pred = [{"time_sec": 0.0, "A": BOX}]
    score = context_scores((0, 0), (0, 0), gt, pred, ["A"])
    assert score.viou == score.viou_int == 0.0


def test_gt_box_pred_null_is_penalized():
    gt = [{"time_sec": 0.0, "A": BOX}]
    pred = [{"time_sec": 0.0, "A": None}]
    score = context_scores((0, 0), (0, 0), gt, pred, ["A"])
    assert score.viou == score.viou_int == 0.0


@pytest.mark.parametrize("bad_value", ["missing", "invalid"])
def test_missing_and_invalid_prediction_are_penalized(bad_value):
    gt = rows([0.0, 0.5])
    second = {"time_sec": 0.5}
    if bad_value == "invalid":
        second["A"] = [10, 10, 5, 5]
    pred = [{"time_sec": 0.0, "A": BOX}, second]
    score = context_scores((0, 0.5), (0, 0.5), gt, pred, ["A"])
    assert score.viou == score.viou_int == 0.5


def test_all_invisible_role_requires_explicit_null_everywhere():
    gt = [{"time_sec": t, "A": None} for t in (0, 0.5)]
    explicit = context_scores((0, 0.5), (0, 0.5), gt, gt, ["A"])
    missing = context_scores((0, 0.5), (0, 0.5), gt, [], ["A"])
    boxed = context_scores((0, 0.5), (0, 0.5), gt, rows([0, 0.5]), ["A"])
    assert explicit.viou == explicit.viou_int == 1.0
    assert missing.viou == missing.viou_int == 0.0
    assert boxed.viou == boxed.viou_int == 0.0


def test_classic_stvg_behavior_when_no_null_states_occur():
    gt = rows([0.0, 0.5, 1.0])
    pred = rows([0.5, 1.0, 1.5])
    score = context_scores((0, 1), (0.5, 1.5), gt, pred, ["A"])
    # Two unit-IoU box pairs over four timestamps in the support union.
    assert score.viou == 0.5


def test_oracle_uses_nullable_classic_null_denominator():
    gt_rows = [
        {"time_sec": 0.0, "A": BOX},
        {"time_sec": 0.5, "A": None},
    ]
    support = [0, 1]
    gt = canonicalize_rows(gt_rows, [0.0, 0.5], support, ["A"])
    self_scores = oracle_role_scores(gt, gt, ["A"])
    missing = canonicalize_rows(
        [{"time_sec": 0.0, "A": BOX}], [0.0, 0.5], support, ["A"],
    )
    missing_scores = oracle_role_scores(gt, missing, ["A"])
    assert self_scores["A"] == 1.0
    assert missing_scores["A"] == 0.5
