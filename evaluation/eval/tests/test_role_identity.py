import pytest

from role_stvg_eval.metrics import context_scores


def test_wrong_semantic_roles_are_never_permuted():
    gt = [{"time_sec": 0.0, "A": [0, 0, 10, 10], "B": [20, 20, 30, 30]}]
    pred = [{"time_sec": 0.0, "A": [20, 20, 30, 30], "B": [0, 0, 10, 10]}]
    score = context_scores((0, 0), (0, 0), gt, pred, ["A", "B"])
    assert score.viou == score.viou_int == 0.0


def test_missing_role_is_zero_and_role_average_is_equal_weighted():
    gt = [{"time_sec": 0.0, "A": [0, 0, 10, 10], "B": [20, 20, 30, 30]}]
    pred = [{"time_sec": 0.0, "A": [0, 0, 10, 10]}]
    score = context_scores((0, 0), (0, 0), gt, pred, ["A", "B"])
    assert score.viou == score.viou_int == 0.5


def test_extra_role_does_not_improve_score():
    gt = [{"time_sec": 0.0, "A": [0, 0, 10, 10]}]
    pred = [{"time_sec": 0.0, "A": [20, 20, 30, 30], "extra": [0, 0, 10, 10]}]
    score = context_scores((0, 0), (0, 0), gt, pred, ["A"])
    assert score.viou == 0.0

