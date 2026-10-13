import pytest

from role_stvg_eval.temporal import interval_support, nominal_index, temporal_iou


def test_continuous_tiou():
    assert temporal_iou((0, 10), (0, 10)) == 1.0
    assert temporal_iou((20, 30), (0, 10)) == 0.0
    assert temporal_iou((0, 20), (0, 10)) == 0.5


def test_robust_nominal_grid_canonicalization():
    assert nominal_index(1.499999999) == 3
    assert nominal_index(1.500000001) == 3
    assert nominal_index(45.25) == 91
    assert nominal_index(45.75) == 92
    assert nominal_index(46.25) == 93
    assert interval_support((1.022168, 2.022168)) == {2, 3, 4}
