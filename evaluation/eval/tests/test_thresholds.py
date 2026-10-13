from role_stvg_eval.aggregate import summarize


def test_thresholds_are_strict():
    scores = summarize([0.1, 0.3, 0.5], "metric")
    assert scores["metric_gt_0.1"] == 2 / 3
    assert scores["metric_gt_0.3"] == 1 / 3
    assert scores["metric_gt_0.5"] == 0.0

