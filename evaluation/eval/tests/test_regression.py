from role_stvg_eval.regression import policy_identity_regression


def test_policy_identity_and_null_everywhere_regression():
    records = [{
        "metadata": {
            "sample_id": "sample", "active_roles": ["visible", "invisible"],
        },
        "answer": [
            {"time_sec": 0.0, "visible": [0, 0, 10, 10], "invisible": None},
            {"time_sec": 0.5, "visible": None, "invisible": None},
        ],
    }]
    report = policy_identity_regression(records, "oracle")
    assert report["passed"]
    assert report["checked_sample_roles"] == 2
    assert report["visible_gt_sample_roles"] == 1
