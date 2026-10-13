from role_stvg_eval.parsing import predicted_interval


def test_explicit_interval_is_preferred():
    pred = {
        "temporal_window": {"start_sec": 2, "end_sec": 3},
        "role_localization": [{"time_sec": 10}, {"time_sec": 20}],
    }
    assert predicted_interval(pred) == ((2.0, 3.0), "explicit")


def test_invalid_explicit_interval_does_not_fallback():
    pred = {
        "temporal_window": {"start_sec": 3, "end_sec": 2},
        "role_localization": [{"time_sec": 10}, {"time_sec": 20}],
    }
    assert predicted_interval(pred) == (None, "explicit_invalid")


def test_timestamp_fallback_is_deterministic():
    pred = {"role_localization": [{"time_sec": 2}, {"time_sec": 3.5}]}
    assert predicted_interval(pred) == ((2.0, 3.5), "timestamps_fallback")

