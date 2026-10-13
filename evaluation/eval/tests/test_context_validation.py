import json
import math
from pathlib import Path

import pytest

from role_stvg_eval.aggregate import evaluate_model
from role_stvg_eval.io import PredictionFile
from role_stvg_eval.validation import validate_context_window


BOX = [0, 0, 10, 10]


def gt_record():
    return {
        "metadata": {
            "sample_id": "sample",
            "event_type": "test",
            "active_roles": ["A"],
            "window_duration": 10.0,
        },
        "answer": {
            "temporal_window": {"start_sec": 1.0, "end_sec": 2.0},
            "role_localization": [
                {"time_sec": 1.0, "A": BOX},
                {"time_sec": 1.5, "A": BOX},
                {"time_sec": 2.0, "A": BOX},
            ],
        },
    }


def source(prediction):
    row = {
        "sample_id": "sample",
        "status": "ok",
        "raw_response": json.dumps(prediction),
        # Deliberately stale persisted fields: formal scoring must ignore them.
        "parse_ok": False,
        "prediction": None,
    }
    return PredictionFile(
        "model", "context_1min", Path("formal.jsonl"), [row],
        {"sample": [row]},
    )


def test_valid_temporal_prediction_scores_when_localization_is_empty():
    prediction = {
        "temporal_window": {"start_sec": 1.0, "end_sec": 2.0},
        "role_localization": [],
    }
    _, samples, _, _ = evaluate_model([gt_record()], source(prediction))
    assert samples[0]["parsed"]
    assert samples[0]["tIoU"] == 1.0
    assert samples[0]["vIoU"] == samples[0]["vIoU_Int"] == 0.0


def test_extra_top_level_field_is_diagnostic_only():
    prediction = {
        "temporal_window": {"start_sec": 1.0, "end_sec": 2.0},
        "role_localization": [
            {"time_sec": 1.0, "A": BOX},
            {"time_sec": 1.5, "A": BOX},
            {"time_sec": 2.0, "A": BOX},
        ],
    }
    extra_prediction = {**prediction, "unexpected": "diagnostic-only"}

    scores, _, _, _ = evaluate_model([gt_record()], source(prediction))
    extra_scores, samples, _, diagnostics = evaluate_model(
        [gt_record()], source(extra_prediction)
    )

    assert extra_scores == scores
    assert json.loads(samples[0]["validation_diagnostics"])[
        "extra_top_level_fields"
    ] == 1
    assert diagnostics["component_diagnostics"]["extra_top_level_fields"] == 1


def test_invalid_temporal_window_zeros_temporal_and_joint_scores():
    prediction = {
        "temporal_window": {"start_sec": 2.0, "end_sec": 1.0},
        "role_localization": [{"time_sec": 1.0, "A": BOX}],
    }
    _, samples, _, _ = evaluate_model([gt_record()], source(prediction))
    assert samples[0]["tIoU"] == 0.0
    assert samples[0]["vIoU"] == samples[0]["vIoU_Int"] == 0.0


@pytest.mark.parametrize(
    ("window", "state"),
    [
        ({"start_sec": -0.1, "end_sec": 1.0}, "outside_video"),
        ({"start_sec": 1.0, "end_sec": 10.1}, "outside_video"),
        ({"start_sec": True, "end_sec": 1.0}, "invalid_endpoint"),
        ({"start_sec": math.nan, "end_sec": 1.0}, "invalid_endpoint"),
        ({"start_sec": 1.0, "end_sec": math.inf}, "invalid_endpoint"),
    ],
)
def test_temporal_window_requires_finite_numeric_in_video_endpoints(window, state):
    result = validate_context_window({"temporal_window": window}, 10.0)
    assert result.value is None
    assert result.state == state
