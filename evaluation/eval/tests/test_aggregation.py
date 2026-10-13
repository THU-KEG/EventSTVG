import json
from pathlib import Path

from role_stvg_eval.aggregate import evaluate_model
from role_stvg_eval.io import PredictionFile


def gt(sample_id):
    return {
        "metadata": {"sample_id": sample_id, "event_type": "test", "active_roles": ["A"]},
        "answer": [{"time_sec": 0.0, "A": [0, 0, 10, 10]}],
        "_spatial_annotation": {
            "video_width": 1000, "video_height": 1000,
            "annotations": {"A": [{"time_sec": 0.0, "bbox_xyxy": [0, 0, 10, 10]}]},
        },
    }


def successful(sample_id):
    prediction = [{"time_sec": 0.0, "A": [0, 0, 10, 10]}]
    return {
        "sample_id": sample_id, "status": "ok", "parse_ok": True,
        "prediction": prediction, "raw_response": json.dumps(prediction),
    }


def failed(sample_id):
    return {
        "sample_id": sample_id, "status": "error", "parse_ok": False,
        "prediction": None, "raw_response": "",
    }


def test_failures_remain_in_denominator_and_last_retry_is_authoritative():
    rows = [successful("one"), successful("two"), failed("two")]
    source = PredictionFile(
        model="model", setting="oracle", path=Path("formal.jsonl"), rows=rows,
        histories={"one": [rows[0]], "two": [rows[1], rows[2]]},
    )
    scores, samples, _, diagnostics = evaluate_model([gt("one"), gt("two")], source)
    assert scores["N"] == 2
    assert scores["m_vIoU"] == 0.5
    assert [row["vIoU"] for row in samples] == [1.0, 0.0]
    assert diagnostics["N_duplicate_sample_ids"] == 1
    assert diagnostics["N_duplicate_records"] == 1


def test_equal_role_then_sample_then_dataset_weighting():
    gt_records = [
        {
            "metadata": {
                "sample_id": "two_roles", "event_type": "test",
                "active_roles": ["A", "B"],
            },
            "answer": [{
                "time_sec": 0.0,
                "A": [0, 0, 10, 10],
                "B": [20, 20, 30, 30],
            }],
        },
        gt("one_role"),
    ]
    first_prediction = [{"time_sec": 0.0, "A": [0, 0, 10, 10]}]
    rows = [
        {
            "sample_id": "two_roles", "status": "ok",
            "raw_response": json.dumps(first_prediction),
        },
        successful("one_role"),
    ]
    source = PredictionFile(
        model="model", setting="oracle", path=Path("formal.jsonl"), rows=rows,
        histories={"two_roles": [rows[0]], "one_role": [rows[1]]},
    )
    scores, samples, _, _ = evaluate_model(gt_records, source)
    assert [row["vIoU"] for row in samples] == [0.5, 1.0]
    assert scores["m_vIoU"] == 0.75


def test_http_error_response_is_returned_but_scores_zero():
    error = failed("error")
    error.update(error_message="Error code: 400 - DataInspectionFailed",
                 transport_diagnostics=[{"http_status": 400}],
                 raw_response=json.dumps([{ "time_sec": 0.0, "A": [0, 0, 10, 10]}]))
    timeout = failed("timeout")
    timeout.update(transport_diagnostics=[{"http_status": 503}, {"http_status": None}])
    rows = [successful("ok"), error, timeout]
    source = PredictionFile(model="model", setting="oracle", path=Path("formal.jsonl"),
                            rows=rows, histories={r["sample_id"]: [r] for r in rows})
    scores, samples, _, diagnostics = evaluate_model(
        [gt("ok"), gt("error"), gt("timeout"), gt("missing")], source)
    assert scores["N"] == 4
    assert scores["m_vIoU"] == 0.25
    assert [r["returned"] for r in samples] == [True, True, False, False]
    assert samples[1]["parse_status"] == "api_error_response"
    assert diagnostics["N_returned"] == 2
    assert diagnostics["N_request_fail"] == 2
    assert diagnostics["N_error_responses"] == 1


def test_legacy_http_error_and_explicit_response_availability():
    from role_stvg_eval.parsing import returned
    assert returned({"status": "error", "error_message": "Error code: 400 - DataInspectionFailed"})
    assert returned({"status": "error", "response_received": True})
    assert not returned({"status": "error", "response_received": False, "http_status": 400})
    assert not returned({"status": "error", "error_type": "TimeoutError"})
