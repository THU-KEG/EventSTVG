from role_stvg_eval.parsing import _frozen_parser_module, reparse_record


def test_reparse_record_is_exactly_the_frozen_extractor():
    raw = 'prose [{"time_sec": 0, "A": null}] trailing text'
    record = {
        "status": "ok",
        "raw_response": raw,
        "parse_ok": False,
        "prediction": None,
    }
    direct = _frozen_parser_module().extract_prediction(raw, "oracle")
    refreshed = reparse_record(record, "oracle")
    assert refreshed.prediction == direct.prediction
    assert refreshed.parse_ok == direct.parse_ok
    assert refreshed.extraction_status == direct.status
    assert refreshed.candidate_count == direct.candidate_count


def test_persisted_prediction_is_not_used_by_reparse():
    record = {
        "status": "ok",
        "raw_response": "not JSON",
        "parse_ok": True,
        "prediction": [{"time_sec": 0, "A": [0, 0, 10, 10]}],
    }
    refreshed = reparse_record(record, "oracle")
    assert not refreshed.parse_ok
    assert refreshed.prediction is None
