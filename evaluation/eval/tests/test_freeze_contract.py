import json
from pathlib import Path

from role_stvg_eval import EVALUATOR_ID, EVALUATOR_VERSION, PARSER_ID
from role_stvg_eval.cli import parse_args
from role_stvg_eval.io import PredictionFile, discover_predictions, setting_filename
from role_stvg_eval.parsing import frozen_parser_identifier
from role_stvg_eval.provenance import build_provenance, sha256_file


def test_frozen_identifiers_match_version_file():
    root = Path(__file__).resolve().parents[1]
    assert (root / "VERSION").read_text(encoding="utf-8").strip() == EVALUATOR_ID
    assert EVALUATOR_ID == "role_stvg_eval_v1.0"
    assert EVALUATOR_VERSION == "1.0.0"
    assert PARSER_ID == frozen_parser_identifier() == "role_stvg_json_extraction_v1"


def test_all_context_durations_use_the_same_canonical_setting_interface():
    for duration in ("1min", "3min", "5min", "10min"):
        setting = f"context_{duration}"
        assert setting_filename(setting) == f"{setting}.jsonl"
        args = parse_args([
            "--outputs", "outputs", "--gt", "gt", "--setting", setting,
            "--out", "artifacts",
        ])
        assert args.setting == setting


def test_discovery_accepts_one_authoritative_suffixed_file_for_a_model(tmp_path):
    model_dir = tmp_path / "Model"
    model_dir.mkdir()
    path = model_dir / "provider_context_3min.jsonl"
    path.write_text(json.dumps({"sample_id": "sample"}) + "\n", encoding="utf-8")
    sources = discover_predictions(tmp_path, "context_3min")
    assert len(sources) == 1
    assert sources[0].model == "Model"
    assert sources[0].path == path


def test_provenance_records_frozen_versions_and_source_hashes(tmp_path):
    gt_root = tmp_path / "gt"
    benchmark = gt_root / "benchmark"
    benchmark.mkdir(parents=True)
    gt_record = {
        "metadata": {
            "sample_id": "sample", "prompt_spec_version": "prompt-v1",
            "prompt_spec_sha256": "prompt-hash",
        },
        "answer": {},
    }
    gt_path = benchmark / "context_1min.jsonl"
    gt_path.write_text(json.dumps(gt_record) + "\n", encoding="utf-8")
    prediction_path = tmp_path / "prediction.jsonl"
    response = {
        "sample_id": "sample", "model": "model-id", "setting": "context_1min",
        "manifest_sha256": "manifest-hash", "raw_response": "{}", "status": "ok",
    }
    prediction_path.write_text(json.dumps(response) + "\n", encoding="utf-8")
    source = PredictionFile(
        "Model", "context_1min", prediction_path, [response], {"sample": [response]},
    )
    provenance = build_provenance(gt_root, "context_1min", [gt_record], [source])
    assert provenance["evaluator_id"] == EVALUATOR_ID
    assert provenance["parser_id"] == PARSER_ID
    assert provenance["ground_truth_sha256"] == sha256_file(gt_path)
    assert provenance["prediction_sources"][0]["source_sha256"] == sha256_file(prediction_path)
    assert provenance["prediction_sources"][0]["manifest_hashes"] == ["manifest-hash"]
