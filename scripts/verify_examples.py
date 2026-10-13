"""Offline checks of frozen scoring, retry handling and production formatting."""
import importlib.util
import json
import math
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eventstvg_evaluate", ROOT / "evaluation/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
from prompt_format import build_answer, build_prompt

SETTINGS = ("oracle", "context_1min", "context_3min", "context_5min", "context_10min")


def verify():
    results = {}
    task_spec = json.loads((ROOT / "examples/task_spec.json").read_text())
    shared_ids = None
    for setting in SETTINGS:
        metadata = ROOT / "examples" / f"{setting}.metadata.jsonl"
        rows = [json.loads(line) for line in metadata.read_text().splitlines()]
        ids = [row["sample_id"] for row in rows]
        if shared_ids is None:
            shared_ids = ids
        assert ids == shared_ids and len(ids) == 3
        for row in rows:
            meta = row["metadata"]
            temporal = {"start_in_window": meta["target_start_in_window"],
                        "end_in_window": meta["target_end_in_window"]}
            answer = build_answer(setting, temporal, row["spatial_annotation"], row["active_roles"])
            assert answer == row["answer"], (setting, row["sample_id"], "answer changed")
            prompt = build_prompt(setting, row["event_type"], row["active_roles"], temporal,
                                  meta, task_spec,
                                  [r["time_sec"] for r in row["answer"]] if setting == "oracle" else None)
            assert prompt == row["prompt"], (setting, row["sample_id"], "prompt changed")
        runs = {}
        for model in ("copy_gt", "missing", "invalid", "duplicate_retry"):
            runs[model] = evaluation.run(metadata, ROOT / "examples/predictions" / model / f"{setting}.jsonl", setting, model)
        copy = runs["copy_gt"]["scores"]
        assert copy["N"] == 3
        assert math.isclose(copy["m_vIoU"], 1.0 if setting == "oracle" else 29/30, abs_tol=1e-12)
        if setting != "oracle":
            assert copy["m_tIoU"] == copy["m_vIoU_Int"] == 1.0
        for metric in ("m_vIoU", "m_tIoU", "m_vIoU_Int"):
            if metric in copy:
                assert runs["duplicate_retry"]["scores"][metric] == copy[metric]
                assert runs["invalid"]["scores"][metric] == 0
        missing = {r["sample_id"]: r for r in runs["missing"]["per_sample"]}
        assert runs["missing"]["scores"]["N"] == 3
        assert missing[ids[-1]]["vIoU"] == 0
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary) / "empty.jsonl"
            empty.write_text("")
            empty_result = evaluation.run(metadata, empty, setting, "empty")["scores"]
            assert empty_result["N"] == 3 and empty_result["m_vIoU"] == 0
        results[setting] = {name: value["scores"] for name, value in runs.items()}
    return {"passed": True, "settings": results, "prompt_answer_pairs_verified": 15,
            "empty_prediction_files_checked": 5, "new_visual_review": False}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
