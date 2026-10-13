"""Score one prediction JSONL against integrated metadata using frozen metrics."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))
from role_stvg_eval.aggregate import evaluate_model
from role_stvg_eval.io import PredictionFile, load_ground_truth, read_jsonl
from role_stvg_eval.version import EVALUATOR_ID, PARSER_ID, SUPPORTED_SETTINGS


def run(metadata: Path, predictions: Path, setting: str, model: str) -> dict:
    rows = read_jsonl(metadata)
    if not rows:
        raise ValueError("GT metadata is empty")
    with tempfile.TemporaryDirectory(prefix="eventstvg-gt-") as directory:
        root = Path(directory)
        target = root / "benchmark" / f"{setting}.jsonl"
        target.parent.mkdir()
        seen = set()
        with target.open("w", encoding="utf8") as output:
            for row in rows:
                sid = row["sample_id"]
                if sid != row["metadata"]["sample_id"] or sid in seen:
                    raise ValueError(f"Inconsistent or duplicate GT ID: {sid}")
                if row["metadata"]["setting"] != setting:
                    raise ValueError(f"GT setting differs from requested setting: {sid}")
                # IDs become Oracle paths: reject traversal before writing.
                if Path(sid).is_absolute() or ".." in Path(sid).parts:
                    raise ValueError(f"Unsafe sample ID: {sid}")
                seen.add(sid)
                output.write(json.dumps({k: row[k] for k in ("metadata", "prompt", "answer")}) + "\n")
                if setting == "oracle":
                    spatial = root / "spatial" / "annotations" / f"{sid}_annotation.json"
                    spatial.parent.mkdir(parents=True, exist_ok=True)
                    spatial.write_text(json.dumps(row["spatial_annotation"]), encoding="utf8")
        prediction_rows = read_jsonl(predictions)
        histories = {}
        for row in prediction_rows:
            if not isinstance(row.get("sample_id"), str):
                raise ValueError("Every prediction record must contain a string sample_id")
            histories.setdefault(row["sample_id"], []).append(row)
        source = PredictionFile(model, setting, predictions, prediction_rows, histories)
        scores, samples, roles, diagnostics = evaluate_model(load_ground_truth(root, setting), source)
        return {
            "evaluator_id": EVALUATOR_ID, "parser_id": PARSER_ID,
            "setting": setting, "scores": scores, "per_sample": samples,
            "per_role": roles, "diagnostics": diagnostics,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--setting", choices=SUPPORTED_SETTINGS, required=True)
    parser.add_argument("--model", default="prediction")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.metadata, args.predictions, args.setting, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
