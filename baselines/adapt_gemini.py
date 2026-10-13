"""Explicit historical Gemini yxyx-to-xyxy serialization adaptation, offline."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/eval"))
from role_stvg_eval.parsing import reparse_record
from role_stvg_eval.version import SUPPORTED_SETTINGS


def swap_role_boxes(prediction):
    value = json.loads(json.dumps(prediction))
    rows = value if isinstance(value, list) else value.get("role_localization") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        return value
    for row in rows:
        if not isinstance(row, dict):
            continue
        for role, box in tuple(row.items()):
            if role != "time_sec" and isinstance(box, list) and len(box) == 4:
                row[role] = [box[1], box[0], box[3], box[2]]
    return value


def adapt(row, setting):
    result = reparse_record(row, setting)
    output = dict(row)
    output["native_raw_response"] = row.get("raw_response")
    output["coordinate_adapter"] = "declared_gemini_yxyx_to_xyxy"
    if result.parse_ok:
        payload = json.dumps(swap_role_boxes(result.prediction), ensure_ascii=False, separators=(",", ":"))
        if result.extraction_status == "whole_response_fence":
            payload = f"```json\n{payload}\n```"
        elif result.extraction_status == "unique_embedded_json":
            payload = f"Adapted coordinate serialization:\n{payload}"
        output["raw_response"] = payload
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--setting", choices=SUPPORTED_SETTINGS, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.out.resolve():
        parser.error("Use a distinct adapted output path; retain the original responses")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf8") as source, args.out.open("x", encoding="utf8") as target:
        for line in source:
            if line.strip():
                target.write(json.dumps(adapt(json.loads(line), args.setting), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
