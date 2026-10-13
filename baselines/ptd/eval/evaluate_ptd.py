#!/usr/bin/env python3
"""Snapshot, adapt, and score completed formal PTD role records offline."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from adapter import (
    ADAPTER_ID, adapt_role_record, atomic_write_json, atomic_write_jsonl,
    merge_role_rows, sha256_file, union_interval,
)


DEFAULT_PTD_ROOT = Path("outputs/PTD")
DEFAULT_GT_ROOT = Path("evaluation_gt")
DEFAULT_EVALUATOR_PARENT = Path(__file__).resolve().parents[3] / "evaluation" / "eval"
AUDIT_RECORD_ID = "a230facd9724350c81b0e719031b412e"


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptd-root", type=Path, default=DEFAULT_PTD_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--evaluator-parent", type=Path, default=DEFAULT_EVALUATOR_PARENT)
    parser.add_argument("--setting", choices=("oracle", "context_1min", "all"), default="all")
    parser.add_argument("--source-root", type=Path, help="Optional record root (used for dry-run validation)")
    parser.add_argument("--output-root", type=Path, help="Defaults to PTD/eval/results")
    parser.add_argument("--adapt-only", action="store_true")
    return parser


def load_gt(io_module: Any, root: Path, setting: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = io_module.load_ground_truth(root, setting)
    return rows, {row["metadata"]["sample_id"]: row for row in rows}


def completed_record_paths(source: Path, setting: str) -> list[Path]:
    root = source / setting / "records" if (source / setting / "records").is_dir() else source / "records"
    if not root.is_dir():
        return []
    return sorted(root.glob("*.json"))


def load_stable_json(path: Path) -> dict[str, Any]:
    # os.replace commits formal records atomically; one read is therefore a
    # complete immutable snapshot. Ignore any unrelated temporary files.
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def make_prediction_row(sample_id: str, setting: str, prediction: object | None) -> dict[str, Any]:
    if prediction is None:
        return {
            "sample_id": sample_id, "status": "adapter_failure",
            "raw_response": None, "parse_ok": False, "prediction": None,
            "adapter_id": ADAPTER_ID,
        }
    raw = json.dumps(prediction, ensure_ascii=False, separators=(",", ":"))
    return {
        "sample_id": sample_id, "status": "ok", "raw_response": raw,
        "parse_ok": True, "prediction": prediction, "adapter_id": ADAPTER_ID,
    }


def write_required_audit(ptd_root: Path, gt_root: Path, evaluator_parent: Path) -> Path | None:
    """Write the fixed Corner/<t6> audit, including the grouped evaluator input."""
    source_path = ptd_root / "oracle" / "records" / f"{AUDIT_RECORD_ID}.json"
    if not source_path.is_file():
        return None
    if str(evaluator_parent) not in sys.path:
        sys.path.insert(0, str(evaluator_parent))
    evaluator_io = importlib.import_module("role_stvg_eval.io")
    source = load_stable_json(source_path)
    sample_id = source["sample_id"]
    gt_rows, gt_by_id = load_gt(evaluator_io, gt_root, "oracle")
    del gt_rows
    gt = gt_by_id[sample_id]
    times = [float(row["time_sec"]) for row in gt["answer"]]
    siblings = []
    for path in completed_record_paths(ptd_root, "oracle"):
        record = load_stable_json(path)
        if record.get("sample_id") == sample_id:
            siblings.append(adapt_role_record(record, path))
    grouped_rows = merge_role_rows(siblings, times)
    target = next(item for item in siblings if item.get("role_key") == "corner_conceder")
    token = target["time_token_map"]["<t6>"]
    bbox_entry = next(
        item for item in target["native_tube_timestamped"]
        if item.get("time_token") == "<t6>"
    )
    audit = {
        "adapter_id": ADAPTER_ID,
        "purpose": "Demonstrate that PTD <t6> is sampled position 6, not six seconds or source frame 6.",
        "sample_id": sample_id,
        "event_type": source["event_type"],
        "role_key": source["role_key"],
        "source_record_path": str(source_path.resolve()),
        "source_record_sha256": sha256_file(source_path),
        "native_prediction": source.get("raw_native_output"),
        "mapping": {
            "token": "<t6>",
            "sampled_position": token["sampled_position"],
            "zero_based_array_offset": token["sampled_position"] - 1,
            "source_frame_index": token["source_frame_index"],
            "timestamp_sec": token["timestamp_sec"],
            "temporal_cell_support_sec": token["cell_support_sec"],
            "native_bbox": bbox_entry["native_bbox"],
            "canonical_bbox": bbox_entry["canonical_bbox"],
            "nearest_evaluator_timestamp_sec": min(
                times, key=lambda value: (abs(value - token["timestamp_sec"]), value)
            ),
        },
        "native_time_interval_tokens": target["native_time_interval_tokens"],
        "predicted_interval_sec": target["predicted_interval_sec"],
        "role_specific_adapted_tube": target["adapted_role_stvg_tube"],
        "final_grouped_oracle_prediction_consumed_by_evaluator": grouped_rows,
        "oracle_evaluation_grid_sec": times,
        "note": "The singleton bbox is projected once to nearest evaluator timestamp 0.5 s; the PTD interval remains [0.30,0.36].",
    }
    destination = ptd_root / "eval" / "audit_examples" / "Corner_H1_568917_w1_corner_conceder_t6.json"
    atomic_write_json(destination, audit)
    return destination


def evaluate_setting(
    setting: str, source: Path, output: Path, gt_root: Path, evaluator_parent: Path,
    adapt_only: bool,
) -> dict[str, Any]:
    if str(evaluator_parent) not in sys.path:
        sys.path.insert(0, str(evaluator_parent))
    evaluator_io = importlib.import_module("role_stvg_eval.io")
    temporal = importlib.import_module("role_stvg_eval.temporal")
    aggregate = importlib.import_module("role_stvg_eval.aggregate")
    version = importlib.import_module("role_stvg_eval.version")

    gt, gt_by_id = load_gt(evaluator_io, gt_root, setting)
    source_paths = completed_record_paths(source, setting)
    source_hashes_before = {str(path.resolve()): sha256_file(path) for path in source_paths}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    adapted_dir = output / "adapted" / setting / "records"
    errors = []
    for path in source_paths:
        try:
            record = load_stable_json(path)
            if record.get("setting") != setting:
                continue
            adapted = adapt_role_record(record, path)
            record_id = record.get("record_id") or path.stem
            destination = adapted_dir / f"{record_id}.json"
            # Resume-safe cache: immutable source hash + adapter version.
            if destination.is_file():
                cached = load_stable_json(destination)
                if cached.get("adapter_id") == ADAPTER_ID and cached.get("source_record_sha256") == adapted["source_record_sha256"]:
                    adapted = cached
            grouped[str(record.get("sample_id"))].append(adapted)
        except Exception as exc:  # preserve a snapshot-level diagnostic, never mutate source
            errors.append({"source_record_path": str(path.resolve()), "error": f"{type(exc).__name__}: {exc}"})

    prediction_rows = []
    completed_roles = 0
    for sample_id, roles in sorted(grouped.items()):
        gt_record = gt_by_id.get(sample_id)
        if gt_record is None:
            errors.append({"sample_id": sample_id, "error": "sample_id_not_in_frozen_gt"})
            continue
        completed_roles += len(roles)
        if setting == "oracle":
            evaluation_times = [float(row["time_sec"]) for row in gt_record["answer"]]
            prediction = merge_role_rows(roles, evaluation_times)
        else:
            combined = union_interval(roles)
            if combined is None:
                prediction = None
                for role in roles:
                    role["adapted_role_stvg_tube"] = []
            else:
                evaluation_times = temporal.generated_timestamps(combined)
                prediction = {
                    "temporal_window": {"start_sec": combined[0], "end_sec": combined[1]},
                    "role_localization": merge_role_rows(roles, evaluation_times),
                }
        prediction_rows.append(make_prediction_row(sample_id, setting, prediction))
        for role in roles:
            record_id = Path(role["source_record_path"]).stem
            atomic_write_json(adapted_dir / f"{record_id}.json", role)

    input_dir = output / "evaluator_inputs" / "PTD-64f"
    input_path = input_dir / f"{setting}.jsonl"
    atomic_write_jsonl(input_path, prediction_rows)
    source_hashes_after = {path: sha256_file(Path(path)) for path in source_hashes_before}
    unchanged = source_hashes_before == source_hashes_after
    if not unchanged:
        raise RuntimeError("formal inference record changed during offline snapshot")

    summary: dict[str, Any] = {
        "adapter_id": ADAPTER_ID,
        "setting": setting,
        "source_root": str(source.resolve()),
        "source_record_count": len(source_paths),
        "adapted_role_count": completed_roles,
        "adapted_sample_count": len(prediction_rows),
        "frozen_gt_sample_count": len(gt),
        "formal_source_records_unchanged": unchanged,
        "evaluator_input": str(input_path.resolve()),
        "errors": errors,
    }
    if not adapt_only:
        prediction_file = evaluator_io.PredictionFile(
            model="PTD-64f", setting=setting, path=input_path,
            rows=prediction_rows,
            histories={row["sample_id"]: [row] for row in prediction_rows},
        )
        scores, samples, role_scores, diagnostics = aggregate.evaluate_model(gt, prediction_file)
        metric_dir = output / "metrics" / setting
        atomic_write_json(metric_dir / "scores.json", scores)
        atomic_write_json(metric_dir / "per_sample.json", samples)
        atomic_write_json(metric_dir / "per_role.json", role_scores)
        atomic_write_json(metric_dir / "diagnostics.json", diagnostics)
        summary.update({
            "evaluator_id": version.EVALUATOR_ID,
            "parser_id": version.PARSER_ID,
            "metrics_dir": str(metric_dir.resolve()),
            "scores": scores,
        })
    atomic_write_json(output / f"snapshot_{setting}.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = args_parser().parse_args(argv)
    output = (args.output_root or args.ptd_root / "eval" / "results").resolve()
    source = (args.source_root or args.ptd_root).resolve()
    settings = ("oracle", "context_1min") if args.setting == "all" else (args.setting,)
    summaries = [
        evaluate_setting(
            setting, source, output, args.gt_root.resolve(),
            args.evaluator_parent.resolve(), args.adapt_only,
        )
        for setting in settings
    ]
    manifest = {"adapter_id": ADAPTER_ID, "snapshots": summaries}
    audit_path = write_required_audit(
        args.ptd_root.resolve(), args.gt_root.resolve(), args.evaluator_parent.resolve()
    )
    manifest["required_audit"] = str(audit_path.resolve()) if audit_path else None
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
