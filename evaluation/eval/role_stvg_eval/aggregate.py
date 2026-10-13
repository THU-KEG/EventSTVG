"""Dataset evaluation and equal-weight sample aggregation."""

from __future__ import annotations

import json
from collections import Counter
from statistics import mean
from typing import Any

from .diagnostics import inspect_rows, rows_for_setting
from .io import PredictionFile
from .metrics import (
    canonical_context_scores, explicit_gt_support, oracle_role_scores,
)
from .parsing import reparse_record, returned
from .temporal import interval
from .validation import (
    canonicalize_rows, context_support, validate_context_window,
)
from .version import EVALUATOR_ID, PARSER_ID


THRESHOLDS = (0.0, 0.1, 0.3, 0.5)


def _record_provenance_value(record: dict[str, Any] | None, key: str) -> object:
    if not isinstance(record, dict):
        return None
    if record.get(key) is not None:
        return record[key]
    prompt_manifest = record.get("prompt_manifest")
    return prompt_manifest.get(key) if isinstance(prompt_manifest, dict) else None


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"m_{prefix}": mean(values) if values else 0.0,
        **{
            f"{prefix}_gt_{threshold:g}": (
                sum(value > threshold for value in values) / len(values) if values else 0.0
            )
            for threshold in THRESHOLDS
        },
    }


def evaluate_model(
    gt_records: list[dict[str, Any]], predictions: PredictionFile
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    setting = predictions.setting
    latest = predictions.latest
    expected_ids = {record["metadata"]["sample_id"] for record in gt_records}
    per_sample: list[dict[str, Any]] = []
    per_role: list[dict[str, Any]] = []
    missing_role_samples = extra_role_samples = invalid_boxes = invalid_timestamps = 0
    component_diagnostics: Counter[str] = Counter()
    extraction_statuses: Counter[str] = Counter()

    for gt_record in gt_records:
        metadata, answer = gt_record["metadata"], gt_record["answer"]
        sample_id = metadata["sample_id"]
        required_roles = tuple(metadata["active_roles"])
        result = latest.get(sample_id)
        reparsed = reparse_record(result, setting)
        extraction_statuses[reparsed.extraction_status] += 1
        pred = reparsed.prediction if reparsed.parse_ok else None
        pred_rows = rows_for_setting(pred, setting)
        row_diagnostics = inspect_rows(pred_rows, required_roles)
        missing_role_samples += bool(row_diagnostics["missing_roles"])
        extra_role_samples += bool(row_diagnostics["extra_roles"])

        base = {
            "evaluator_id": EVALUATOR_ID,
            "parser_id": PARSER_ID,
            "sample_id": sample_id,
            "model": predictions.model,
            "setting": setting,
            "event_type": metadata["event_type"],
            "returned": returned(result),
            "request_fail": not returned(result),
            "parsed": reparsed.parse_ok,
            "parse_status": reparsed.extraction_status,
            "parse_candidate_count": reparsed.candidate_count,
            "gt_start": None,
            "gt_end": None,
            "pred_start": None,
            "pred_end": None,
            "pred_interval_source": "not_applicable" if setting == "oracle" else "missing",
            "tIoU": None if setting == "oracle" else 0.0,
            "grid_tIoU": None if setting == "oracle" else 0.0,
            "vIoU": 0.0,
            "vIoU_Int": None if setting == "oracle" else 0.0,
            "num_gt_roles": len(required_roles),
            "num_missing_roles": len(row_diagnostics["missing_roles"]),
            "num_extra_roles": len(row_diagnostics["extra_roles"]),
            "missing_roles": json.dumps(row_diagnostics["missing_roles"]),
            "extra_roles": json.dumps(row_diagnostics["extra_roles"]),
            "invalid_boxes": row_diagnostics["invalid_boxes"],
            "invalid_timestamps": row_diagnostics["invalid_timestamps"],
            "temporal_overlap_frames": None if setting == "oracle" else 0,
            "temporal_union_frames": None if setting == "oracle" else 0,
            "attempt_records": len(predictions.histories.get(sample_id, [])),
            "prompt_spec_version": (
                _record_provenance_value(result, "prompt_spec_version")
                or metadata.get("prompt_spec_version")
            ),
            "prompt_spec_sha256": (
                _record_provenance_value(result, "prompt_spec_sha256")
                or metadata.get("prompt_spec_sha256")
            ),
            "manifest_sha256": _record_provenance_value(result, "manifest_sha256"),
            "source_prediction_file": str(predictions.path.resolve()),
            "validation_diagnostics": "{}",
        }

        if setting == "oracle":
            gt_times, support = explicit_gt_support(answer)
            gt_view = canonicalize_rows(
                answer, gt_times, support, required_roles,
            )
            pred_view = canonicalize_rows(
                pred_rows, gt_times, support, required_roles,
            )
            component_diagnostics.update(pred_view.diagnostics)
            base["validation_diagnostics"] = json.dumps(
                pred_view.diagnostics, sort_keys=True,
            )
            scores = oracle_role_scores(gt_view, pred_view, required_roles)
            base["vIoU"] = mean(scores.values()) if scores else 0.0
            for role in required_roles:
                per_role.append({
                    "evaluator_id": EVALUATOR_ID,
                    "parser_id": PARSER_ID,
                    "sample_id": sample_id,
                    "model": predictions.model,
                    "setting": setting,
                    "event_type": metadata["event_type"],
                    "role": role,
                    "role_vIoU": scores[role],
                    "role_vIoU_Int": None,
                })
        else:
            extra_top_level_fields = (
                set(pred) - {"temporal_window", "role_localization"}
                if isinstance(pred, dict) else set()
            )
            gt_window = interval(answer.get("temporal_window"))
            if gt_window is None:
                raise ValueError(f"invalid GT interval: {sample_id}")
            gt_rows = answer.get("role_localization")
            gt_times, gt_support = explicit_gt_support(gt_rows)
            gt_view = canonicalize_rows(
                gt_rows, gt_times, gt_support, required_roles,
            )

            validated_window = validate_context_window(
                pred, metadata.get("window_duration"),
            )
            pred_window = validated_window.value
            pred_times, pred_support = context_support(pred_window)
            pred_view = canonicalize_rows(
                pred_rows, pred_times, pred_support, required_roles,
            )
            sample_diagnostics = Counter(pred_view.diagnostics)
            if extra_top_level_fields:
                sample_diagnostics["extra_top_level_fields"] += len(
                    extra_top_level_fields
                )
            sample_diagnostics[f"temporal_window_{validated_window.state}"] += 1
            sample_diagnostics["extra_temporal_fields"] += len(
                validated_window.extra_fields
            )
            component_diagnostics.update(sample_diagnostics)
            base["validation_diagnostics"] = json.dumps(
                dict(sample_diagnostics), sort_keys=True,
            )

            localization_scorable = (
                pred_window is not None and pred_view.container_state == "valid"
            )
            score = canonical_context_scores(
                gt_window,
                pred_window,
                gt_view,
                pred_view,
                required_roles,
                localization_scorable=localization_scorable,
            )
            base.update({
                "gt_start": gt_window[0],
                "gt_end": gt_window[1],
                "pred_start": pred_window[0] if pred_window else None,
                "pred_end": pred_window[1] if pred_window else None,
                "pred_interval_source": (
                    "explicit" if validated_window.state == "valid"
                    else f"explicit_{validated_window.state}"
                ),
                "tIoU": score.tiou,
                "grid_tIoU": score.grid_tiou,
                "vIoU": score.viou,
                "vIoU_Int": score.viou_int,
                "temporal_overlap_frames": score.temporal_overlap_frames,
                "temporal_union_frames": score.temporal_union_frames,
            })
            for role_score in score.roles:
                per_role.append({
                    "evaluator_id": EVALUATOR_ID,
                    "parser_id": PARSER_ID,
                    "sample_id": sample_id,
                    "model": predictions.model,
                    "setting": setting,
                    "event_type": metadata["event_type"],
                    "role": role_score.role,
                    "role_vIoU": role_score.viou,
                    "role_vIoU_Int": role_score.viou_int,
                    "spatial_iou_sum": role_score.spatial_iou_sum,
                })

        invalid_boxes += int(base["invalid_boxes"])
        invalid_timestamps += int(base["invalid_timestamps"])
        per_sample.append(base)

    viou_values = [float(row["vIoU"]) for row in per_sample]
    scores = {
        "evaluator_id": EVALUATOR_ID,
        "parser_id": PARSER_ID,
        "model": predictions.model,
        "setting": setting,
        "N": len(per_sample),
    }
    if setting == "oracle":
        scores.update(summarize(viou_values, "vIoU"))
    else:
        tiou_values = [float(row["tIoU"]) for row in per_sample]
        viou_int_values = [float(row["vIoU_Int"]) for row in per_sample]
        scores.update(summarize(tiou_values, "tIoU"))
        scores.update(summarize(viou_values, "vIoU"))
        scores.update(summarize(viou_int_values, "vIoU_Int"))
        overlap = [row for row in per_sample if float(row["tIoU"]) > 0.0]
        conditional = [float(row["vIoU_Int"]) for row in overlap]
        scores["N_temporal_overlap"] = len(overlap)
        scores["m_vIoU_Int_given_tIoU_gt_0"] = mean(conditional) if conditional else 0.0
        for threshold in (0.1, 0.3, 0.5):
            scores[f"vIoU_Int_gt_{threshold:g}_given_tIoU_gt_0"] = (
                sum(value > threshold for value in conditional) / len(conditional)
                if conditional else 0.0
            )

    diagnostics = {
        "evaluator_id": EVALUATOR_ID,
        "parser_id": PARSER_ID,
        "model": predictions.model,
        "setting": setting,
        "N_expected": len(gt_records),
        "N_output_records_found": len(predictions.rows),
        "N_unique_output_sample_ids": len(predictions.histories),
        "N_returned": sum(bool(row["returned"]) for row in per_sample),
        "N_request_fail": sum(bool(row["request_fail"]) for row in per_sample),
        "N_error_responses": sum(row["returned"] and latest[row["sample_id"]].get("status") == "error" for row in per_sample),
        "N_parsed": sum(bool(row["parsed"]) for row in per_sample),
        "N_missing": sum(sample_id not in latest for sample_id in expected_ids),
        "N_duplicate_sample_ids": len(predictions.duplicate_ids),
        "N_duplicate_records": len(predictions.rows) - len(predictions.histories),
        "N_missing_required_roles": missing_role_samples,
        "N_extra_roles": extra_role_samples,
        "N_invalid_boxes": invalid_boxes,
        "N_invalid_timestamps": invalid_timestamps,
        "N_temporal_overlap_gt_0": (
            sum(float(row["tIoU"]) > 0.0 for row in per_sample)
            if setting != "oracle" else None
        ),
        "extraction_statuses": dict(sorted(extraction_statuses.items())),
        "component_diagnostics": dict(sorted(component_diagnostics.items())),
        "unexpected_output_sample_ids": sorted(set(latest) - expected_ids),
        "duplicate_policy": "last record per sample_id is authoritative",
    }
    return scores, per_sample, per_role, diagnostics
