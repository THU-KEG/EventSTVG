"""Reproducible source and protocol provenance for evaluation artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from .io import PredictionFile, ground_truth_path
from .parsing import frozen_parser_identifier, frozen_parser_path
from .version import EVALUATOR_ID, EVALUATOR_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values(records: Iterable[dict[str, Any]], key: str) -> list[str]:
    found: set[str] = set()
    for record in records:
        value = record.get(key)
        if value is None and isinstance(record.get("prompt_manifest"), dict):
            value = record["prompt_manifest"].get(key)
        if value is not None:
            found.add(str(value))
    return sorted(found)


def _metadata_values(records: Iterable[dict[str, Any]], key: str) -> list[str]:
    return sorted({
        str(record["metadata"][key])
        for record in records
        if isinstance(record.get("metadata"), dict) and record["metadata"].get(key) is not None
    })


def evaluator_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    sources = sorted(package.glob("*.py"))
    policy = package.parent / "EVAL_POLICY.md"
    version = package.parent / "VERSION"
    return {
        str(path.resolve()): sha256_file(path)
        for path in [*sources, policy, version]
    }


def build_provenance(
    gt_root: Path,
    setting: str,
    gt_records: list[dict[str, Any]],
    prediction_sources: list[PredictionFile],
) -> dict[str, Any]:
    gt_path = ground_truth_path(gt_root, setting).resolve()
    parser_path = frozen_parser_path().resolve()
    gt_prompt_versions = _metadata_values(gt_records, "prompt_spec_version")
    gt_prompt_hashes = _metadata_values(gt_records, "prompt_spec_sha256")
    sources = []
    for source in prediction_sources:
        authoritative = list(source.latest.values())
        response_prompt_versions = _values(authoritative, "prompt_spec_version")
        response_prompt_hashes = _values(authoritative, "prompt_spec_sha256")
        sources.append({
            "model": source.model,
            "setting": setting,
            "source_file": str(source.path.resolve()),
            "source_sha256": sha256_file(source.path.resolve()),
            "records": len(source.rows),
            "unique_sample_ids": len(source.histories),
            "duplicate_records": len(source.rows) - len(source.histories),
            "authority_policy": "last record per sample_id is authoritative",
            "record_model_ids": _values(authoritative, "model"),
            "record_settings": _values(authoritative, "setting"),
            "response_prompt_versions": response_prompt_versions,
            "response_prompt_hashes": response_prompt_hashes,
            "effective_prompt_versions": response_prompt_versions or gt_prompt_versions,
            "effective_prompt_hashes": response_prompt_hashes or gt_prompt_hashes,
            "manifest_hashes": _values(authoritative, "manifest_sha256"),
            "manifest_paths": _values(authoritative, "manifest_path"),
        })
    return {
        "evaluator_id": EVALUATOR_ID,
        "evaluator_version": EVALUATOR_VERSION,
        "parser_id": frozen_parser_identifier(),
        "parser_source": str(parser_path),
        "parser_source_sha256": sha256_file(parser_path),
        "setting": setting,
        "ground_truth_file": str(gt_path),
        "ground_truth_sha256": sha256_file(gt_path),
        "ground_truth_records": len(gt_records),
        "prompt_versions": gt_prompt_versions,
        "prompt_hashes": gt_prompt_hashes,
        "ground_truth_source_manifests": _metadata_values(gt_records, "source_manifest"),
        "prediction_sources": sources,
        "evaluator_source_hashes": evaluator_source_hashes(),
    }
