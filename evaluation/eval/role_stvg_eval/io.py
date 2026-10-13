"""Input discovery and append-only retry history handling."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .version import SUPPORTED_SETTINGS


@dataclass
class PredictionFile:
    model: str
    setting: str
    path: Path
    rows: list[dict[str, Any]]
    histories: dict[str, list[dict[str, Any]]]

    @property
    def latest(self) -> dict[str, dict[str, Any]]:
        return {sample_id: history[-1] for sample_id, history in self.histories.items()}

    @property
    def duplicate_ids(self) -> set[str]:
        return {sample_id for sample_id, history in self.histories.items() if len(history) > 1}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def setting_filename(setting: str) -> str:
    if setting not in SUPPORTED_SETTINGS:
        raise ValueError(f"unsupported evaluator setting: {setting!r}")
    return f"{setting}.jsonl"


def ground_truth_path(gt_root: Path, setting: str) -> Path:
    return gt_root / "benchmark" / setting_filename(setting)


def load_ground_truth(gt_root: Path, setting: str) -> list[dict[str, Any]]:
    path = ground_truth_path(gt_root, setting)
    rows = read_jsonl(path)
    ids = [row.get("metadata", {}).get("sample_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate GT sample IDs in {path}")
    if setting == "oracle":
        for row in rows:
            sample_id = row["metadata"]["sample_id"]
            annotation_path = gt_root / "spatial" / "annotations" / f"{sample_id}_annotation.json"
            row["_spatial_annotation"] = json.loads(annotation_path.read_text(encoding="utf-8"))
    return rows


def discover_predictions(outputs_root: Path, setting: str) -> list[PredictionFile]:
    filename = setting_filename(setting)
    discovered = []
    for model_directory in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        candidates = []
        exact = model_directory / filename
        if exact.is_file():
            candidates.append(exact)
        candidates.extend(
            path for path in sorted(model_directory.glob(f"*_{filename}"))
            if path not in candidates
        )
        if len(candidates) > 1:
            rendered = ", ".join(str(path) for path in candidates)
            raise ValueError(
                f"multiple prediction files for {model_directory.name} / {setting}: {rendered}"
            )
        if not candidates:
            continue
        path = candidates[0]
        rows = read_jsonl(path)
        histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line_number, row in enumerate(rows, start=1):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str):
                raise ValueError(f"{path}:{line_number}: missing string sample_id")
            histories[sample_id].append(row)
        discovered.append(PredictionFile(path.parent.name, setting, path, rows, dict(histories)))
    if not discovered:
        raise FileNotFoundError(f"no {filename} predictions under {outputs_root}")
    return discovered
