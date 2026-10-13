"""Prediction-state and interval extraction from persisted formal outputs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import re
from pathlib import Path
import sys
from typing import Any

from .temporal import interval, timestamps_interval
from .version import PARSER_ID


@dataclass(frozen=True)
class ReparsedPrediction:
    prediction: object | None
    parse_ok: bool
    extraction_status: str
    candidate_count: int


@lru_cache(maxsize=1)
def _frozen_parser_module():
    path = frozen_parser_path()
    name = "role_stvg_frozen_parser_policy"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if getattr(module, "PARSER_POLICY", None) != PARSER_ID:
        raise RuntimeError(
            f"frozen parser identifier mismatch: {getattr(module, 'PARSER_POLICY', None)!r}"
        )
    return module


def frozen_parser_path() -> Path:
    return Path(__file__).resolve().parents[2] / "bench" / "parser_policy.py"


def frozen_parser_identifier() -> str:
    return str(_frozen_parser_module().PARSER_POLICY)


def reparse_record(
    record: dict[str, Any] | None, setting: str,
) -> ReparsedPrediction:
    """Re-extract the authoritative raw response with the frozen parser."""
    if not returned(record):
        return ReparsedPrediction(None, False, "api_response_not_available", 0)
    if record.get("status") == "error":
        return ReparsedPrediction(None, False, "api_error_response", 0)
    if not isinstance(record.get("raw_response"), str):
        return ReparsedPrediction(None, False, "completion_not_available", 0)
    extraction = _frozen_parser_module().extract_prediction(
        record["raw_response"], setting,
    )
    return ReparsedPrediction(
        prediction=extraction.prediction,
        parse_ok=extraction.parse_ok,
        extraction_status=extraction.status,
        candidate_count=extraction.candidate_count,
    )


def returned(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    if isinstance(record.get("response_received"), bool):
        return record["response_received"]
    if record.get("status") == "ok":
        return True
    if record.get("raw_response") not in (None, "") or record.get("error_response") is not None:
        return True
    # Only the final attempt determines whether a response was received.
    diagnostics = record.get("transport_diagnostics") or []
    final = diagnostics[-1] if diagnostics else record
    status = final.get("http_status", final.get("status_code"))
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        return True
    # Legacy records retained the HTTP response only in the SDK error message.
    return bool(re.search(r"\bError code: [1-5]\d{2}\b", str(record.get("error_message") or "")))


def parsed(record: dict[str, Any] | None) -> bool:
    return bool(
        returned(record)
        and record.get("status") != "error"
        and record.get("parse_ok") is True
        and record.get("prediction") is not None
    )


def prediction(record: dict[str, Any] | None) -> object | None:
    return record.get("prediction") if parsed(record) else None


def context_rows(value: object) -> object:
    return value.get("role_localization") if isinstance(value, dict) else None


def predicted_interval(value: object) -> tuple[tuple[float, float] | None, str]:
    """Prefer an explicit interval; never replace an explicit invalid value."""
    if not isinstance(value, dict):
        return None, "unavailable"
    if "temporal_window" in value:
        result = interval(value.get("temporal_window"))
        return result, "explicit" if result is not None else "explicit_invalid"
    rows = value.get("role_localization")
    if isinstance(rows, list):
        result = timestamps_interval(
            row.get("time_sec") for row in rows if isinstance(row, dict)
        )
        return result, "timestamps_fallback" if result is not None else "unavailable"
    return None, "unavailable"
