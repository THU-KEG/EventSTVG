"""Frozen extraction-only policy for Role-STVG model responses.

This module extracts JSON predictions.  It intentionally does not validate
prompt-format compliance and does not evaluate prediction correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal


PARSER_POLICY = "role_stvg_json_extraction_v1"
Setting = Literal[
    "oracle", "oracle_with_bbox", "context_1min", "context_3min",
    "context_5min", "context_10min",
]


@dataclass(frozen=True)
class ExtractionResult:
    prediction: Any | None
    status: Literal[
        "complete_json", "whole_response_fence", "unique_embedded_json",
        "no_answer_candidate", "multiple_answer_candidates",
    ]
    candidate_count: int

    @property
    def parse_ok(self) -> bool:
        return self.prediction is not None


def is_answer_shaped(value: Any, setting: Setting) -> bool:
    """Apply only the basic top-level answer shape used for extraction."""
    if setting in {"oracle", "oracle_with_bbox"}:
        return (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(row, dict) and "time_sec" in row for row in value)
        )
    if setting.startswith("context_"):
        return (
            isinstance(value, dict)
            and "temporal_window" in value
            and "role_localization" in value
        )
    raise ValueError(f"unsupported parser setting: {setting!r}")


def _strict_decode(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _answer_candidates(text: str, setting: Setting) -> list[Any]:
    """Locate complete answer-shaped JSON occurrences without repairing them."""
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for offset, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            continue
        if is_answer_shaped(value, setting):
            candidates.append(value)
    return candidates


def extract_prediction(raw: str, setting: Setting) -> ExtractionResult:
    """Extract exactly one answer occurrence under the frozen policy."""
    text = raw.strip()

    value = _strict_decode(text)
    if value is not None and is_answer_shaped(value, setting):
        return ExtractionResult(value, "complete_json", 1)

    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE,
    )
    # A response containing additional fence markers is not one whole-response
    # fence; it must go through occurrence counting below.
    if fence and "```" not in fence.group(1):
        value = _strict_decode(fence.group(1))
        if value is not None and is_answer_shaped(value, setting):
            return ExtractionResult(value, "whole_response_fence", 1)
        return ExtractionResult(None, "no_answer_candidate", 0)

    candidates = _answer_candidates(text, setting)
    if len(candidates) == 1:
        return ExtractionResult(candidates[0], "unique_embedded_json", 1)
    if not candidates:
        return ExtractionResult(None, "no_answer_candidate", 0)
    return ExtractionResult(None, "multiple_answer_candidates", len(candidates))


def parse_response(raw: str, setting: Setting) -> Any | None:
    """Compatibility wrapper returning only the extracted prediction."""
    return extract_prediction(raw, setting).prediction
