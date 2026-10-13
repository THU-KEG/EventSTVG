"""Explicit Stage-1 trim quality-control exclusions.

Entries here are excluded before target eligibility and matched-trim inventory
construction, so they cannot be used as either evaluation targets or visual
distractors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

INVALID_SAMPLE_REASONS = {
    (
        "italy_serie-a/2016-2017/"
        "2016-09-21 - 21-45 Genoa 0 - 0 Napoli/"
        "Kick-off_H2_17"
    ): "explicit_qc_problem_reported",
}

def invalid_sample_reason(sample_id: str) -> str | None:
    return INVALID_SAMPLE_REASONS.get(sample_id)


def load_manual_exclusions(path: Path) -> tuple[dict[str, dict], dict]:
    """Load reviewer-backed exclusions with reproducible provenance."""
    records: dict[str, dict] = {}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        payload = handle.read()
    digest.update(payload)
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = record["sample_id"]
        if sample_id in records:
            raise ValueError(
                f"duplicate manual exclusion for {sample_id} at line {line_number}"
            )
        if record.get("reason") != "manual_same_event_duplicate":
            raise ValueError(f"unsupported manual exclusion reason at line {line_number}")
        if not record.get("kept_sample_id") or not record.get("review_group_id"):
            raise ValueError(f"incomplete manual exclusion at line {line_number}")
        records[sample_id] = record
    return records, {
        "manual_exclusion_manifest_path": str(path.resolve()),
        "manual_exclusion_manifest_sha256": digest.hexdigest(),
        "manual_exclusion_count": len(records),
        "manual_exclusion_reason": "manual_same_event_duplicate",
    }
