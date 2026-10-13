from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..quality_control import load_manual_exclusions


class ManualExclusionTests(unittest.TestCase):
    def test_loads_reviewer_backed_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exclusions.jsonl"
            record = {
                "sample_id": "game/drop", "kept_sample_id": "game/keep",
                "review_group_id": "collision-001",
                "reason": "manual_same_event_duplicate",
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            records, provenance = load_manual_exclusions(path)
        self.assertEqual(records["game/drop"], record)
        self.assertEqual(provenance["manual_exclusion_count"], 1)
        self.assertEqual(len(provenance["manual_exclusion_manifest_sha256"]), 64)

    def test_rejects_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exclusions.jsonl"
            record = {
                "sample_id": "same", "kept_sample_id": "keep",
                "review_group_id": "collision-001",
                "reason": "manual_same_event_duplicate",
            }
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate manual exclusion"):
                load_manual_exclusions(path)


if __name__ == "__main__":
    unittest.main()
