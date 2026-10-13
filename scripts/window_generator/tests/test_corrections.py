from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..corrections import apply_corrections, sha256_file
from ..events import EventIndex
from ..generator import generate_all
from ..models import EventOccurrence, TargetClip
from ..windows import generate_window


def occurrence(name: str, timestamp: float, sequence="Corner"):
    return EventOccurrence(name, "league/season/game", 1, sequence,
                           timestamp, sequence, round(timestamp * 1000))


def target(root: Path, sample_id: str, item: EventOccurrence,
           event_type="corner_kick", start=100.0, end=103.0):
    video = root / (sample_id.replace("/", "_") + ".mp4")
    video.parent.mkdir(parents=True, exist_ok=True); video.write_bytes(b"video")
    return TargetClip(
        sample_id, item.game, 1, video, video, video, video, "Corner",
        event_type, item.sequence_event_type, item.soccernet_event_time_abs,
        item.soccernet_event_time_abs, start, end, item,
        start <= item.soccernet_event_time_abs <= end,
    )


def record(item: TargetClip, status="verified", start=110.0, end=113.0,
           offset=10.0):
    return {
        "sample_id": item.sample_id, "status": status,
        "local_t_s": 1.0, "local_t_e": 4.0,
        "stored_game_ts_start": item.clip_start,
        "stored_game_ts_end": item.clip_end,
        "corrected_game_ts_start": start, "corrected_game_ts_end": end,
        "offset_sec": offset, "confidence": .99,
        "recovery_method": "test",
    }


def write_sidecar(path: Path, records: list[dict]):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                            for row in records))


class CorrectionOverlayTests(unittest.TestCase):
    def test_small_negative_start_clamps_only_with_verified_manual_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = target(root, "kickoff", occurrence("a", 0), "kick_off", 0, 3)
            sidecar = root / "c.jsonl"
            write_sidecar(sidecar, [record(item, start=-0.08, end=2.92, offset=-0.08)])
            audit = root / "audit.jsonl"
            write_sidecar(audit, [{"sample_id": item.sample_id, "result": "pass"}])
            with patch("window_generator.corrections.probe_frame_rate", return_value=25.0):
                corrected, _, _, provenance = apply_corrections(
                    [item], [item], [], sidecar, False, 0.08, audit)
        self.assertEqual(corrected[0].clip_start, 0.0)
        self.assertEqual(corrected[0].timestamp_correction["raw_corrected_game_ts_start"], -0.08)
        self.assertTrue(corrected[0].timestamp_correction["source_boundary_clamped"])
        self.assertEqual(corrected[0].timestamp_correction["source_boundary_clamp_amount_sec"], 0.08)
        self.assertEqual(corrected[0].timestamp_correction["source_frame_tolerance_sec"], 0.08)
        self.assertEqual(corrected[0].timestamp_correction["source_boundary_clamp_reason"],
                         "verified_manual_pass_within_two_source_frames")
        self.assertEqual(provenance["source_boundary_clamped_count"], 1)

    def test_negative_start_has_no_clamp_without_manual_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = target(root, "kickoff", occurrence("a", 0), "kick_off", 0, 3)
            sidecar = root / "c.jsonl"
            write_sidecar(sidecar, [record(item, start=-0.08, end=2.92, offset=-0.08)])
            audit = root / "audit.jsonl"
            write_sidecar(audit, [{"sample_id": item.sample_id, "result": "fail"}])
            corrected = apply_corrections(
                [item], [item], [], sidecar, False, 0.08, audit)[0]
        self.assertEqual(corrected[0].clip_start, -0.08)
        self.assertFalse(corrected[0].timestamp_correction["source_boundary_clamped"])

    def test_negative_and_large_positive_offsets_are_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); first = target(root, "first", occurrence("a", 100))
            second = target(root, "second", occurrence("b", 200))
            sidecar = root / "corrections.jsonl"
            write_sidecar(sidecar, [record(first, start=93, end=96, offset=-7),
                                    record(second, start=155, end=158, offset=55)])
            targets, inventory, removed, _ = apply_corrections(
                [first, second], [first, second], [], sidecar)
        self.assertEqual([(x.clip_start, x.clip_end) for x in targets],
                         [(93, 96), (155, 158)])
        self.assertEqual(removed, []); self.assertEqual(len(inventory), 2)

    def test_missing_and_ambiguous_have_no_silent_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); missing = target(root, "missing", occurrence("a", 100))
            ambiguous = target(root, "ambiguous", occurrence("b", 200))
            sidecar = root / "corrections.jsonl"
            write_sidecar(sidecar, [record(ambiguous, "ambiguous")])
            targets, inventory, removed, _ = apply_corrections(
                [missing, ambiguous], [missing, ambiguous], [], sidecar)
        self.assertEqual(targets, []); self.assertEqual(inventory, [])
        self.assertEqual({row["reason"] for row in removed}, {
            "timestamp_correction_missing", "timestamp_correction_status_ambiguous"})

    def test_probable_disabled_by_default_and_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); item = target(root, "probable", occurrence("a", 100))
            sidecar = root / "corrections.jsonl"; write_sidecar(sidecar, [record(item, "probable")])
            disabled = apply_corrections([item], [item], [], sidecar)
            enabled = apply_corrections([item], [item], [], sidecar, True)
        self.assertEqual(disabled[0], []); self.assertEqual(len(enabled[0]), 1)

    def test_manifest_hash_and_original_corrected_bounds_are_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); item = target(root, "one", occurrence("a", 100))
            sidecar = root / "corrections.jsonl"; write_sidecar(sidecar, [record(item)])
            targets, _, _, provenance = apply_corrections([item], [item], [], sidecar)
            expected_hash = sha256_file(sidecar)
        correction = targets[0].timestamp_correction
        self.assertEqual(provenance["correction_manifest_sha256"], expected_hash)
        self.assertEqual(correction["original_stored_game_ts_start"], 100)
        self.assertEqual(correction["corrected_game_ts_start"], 110)

    def test_corrected_target_previously_outside_one_minute_is_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = occurrence("a", 671, "Direct free-kick")
            item = target(root, "foul", source, "direct_free_kick", 630, 633)
            sidecar = root / "c.jsonl"; write_sidecar(
                sidecar, [record(item, start=671, end=674, offset=41)])
            corrected = apply_corrections([item], [item], [], sidecar)[0][0]
            result = generate_window(corrected, [corrected], [source],
                                     "context_1min", 1000, 7, 10)
        self.assertTrue(result.retained)
        self.assertLessEqual(result.start, corrected.clip_start)
        self.assertGreaterEqual(result.end, corrected.clip_end)

    def test_corrected_foul_count_and_ordinal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_occ = occurrence("a", 100, "Yellow card")
            second_occ = occurrence("b", 200, "Yellow card")
            first = target(root, "f1", first_occ, "foul_yellow_card", 100, 103)
            second = target(root, "f2", second_occ, "foul_yellow_card", 200, 203)
            sidecar = root / "c.jsonl"; write_sidecar(sidecar, [
                record(first, start=100, end=103, offset=0),
                record(second, start=200, end=203, offset=0)])
            corrected = apply_corrections(
                [first, second], [first, second], [], sidecar)[0]
            result = generate_window(corrected[1], corrected,
                                     [first_occ, second_occ],
                                     "context_5min", 1000, 7)
        self.assertTrue(result.retained)
        self.assertEqual(len(result.contained_same_type_trims), 2)
        self.assertEqual(result.target_order, 2)

    def test_w1_w2_corrections_remain_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = occurrence("same", 100)
            w1 = target(root, "Corner_H1_100000_w1", source, "corner_won")
            w2 = target(root, "Corner_H1_100000_w2", source, "corner_kick")
            sidecar = root / "c.jsonl"; write_sidecar(sidecar, [
                record(w1, start=90, end=93, offset=-10),
                record(w2, start=105, end=108, offset=5)])
            corrected = apply_corrections([w1, w2], [w1, w2], [], sidecar)[0]
        self.assertEqual([(x.event_type, x.clip_start) for x in corrected],
                         [("corner_won", 90), ("corner_kick", 105)])

    def test_corrected_manifests_are_byte_identical_on_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = occurrence("a", 110)
            item = target(root, "one", source, start=100, end=103)
            sidecar = root / "c.jsonl"; write_sidecar(sidecar, [record(item)])
            corrected, inventory, removed, provenance = apply_corrections(
                [item], [item], [], sidecar)
            output = root / "output"
            with patch("window_generator.generator.probe_duration", return_value=1000):
                generate_all(corrected, inventory, removed, EventIndex([source]),
                             output, 7, False, False, "ffprobe", 10, True, provenance)
                before = {p.relative_to(output): p.read_bytes()
                          for p in output.rglob("*") if p.is_file()}
                generate_all(corrected, inventory, removed, EventIndex([source]),
                             output, 7, False, False, "ffprobe", 10, True, provenance)
                after = {p.relative_to(output): p.read_bytes()
                         for p in output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
