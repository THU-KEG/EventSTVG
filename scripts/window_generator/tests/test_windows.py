from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..events import (
    EventIndex,
    NORMALIZED_EVENT_TYPES,
    normalized_event_type,
    occurrences_from_labels,
    sequence_event_type,
)
from ..generator import (
    _generation_provenance,
    _run_ffmpeg,
    _sample_metadata,
    _validate_existing_video,
    generate_all,
    target_clip_within_half,
)
from ..models import EventOccurrence, TargetClip
from ..sources import Roots, discover_trim_inventory, probe_duration
from ..windows import (
    choose_interval,
    contained_same_type_trims,
    generate_window,
    stable_seed,
)


EXPECTED_MAPPING = {
    ("Clearance", None): "goal_kick",
    ("Corner", "w1"): "corner_won",
    ("Corner", "w2"): "corner_kick",
    ("Direct_free-kick", "w1"): "foul_direct_free_kick",
    ("Direct_free-kick", "w2"): "direct_free_kick",
    ("Goal", None): "goal",
    ("Kick-off", None): "kick_off",
    ("Offside", "w1"): "offside",
    ("Offside", "w2"): "indirect_free_kick",
    ("Penalty", "w1"): "foul_penalty",
    ("Penalty", "w2"): "penalty_kick",
    ("Red_card", None): "foul_red_card",
    ("Shots_off_target", None): "shot_off_target",
    ("Shots_on_target", None): "shot_on_target",
    ("Throw-in", None): "throw_in",
    ("Yellow_card", None): "foul_yellow_card",
    ("Yellowtored_card", None): "foul_second_yellow_red_card",
}


def occurrence(
    name: str, time: float, sequence_type: str = "Goal"
) -> EventOccurrence:
    return EventOccurrence(
        source_occurrence_id=name,
        game="league/season/game",
        half=1,
        sequence_event_type=sequence_type,
        soccernet_event_time_abs=time,
        source_label=sequence_type,
        source_position_ms=round(time * 1000),
    )


def target(
    item: EventOccurrence,
    start: float,
    end: float,
    target_type: str = "goal",
    sample_id: str | None = None,
    path: Path = Path("/tmp/example"),
) -> TargetClip:
    return TargetClip(
        sample_id=sample_id or f"sample-{item.source_occurrence_id}",
        game=item.game,
        half=1,
        source_clip=path,
        role_annotation_path=path,
        trim_annotation_path=path,
        source_half_video=path,
        filename_label=item.source_label,
        event_type=target_type,
        sequence_event_type=item.sequence_event_type,
        event_position=item.soccernet_event_time_abs,
        target_event_time=item.soccernet_event_time_abs,
        clip_start=start,
        clip_end=end,
        occurrence=item,
        source_timestamp_in_trim=start <= item.soccernet_event_time_abs <= end,
    )


class MappingTests(unittest.TestCase):
    def test_all_required_mappings(self):
        self.assertEqual(len(EXPECTED_MAPPING), len(NORMALIZED_EVENT_TYPES))
        for (label, window), expected in EXPECTED_MAPPING.items():
            self.assertEqual(normalized_event_type(label, window), expected)

    def test_paired_windows_share_source_sequence_but_not_role_type(self):
        for label, first, second, sequence in (
            ("Corner", "corner_won", "corner_kick", "Corner"),
            ("Direct_free-kick", "foul_direct_free_kick",
             "direct_free_kick", "Direct free-kick"),
            ("Penalty", "foul_penalty", "penalty_kick", "Penalty"),
            ("Offside", "offside", "indirect_free_kick", "Offside"),
        ):
            self.assertEqual(normalized_event_type(label, "w1"), first)
            self.assertEqual(normalized_event_type(label, "w2"), second)
            self.assertNotEqual(first, second)
            self.assertEqual(sequence_event_type(first), sequence)
            self.assertEqual(sequence_event_type(second), sequence)

    def test_different_source_sequences_coexist_at_same_timestamp(self):
        foul = occurrence("foul", 100, "Direct free-kick")
        card = occurrence("card", 100, "Yellow card")
        index = EventIndex([foul, card])
        self.assertEqual(len(index.occurrences), 2)

    def test_same_source_annotation_is_deduplicated(self):
        first = occurrence("first", 100, "Direct free-kick")
        second = occurrence("second", 100, "Direct free-kick")
        self.assertEqual(len(EventIndex([first, second]).occurrences), 1)

    def test_event_time_comes_from_labels_v2_position(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "Labels-v2.json"
            labels.write_text(json.dumps({"annotations": [{
                "gameTime": "1 - 02:03",
                "label": "Corner",
                "position": "123456",
            }]}), encoding="utf-8")
            records = occurrences_from_labels(labels, "league/season/game")
        self.assertEqual(records[0].soccernet_event_time_abs, 123.456)

    def test_quality_control_sample_never_enters_targets_or_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative_parent = Path(
                "england_epl/2015-2016/"
                "2015-10-03 - 19-30 Chelsea 1 - 3 Southampton"
            )
            stem = "Shots_on_target_H2_202233"
            roots = Roots(
                root / "clips", root / "roles", root / "trims", root / "data"
            )
            for base in roots.__dict__.values():
                (base / relative_parent).mkdir(parents=True)
            (roots.trimmed_clips / relative_parent / f"{stem}.mp4").touch()
            (roots.role_annotations / relative_parent /
             f"{stem}_annotation.json").write_text("{}")
            (roots.role_annotations / relative_parent /
             f"{stem}_problem.json").write_text(json.dumps({"clip": stem}))
            (roots.trim_annotations / relative_parent /
             f"{stem}_trim.json").write_text(json.dumps({
                 "game_ts_start": 200,
                 "game_ts_end": 205,
             }))
            (roots.soccernet_data / relative_parent /
             "Labels-v2.json").write_text("{}")
            (roots.soccernet_data / relative_parent / "2_720p.mp4").touch()
            source = EventOccurrence(
                source_occurrence_id="source",
                game=relative_parent.as_posix(),
                half=2,
                sequence_event_type="Shots on target",
                soccernet_event_time_abs=202.233,
                source_label="Shots on target",
                source_position_ms=202233,
            )
            targets, inventory, removed = discover_trim_inventory(
                roots, EventIndex([source]), 2.0
            )
        self.assertEqual(targets, [])
        self.assertEqual(inventory, [])
        self.assertEqual(
            removed[0]["reason"],
            "problem_flag_present",
        )
        self.assertTrue(removed[0]["quality_control_exclusion"])

    def test_reported_kickoff_is_the_only_game_specific_qc_exclusion(self):
        from ..quality_control import invalid_sample_reason
        prefix = (
            "italy_serie-a/2016-2017/"
            "2016-09-21 - 21-45 Genoa 0 - 0 Napoli/"
        )
        self.assertEqual(
            invalid_sample_reason(prefix + "Kick-off_H2_17"),
            "explicit_qc_problem_reported",
        )
        self.assertIsNone(invalid_sample_reason(prefix + "Corner_H2_80449_w1"))

    def test_problem_flag_never_enters_targets_or_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative_parent = Path("league/season/game")
            stem = "Goal_H1_100000"
            roots = Roots(
                root / "clips", root / "roles", root / "trims", root / "data"
            )
            for base in roots.__dict__.values():
                (base / relative_parent).mkdir(parents=True)
            (roots.trimmed_clips / relative_parent / f"{stem}.mp4").touch()
            (roots.role_annotations / relative_parent /
             f"{stem}_annotation.json").write_text("{}")
            problem = (
                roots.role_annotations / relative_parent /
                f"{stem}_problem.json"
            )
            problem.write_text("{}")
            targets, inventory, removed = discover_trim_inventory(
                roots, EventIndex([]), 2.0
            )
        self.assertEqual(targets, [])
        self.assertEqual(inventory, [])
        self.assertEqual(removed[0]["reason"], "problem_flag_present")
        self.assertEqual(removed[0]["problem_flag"], str(problem.resolve()))
        self.assertTrue(removed[0]["quality_control_exclusion"])


class OracleTests(unittest.TestCase):
    def test_policy_v8_oracle_is_target_only(self):
        target_event = occurrence("target", 200)
        other_event = occurrence("other", 210)
        clip = target(target_event, 190, 205, path=Path("/tmp"))
        contained = contained_same_type_trims(
            clip, [clip, target(other_event, 208, 212, sample_id="other", path=Path("/tmp"))],
            190, 205,
        )
        metadata = _sample_metadata(
            clip, "oracle", 190, 205, contained, 1,
            {"contains_target_trim": True}, clip.source_clip,
            [clip], 7, 10, qualified_occurrences=(target_event,),
            policy_version=8,
        )
        self.assertEqual(metadata["qualified_source_occurrence_ids"], ["target"])
        self.assertEqual(metadata["same_event_count"], 1)
        self.assertEqual(metadata["target_event_order"], 1)
        self.assertEqual(metadata["num_prior_same_events"], 0)
        self.assertEqual(metadata["num_same_event_distractors"], 0)
        self.assertFalse(metadata["requires_disambiguation"])

    def test_outside_source_timestamp_is_external_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "half.mp4"
            source.write_bytes(b"signature")
            item = occurrence("target", 200)
            clip = target(item, 10, 15, path=source)
            contained = contained_same_type_trims(clip, [clip], 10, 15)
            metadata = _sample_metadata(
                clip, "oracle", 10, 15, contained, 1,
                {"contains_target_trim": True}, clip.source_clip, [clip], 7, 10,
            )
        self.assertFalse(metadata["source_timestamp_inside_context"])
        self.assertEqual(metadata["contained_same_type_occurrence_ids"],
                         ["target"])
        self.assertEqual(metadata["target_trim_start_rel"], 0)
        self.assertEqual(metadata["target_trim_end_rel"], 5)
        self.assertEqual(metadata["same_event_count"], 1)
        self.assertEqual(metadata["num_prior_same_events"], 0)
        self.assertFalse(metadata["requires_disambiguation"])

    def test_distinct_same_type_occurrence_is_retained_by_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            first = occurrence("first", 110)
            second = occurrence("second", 125)
            clip = target(first, 100, 200, path=source)
            other = target(
                second, 120, 130, sample_id="other", path=source
            )
            with patch(
                "window_generator.generator.probe_duration",
                return_value=1000.0,
            ):
                generate_all(
                    [clip], [clip, other], [],
                    EventIndex([first, second]), root / "output", 7,
                    True, False, "ffprobe", 10,
                )
            manifest = json.loads(
                (root / "output/oracle/manifest.json").read_text()
            )
        self.assertEqual(manifest["summary"]["total_samples"], 1)
        self.assertEqual(
            manifest["summary"]["removal_reason_distribution"],
            {},
        )
        self.assertEqual(
            manifest["summary"]["same_event_count_distribution"], {"2": 1}
        )
        self.assertEqual(
            manifest["summary"]["target_event_order_distribution"], {"1": 1}
        )


class OneMinuteTests(unittest.TestCase):
    def test_policy_v8_one_minute_allows_repeated_occurrence(self):
        item = occurrence("target", 500)
        other = occurrence("other", 520)
        clip = target(item, 490, 500)
        result = generate_window(
            clip, [clip], [item, other], "context_1min", 1000, 0, 10, 8
        )
        self.assertTrue(result.retained)
        self.assertEqual(len(result.contained_same_type_occurrences), 2)
        self.assertEqual(result.target_order, 1)

    def test_target_timestamp_may_be_outside_and_target_is_exempt(self):
        item = occurrence("target", 500)
        clip = target(item, 100, 110)
        result = generate_window(
            clip, [clip], [item], "context_1min", 700, 7, 10
        )
        self.assertTrue(result.retained)
        self.assertFalse(result.start <= item.soccernet_event_time_abs
                         <= result.end)
        self.assertEqual(len(result.contained_same_type_occurrences), 1)
        self.assertEqual(result.target_order, 1)

    def test_overlapping_same_type_trim_invalidates_fixed_start(self):
        item = occurrence("target", 500)
        clip = target(item, 100, 160)
        other = target(
            occurrence("other", 600), 99, 100.1, sample_id="other"
        )
        result = generate_window(
            clip, [clip, other], [item, other.occurrence],
            "context_1min", 700, 7, 0,
        )
        self.assertFalse(result.retained)

    def test_different_role_type_does_not_invalidate(self):
        item = occurrence("target", 500, "Corner")
        clip = target(item, 100, 160, "corner_won")
        paired = target(
            item, 120, 130, "corner_kick", sample_id="paired-w2"
        )
        result = generate_window(
            clip, [clip, paired], [item], "context_1min", 700, 7, 10
        )
        self.assertTrue(result.retained)

    def test_other_same_sequence_occurrence_invalidates(self):
        item = occurrence("target", 500)
        other = occurrence("other", 120)
        clip = target(item, 100, 160)
        result = generate_window(
            clip, [clip], [other, item], "context_1min", 700, 7, 10
        )
        self.assertFalse(result.retained)

    def test_other_soccernet_type_does_not_invalidate(self):
        item = occurrence("target", 500, "Goal")
        clip = target(item, 100, 160)
        unrelated = occurrence("card", 120, "Yellow card")
        result = generate_window(
            clip, [clip], [item], "context_1min", 700, 7, 10
        )
        self.assertTrue(result.retained)
        self.assertNotEqual(unrelated.sequence_event_type,
                            clip.sequence_event_type)

    def test_safety_margin_boundary_is_excluded(self):
        item = occurrence("target", 500)
        clip = target(item, 100, 160)
        at_boundary = occurrence("other", 90)
        excluded = generate_window(
            clip, [clip], [at_boundary, item],
            "context_1min", 700, 7, 10,
        )
        just_outside = occurrence("other", 89.998)
        allowed = generate_window(
            clip, [clip], [just_outside, item],
            "context_1min", 700, 7, 10,
        )
        self.assertFalse(excluded.retained)
        self.assertTrue(allowed.retained)


class ThreeMinuteTests(unittest.TestCase):
    def test_policy_v8_three_minute_window_is_exact_and_contains_target(self):
        item = occurrence("target", 500)
        clip = target(item, 490, 500)
        result = generate_window(
            clip, [clip], [item], "context_3min", 1000, 42, 10, 8
        )
        self.assertTrue(result.retained)
        self.assertAlmostEqual(result.end - result.start, 180.0)
        self.assertLessEqual(result.start, clip.clip_start)
        self.assertGreaterEqual(result.end, clip.clip_end)


class FiveMinuteTests(unittest.TestCase):
    def test_policy_v8_five_minute_allows_single_occurrence(self):
        item = occurrence("target", 500)
        clip = target(item, 490, 500)
        result = generate_window(
            clip, [clip], [item], "context_5min", 1000, 0, 10, 8
        )
        self.assertTrue(result.retained)
        self.assertEqual(len(result.contained_same_type_occurrences), 1)

    def test_policy_v8_ordinal_uses_canonical_time_not_trim_chronology(self):
        item = occurrence("target", 500)
        other = occurrence("other", 400)
        clip = target(item, 490, 500)
        other_trim = target(other, 300, 310, sample_id="other")
        result = generate_window(
            clip, [clip, other_trim], [other, item], "context_5min",
            1000, 0, 10, 8
        )
        self.assertTrue(result.retained)
        self.assertEqual(result.target_order, 2)
        self.assertEqual(
            [x.source_occurrence_id for x in result.contained_same_type_occurrences],
            ["other", "target"],
        )
    def test_contained_same_type_trim_with_external_timestamps_is_valid(self):
        first = occurrence("first", 600, "Corner")
        second = occurrence("second", 650, "Corner")
        clip = target(first, 100, 110, "corner_won")
        distractor = target(
            second, 350, 360, "corner_won", sample_id="distractor"
        )
        result = generate_window(
            clip, [clip, distractor], [first, second],
            "context_5min", 700, 7,
        )
        self.assertTrue(result.retained)
        self.assertEqual(len(result.contained_same_type_occurrences), 2)
        self.assertEqual(result.target_order, 1)
        self.assertTrue(all(
            not (result.start <= occurrence_.soccernet_event_time_abs
                 <= result.end)
            for occurrence_ in result.contained_same_type_occurrences
        ))

    def test_timestamp_without_contained_trim_is_insufficient(self):
        first = occurrence("first", 100)
        second = occurrence("second", 200)
        clip = target(first, 100, 110)
        result = generate_window(
            clip, [clip], [first, second], "context_5min", 700, 7
        )
        self.assertFalse(result.retained)
        self.assertEqual(result.reason,
                         "no_fully_contained_same_type_distractor")

    def test_partially_overlapping_trim_is_insufficient(self):
        first = occurrence("first", 100)
        second = occurrence("second", 400)
        clip = target(first, 100, 110)
        partial = target(second, 350, 410, sample_id="partial")
        result = generate_window(
            clip, [clip, partial], [first, second],
            "context_5min", 700, 7,
        )
        self.assertFalse(result.retained)

    def test_different_role_type_is_insufficient(self):
        source1 = occurrence("first", 100, "Corner")
        source2 = occurrence("second", 200, "Corner")
        won = target(source1, 100, 110, "corner_won")
        kick = target(
            source2, 200, 210, "corner_kick", sample_id="kick"
        )
        result = generate_window(
            won, [won, kick], [source1, source2],
            "context_5min", 700, 7,
        )
        self.assertFalse(result.retained)

    def test_shared_occurrence_duplicates_are_deduplicated(self):
        first = occurrence("first", 100)
        second = occurrence("second", 200)
        clip = target(first, 100, 110)
        duplicate = target(
            first, 120, 130, sample_id="same-occurrence-duplicate"
        )
        distractor = target(
            second, 200, 210, sample_id="distinct-occurrence"
        )
        result = generate_window(
            clip, [clip, duplicate, distractor], [first, second],
            "context_5min", 700, 7,
        )
        self.assertTrue(result.retained)
        self.assertEqual(len(result.contained_same_type_trims), 2)

    def test_order_comes_from_source_occurrences(self):
        occurrences = [
            occurrence("one", 100), occurrence("two", 200),
            occurrence("three", 300),
        ]
        trims = [
            target(occurrences[2], 200, 210, sample_id="filename-a"),
            target(occurrences[0], 50, 60, sample_id="filename-z"),
            target(occurrences[1], 250, 260, sample_id="target"),
        ]
        result = generate_window(
            trims[2], trims, occurrences, "context_5min", 700, 4
        )
        self.assertTrue(result.retained)
        self.assertEqual(len(result.contained_same_type_trims), 3)
        self.assertEqual(result.target_order, 2)

    def test_partial_same_type_trim_crossing_either_boundary_is_rejected(self):
        source = [
            occurrence("left", 100), occurrence("target", 200),
            occurrence("distractor", 300), occurrence("right", 400),
        ]
        clip = target(source[1], 100, 400, sample_id="target")
        distractor = target(
            source[2], 200, 210, sample_id="distractor"
        )
        for partial in (
            target(source[0], 95, 101, sample_id="partial-left"),
            target(source[3], 399, 405, sample_id="partial-right"),
        ):
            result = generate_window(
                clip, [clip, distractor, partial], source,
                "context_5min", 700, 7,
            )
            self.assertFalse(result.retained)
            self.assertEqual(
                result.reason,
                "partial_same_type_trim_at_5min_context_boundary",
            )

    def test_complete_or_disjoint_same_type_trims_remain_valid(self):
        source = [
            occurrence("target", 200), occurrence("inside", 300),
            occurrence("disjoint", 400),
        ]
        clip = target(source[0], 100, 400, sample_id="target")
        inside = target(source[1], 200, 210, sample_id="inside")
        disjoint = target(source[2], 400, 410, sample_id="disjoint")
        result = generate_window(
            clip, [clip, inside, disjoint], source,
            "context_5min", 700, 7,
        )
        self.assertTrue(result.retained)
        self.assertEqual(
            [item.occurrence.source_occurrence_id
             for item in result.contained_same_type_trims],
            ["target", "inside"],
        )
        self.assertEqual(result.target_order, 1)


class TenMinuteTests(unittest.TestCase):
    def test_untrimmed_timestamp_inside_context_counts_toward_ordinal(self):
        earlier = occurrence("untrimmed-earlier", 105)
        target_occurrence = occurrence("target", 650)
        clip = target(target_occurrence, 100, 110)
        result = generate_window(
            clip, [clip], [earlier, target_occurrence],
            "context_10min", 700, 7,
        )
        self.assertTrue(result.retained)
        self.assertEqual(
            [item.source_occurrence_id
             for item in result.contained_same_type_occurrences],
            ["untrimmed-earlier", "target"],
        )
        self.assertEqual(result.target_order, 2)

    def test_occurrence_entering_by_timestamp_and_trim_is_deduplicated(self):
        target_occurrence = occurrence("target", 100)
        other_occurrence = occurrence("other", 200)
        clip = target(target_occurrence, 100, 110)
        other = target(other_occurrence, 200, 210, sample_id="other-trim")
        result = generate_window(
            clip, [clip, other], [target_occurrence, other_occurrence],
            "context_10min", 700, 7,
        )
        self.assertTrue(result.retained)
        self.assertEqual(len(result.contained_same_type_occurrences), 2)

    def test_counts_one_two_and_three_are_all_accepted(self):
        for count in (1, 2, 3):
            occurrences = [
                occurrence(str(i), 100 + i * 100) for i in range(count)
            ]
            trims = [
                target(item, 100 + i * 100, 110 + i * 100,
                       sample_id=f"trim-{i}")
                for i, item in enumerate(occurrences)
            ]
            result = generate_window(
                trims[0], trims, occurrences,
                "context_10min", 700, 7,
            )
            self.assertTrue(result.retained)
            self.assertEqual(len(result.contained_same_type_trims), count)

    def test_target_trim_and_half_are_only_base_constraints(self):
        item = occurrence("target", 2000)
        clip = target(item, 680, 700)
        result = generate_window(
            clip, [clip], [item], "context_10min", 720, 7
        )
        self.assertTrue(result.retained)
        self.assertGreaterEqual(result.end, clip.clip_end)
        self.assertLessEqual(result.end, 720)

    def test_partial_same_type_trim_is_rejected(self):
        source = [occurrence("target", 200), occurrence("partial", 400)]
        clip = target(source[0], 100, 700, sample_id="target")
        partial = target(source[1], 699, 710, sample_id="partial")
        result = generate_window(
            clip, [clip, partial], source, "context_10min", 800, 7
        )
        self.assertFalse(result.retained)
        self.assertEqual(
            result.reason,
            "partial_same_type_trim_at_10min_context_boundary",
        )


class DeterminismAndProvenanceTests(unittest.TestCase):
    def test_stable_configuration_reproduces_interval(self):
        item = occurrence("target", 500)
        clip = target(item, 100, 110)
        first = generate_window(
            clip, [clip], [item], "context_10min", 700, 7
        )
        second = generate_window(
            clip, [clip], [item], "context_10min", 700, 7
        )
        self.assertEqual(first.start, second.start)

    def test_union_sampling_is_weighted_by_continuous_length(self):
        short_interval_hits = sum(
            choose_interval([(0, 1), (10, 19)], seed) <= 1
            for seed in range(2000)
        )
        self.assertGreater(short_interval_hits, 140)
        self.assertLess(short_interval_hits, 260)

    def test_changed_configuration_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"source")
            clip = target(occurrence("target", 5), 1, 2, path=source)
            first = _generation_provenance(
                clip, "context_1min", 0, 60, 7, 10
            )
            changed = _generation_provenance(
                clip, "context_1min", 0, 60, 7, 5
            )
        self.assertNotEqual(first["generation_fingerprint"],
                            changed["generation_fingerprint"])

    def test_stale_existing_video_without_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            existing = root / "context.mp4"
            existing.write_bytes(b"old")
            clip = target(occurrence("target", 5), 1, 2, path=source)
            expected = _generation_provenance(
                clip, "context_1min", 0, 60, 7, 10
            )
            with self.assertRaisesRegex(RuntimeError, "no provenance"):
                _validate_existing_video(existing, expected, "ffprobe")

    def test_abnormal_target_trim_is_rejected(self):
        item = occurrence("target", 2718.789, "Shots off target")
        abnormal = target(item, 2726.008, 2732.277)
        self.assertFalse(target_clip_within_half(abnormal, 2730.024))


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "ffmpeg and ffprobe are required",
)
class VideoIntegrationTests(unittest.TestCase):
    def test_generated_video_and_shifted_annotation_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "context.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ], check=True)
            _run_ffmpeg(source, output, 1.0, 2.0)
            actual_duration = probe_duration(output)
            self.assertAlmostEqual(actual_duration, 2.0, delta=0.15)

            item = occurrence("target", 20)
            clip = target(item, 1.5, 2.5, path=source)
            contained = contained_same_type_trims(clip, [clip], 1.0, 3.0)
            metadata = _sample_metadata(
                clip, "oracle", 1.0, 3.0, contained, 1,
                {"contains_target_trim": True}, output, [clip], 7, 10,
            )
            # A Role-STVG annotation at 0.4s inside the trim lands at 0.9s
            # inside the generated context and at frame 9 for the 10fps source.
            annotation_time_in_trim = 0.4
            shifted_time = (
                metadata["target_trim_start_rel"] + annotation_time_in_trim
            )
            shifted_frame = round(shifted_time * 10)
            self.assertAlmostEqual(shifted_time, 0.9)
            self.assertEqual(shifted_frame, 9)
            self.assertGreaterEqual(shifted_frame, 0)
            self.assertLess(shifted_frame, round(actual_duration * 10))


if __name__ == "__main__":
    unittest.main()
