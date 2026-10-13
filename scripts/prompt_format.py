#!/usr/bin/env python3
"""Generate and validate the final Role-STVG benchmark JSONL artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parent
SOCCERROLE = ROOT / "soccerrole"
TEMPORAL = SOCCERROLE / "temporal"
SPATIAL = SOCCERROLE / "spatial"
ORDINAL = SOCCERROLE / "ordinal"
SPEC_PATH = SOCCERROLE / "prompt_spec.json"
SOURCE = Path("frozen_manifests")
DEFAULT_OUTPUT = SOCCERROLE / "benchmark"
SETTINGS = (
    "oracle", "context_1min", "context_3min", "context_5min", "context_10min",
)
CONTEXTS = SETTINGS[1:]
EXPECTED_SAMPLES = 3516
SHOT_EVENTS = {"shot_on_target", "shot_off_target", "goal"}
STAGE_EVENT_BY_BENCHMARK_EVENT = {
    "foul_direct_free_kick": "foul",
    "foul_penalty": "foul",
    "foul_yellow_card": "yellow_card",
    "foul_red_card": "red_card",
    "foul_second_yellow_red_card": "second_yellow_red_card",
}
ORDINAL_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth",
}
REQUIRED_METADATA = {
    "record_schema_version", "sample_id", "game", "half", "event_type",
    "setting", "video_path", "spatial_annotation_path", "active_roles",
    "window_start", "window_end", "target_start_in_window",
    "target_end_in_window", "sn_anchor_time_abs",
    "target_event_annotation_identifiers",
    "prompt_spec_version", "prompt_spec_sha256", "predicate_role_structure",
}
GENERATOR_LEAKS = (
    "codex need", "this is for generating", "based on the “requires_disambiguation”",
    "prompt-generation rule", "skip next step", "generator itself",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0 else rounded


def normalize_bbox(box: list[float], width: float, height: float) -> list[float]:
    return [
        number(1000 * box[0] / width), number(1000 * box[1] / height),
        number(1000 * box[2] / width), number(1000 * box[3] / height),
    ]


def occurrence_phrase(order: int, requires_disambiguation: bool) -> str:
    if not requires_disambiguation:
        return "only occurrence"
    word = ORDINAL_WORDS.get(order)
    if word is None:
        suffix = "th" if 10 <= order % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(order % 10, "th")
        word = f"{order}{suffix}"
    return f"{word} occurrence"


def active_roles(annotation: dict, event_spec: dict) -> list[str]:
    nonempty = {role for role, entries in annotation["annotations"].items() if entries}
    unknown = nonempty - event_spec["role_definitions"].keys()
    if unknown:
        raise RuntimeError(f"Stage 2 roles absent from prompt spec: {sorted(unknown)}")
    roles = [role for role in event_spec["predicate_order"] if role in nonempty]
    if set(roles) != nonempty or not roles:
        raise RuntimeError("Could not deterministically order active Stage 2 roles")
    return roles


def predicate_structure(event_type: str, roles: list[str], event_spec: dict) -> str:
    present = set(roles)
    if event_type in SHOT_EVENTS:
        pieces = []
        if "last_passer" in present:
            pieces.extend(["last_passer", "pass"])
        pieces.extend(["shooter", "shoot"])
        if "goalkeeper" in present:
            pieces.append("goalkeeper")
        return " — ".join(pieces)
    if event_type == "offside":
        pieces = []
        if "offside_passer" in present:
            pieces.extend(["offside_passer", "pass"])
        pieces.append("offside_player")
        if "second_last_defender" in present:
            pieces.extend(["offside", "second_last_defender"])
        return " — ".join(pieces)
    if len(roles) == 1:
        return f'{roles[0]} — {event_spec["predicate"]}'
    return f' — {event_spec["predicate"]} — '.join(roles)


def temporal_boundary(event_type: str, roles: list[str], event_spec: dict) -> str:
    if event_type in SHOT_EVENTS:
        suffix = "with_last_passer" if "last_passer" in roles else "without_last_passer"
        return event_spec[f"temporal_boundary_{suffix}"]
    return event_spec["temporal_boundary"]


def schema_example(setting: str, roles: list[str]) -> str:
    role_lines = ",\n".join(
        f'    "{role}": [xmin, ymin, xmax, ymax] or null' for role in roles
    )
    if setting == "oracle":
        return "[\n  {\n    \"time_sec\": number,\n" + role_lines + "\n  }\n]"
    role_lines = ",\n".join(
        f'      "{role}": [xmin, ymin, xmax, ymax] or null' for role in roles
    )
    return (
        "{\n  \"temporal_window\": {\n    \"start_sec\": number,\n"
        "    \"end_sec\": number\n  },\n  \"role_localization\": [\n    {\n"
        "      \"time_sec\": number,\n" + role_lines + "\n    }\n  ]\n}"
    )


def build_prompt(setting: str, event_type: str, roles: list[str], temporal: dict,
                 ordinal: dict | None, spec: dict,
                 oracle_timestamps: list[float] | None = None) -> str:
    event = spec["events"][event_type]
    if setting == "oracle":
        task = spec["setting_task_contexts"][setting].format(event_name=event["name"])
    else:
        assert ordinal is not None
        phrase = occurrence_phrase(ordinal["target_event_order"], ordinal["requires_disambiguation"])
        task = spec["setting_task_contexts"][setting].format(
            event_name=event["name"], occurrence_phrase=phrase)
    definitions = "\n".join(
        f'- `{role}`: {event["role_definitions"][role]}' for role in roles)
    boundary_section = ""
    if setting != "oracle":
        boundary_section = (
            "## EVENT-SPECIFIC TEMPORAL BOUNDARY\n\n"
            f"{temporal_boundary(event_type, roles, event)}\n\n"
        )
    rule_kind = "oracle" if setting == "oracle" else "context"
    rules = spec["rules"][rule_kind]
    if setting == "oracle":
        if not oracle_timestamps:
            raise RuntimeError("Oracle prompt requires its exact timestamp series")
        timestamp_series = "[" + ", ".join(
            f"{float(timestamp):.1f}" for timestamp in oracle_timestamps
        ) + "]"
        rules = rules.format(oracle_timestamps=timestamp_series)
    schema_rule = spec["strict_schema_rules"]["oracle" if setting == "oracle" else "context"]
    return (
        f"## SETTING-SPECIFIC TASK CONTEXT\n\n{task}\n\n"
        f"## EVENT DEFINITION\n\n**{event['name']}**: {event['definition']}\n\n"
        f"## ROLE DEFINITIONS\n\n{definitions}\n\n"
        f"{boundary_section}"
        f"## RULES\n\n{rules}\n\n"
        f"## STRICT JSON SCHEMA\n\n{schema_rule}\n\n"
        f"```json\n{schema_example(setting, roles)}\n```"
    )


def build_answer(setting: str, temporal: dict, annotation: dict,
                 roles: list[str]) -> list[dict] | dict:
    width, height = float(annotation["video_width"]), float(annotation["video_height"])
    by_role_frame = {
        role: {entry["frame_index"]: entry for entry in annotation["annotations"][role]}
        for role in roles
    }
    duration = float(temporal["end_in_window"]) - float(temporal["start_in_window"])
    localizations = []
    for index, frame in enumerate(annotation["keyframe_frames"]):
        local_time = index * 0.5
        if local_time > duration + 1e-6:
            continue
        item = {"time_sec": number(local_time if setting == "oracle"
                                    else float(temporal["start_in_window"]) + local_time)}
        for role in roles:
            entry = by_role_frame[role].get(frame)
            item[role] = (normalize_bbox(entry["bbox_xyxy"], width, height)
                          if entry is not None else None)
        localizations.append(item)
    if not localizations:
        raise RuntimeError("Answer has no localization timestamps")
    if setting == "oracle":
        return localizations
    return {
        "temporal_window": {
            "start_sec": number(temporal["start_in_window"]),
            "end_sec": number(temporal["end_in_window"]),
        },
        "role_localization": localizations,
    }

