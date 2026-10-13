from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .models import EventOccurrence


NORMALIZED_EVENT_TYPES = {
    ("clearance", None): "goal_kick",
    ("corner", "w1"): "corner_won",
    ("corner", "w2"): "corner_kick",
    ("direct_free-kick", "w1"): "foul_direct_free_kick",
    ("direct_free-kick", "w2"): "direct_free_kick",
    ("goal", None): "goal",
    ("kick-off", None): "kick_off",
    ("offside", "w1"): "offside",
    ("offside", "w2"): "indirect_free_kick",
    ("penalty", "w1"): "foul_penalty",
    ("penalty", "w2"): "penalty_kick",
    ("red_card", None): "foul_red_card",
    ("shots_off_target", None): "shot_off_target",
    ("shots_on_target", None): "shot_on_target",
    ("throw-in", None): "throw_in",
    ("yellow_card", None): "foul_yellow_card",
    ("yellowtored_card", None): "foul_second_yellow_red_card",
}

SEQUENCE_EVENT_TYPES = {
    "goal_kick": "Clearance",
    "corner_won": "Corner",
    "corner_kick": "Corner",
    "foul_direct_free_kick": "Direct free-kick",
    "direct_free_kick": "Direct free-kick",
    "goal": "Goal",
    "kick_off": "Kick-off",
    "offside": "Offside",
    "indirect_free_kick": "Offside",
    "foul_penalty": "Penalty",
    "penalty_kick": "Penalty",
    "foul_red_card": "Red card",
    "foul_yellow_card": "Yellow card",
    "foul_second_yellow_red_card": "Yellow->red card",
    "shot_off_target": "Shots off target",
    "shot_on_target": "Shots on target",
    "throw_in": "Throw-in",
}

STEM_RE = re.compile(
    r"^(?P<label>.+?)_H(?P<half>[12])_(?P<position>\d+)(?:_(?P<window>w[12]))?$"
)


def normalized_event_type(filename_label: str, window: str | None) -> str:
    key = (filename_label.lower(), window)
    try:
        return NORMALIZED_EVENT_TYPES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported depicted event mapping: {key}") from exc


def sequence_event_type(target_event_type: str) -> str:
    return SEQUENCE_EVENT_TYPES[target_event_type]


def _source_id(game: str, half: int, sequence_type: str, position_ms: int) -> str:
    value = f"{game}|{half}|{sequence_type}|{position_ms}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def occurrences_from_labels(labels_path: Path, game: str) -> list[EventOccurrence]:
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    relevant = set(SEQUENCE_EVENT_TYPES.values())
    unique: dict[tuple[int, str, int], EventOccurrence] = {}
    for annotation in raw.get("annotations", []):
        label = str(annotation.get("label"))
        if label not in relevant:
            continue
        half = int(str(annotation["gameTime"]).split(" - ")[0])
        position_ms = int(annotation["position"])
        key = (half, label, position_ms)
        unique[key] = EventOccurrence(
            source_occurrence_id=_source_id(game, half, label, position_ms),
            game=game,
            half=half,
            sequence_event_type=label,
            soccernet_event_time_abs=position_ms / 1000.0,
            source_label=label,
            source_position_ms=position_ms,
        )
    return sorted(
        unique.values(),
        key=lambda x: (x.half, x.sequence_event_type,
                       x.soccernet_event_time_abs),
    )


class EventIndex:
    def __init__(self, occurrences: list[EventOccurrence]):
        unique = {
            (item.game, item.half, item.sequence_event_type,
             item.source_position_ms): item
            for item in occurrences
        }
        self.occurrences = sorted(
            unique.values(),
            key=lambda x: (
                x.game, x.half, x.sequence_event_type,
                x.soccernet_event_time_abs,
            ),
        )
        by_type: dict[tuple[str, int, str], list[EventOccurrence]] = defaultdict(list)
        by_position: dict[
            tuple[str, int, str, int], list[EventOccurrence]
        ] = defaultdict(list)
        for occurrence in self.occurrences:
            by_type[(occurrence.game, occurrence.half,
                     occurrence.sequence_event_type)].append(occurrence)
            by_position[(occurrence.game, occurrence.half,
                         occurrence.sequence_event_type,
                         occurrence.source_position_ms)].append(occurrence)
        self.by_type = dict(by_type)
        self.by_position = dict(by_position)

    def sequence(
        self, game: str, half: int, sequence_type: str
    ) -> list[EventOccurrence]:
        return self.by_type.get((game, half, sequence_type), [])

    def match(
        self, game: str, half: int, sequence_type: str, position_ms: int,
        tolerance_seconds: float,
    ) -> list[EventOccurrence]:
        exact = self.by_position.get((game, half, sequence_type, position_ms), [])
        if exact:
            return exact
        tolerance = tolerance_seconds
        return [
            item for item in self.sequence(game, half, sequence_type)
            if abs(item.soccernet_event_time_abs - position_ms / 1000.0)
            <= tolerance
        ]

    def counts_by_game_half_type(self) -> list[dict]:
        return [
            {"game": game, "half": half, "sequence_event_type": event_type,
             "count": len(records)}
            for (game, half, event_type), records in sorted(self.by_type.items())
        ]

    def type_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(
            item.sequence_event_type for item in self.occurrences
        ).items()))

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for occurrence in self.occurrences:
                handle.write(json.dumps(occurrence.as_dict(), sort_keys=True) + "\n")
