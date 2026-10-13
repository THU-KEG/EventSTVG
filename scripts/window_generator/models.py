from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EventOccurrence:
    source_occurrence_id: str
    game: str
    half: int
    sequence_event_type: str
    soccernet_event_time_abs: float
    source_label: str
    source_position_ms: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetClip:
    sample_id: str
    game: str
    half: int
    source_clip: Path
    role_annotation_path: Path | None
    trim_annotation_path: Path
    source_half_video: Path
    filename_label: str
    event_type: str
    sequence_event_type: str
    event_position: float
    target_event_time: float
    clip_start: float
    clip_end: float
    occurrence: EventOccurrence
    source_timestamp_in_trim: bool
    timestamp_correction: dict[str, Any] | None = None


@dataclass(frozen=True)
class WindowResult:
    retained: bool
    reason: str | None
    start: float | None = None
    end: float | None = None
    contained_same_type_trims: tuple[TargetClip, ...] = field(default_factory=tuple)
    contained_same_type_occurrences: tuple[EventOccurrence, ...] = field(
        default_factory=tuple
    )
    target_order: int | None = None
    checks: dict[str, bool] = field(default_factory=dict)
