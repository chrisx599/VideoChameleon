from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TimelineSegment:
    segment_id: str
    project_id: str
    t_start: float
    t_end: float
    kind: str
    status: str = "planned"
    active_clip_id: Optional[str] = None


@dataclass
class ClipTake:
    clip_id: str
    segment_id: str
    take_index: int
    output_path: str
    prompt: str = ""
    negative_prompt: str = ""
    model: str = ""
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = False


@dataclass
class StoryBeat:
    beat_id: str
    segment_id: str
    beat_type: str
    summary: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityState:
    state_id: str
    project_id: str
    entity_name: str
    t_start: float
    t_end: float
    state: Dict[str, Any] = field(default_factory=dict)
    source_clip_id: Optional[str] = None

