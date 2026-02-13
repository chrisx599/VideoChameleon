from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .store import ProjectMemoryStore, default_project_db_path


@dataclass
class ProjectMemoryService:
    """
    High-level timeline-centric memory API.

    This service hides DB details and provides a stable interface for agent tools.
    """

    project_id: str
    store: ProjectMemoryStore

    @classmethod
    def open(cls, project_id: str, db_path: Optional[Path] = None) -> "ProjectMemoryService":
        store = ProjectMemoryStore.open(project_id=project_id, db_path=db_path)
        return cls(project_id=store.project_id, store=store)

    @property
    def db_path(self) -> Path:
        return self.store.db_path

    def close(self) -> None:
        self.store.close()

    def upsert_segment(self, t_start: float, t_end: float, kind: str, status: str = "planned") -> str:
        if t_end <= t_start:
            raise ValueError(f"t_end must be greater than t_start, got {t_start} -> {t_end}")
        return self.store.upsert_segment(t_start=t_start, t_end=t_end, kind=kind, status=status)

    def get_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_segment(segment_id=segment_id)

    def list_segments(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        kind: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.store.list_segments(t_start=t_start, t_end=t_end, kind=kind, status=status)

    def update_segment_status(self, segment_id: str, status: str) -> bool:
        return self.store.update_segment_status(segment_id=segment_id, status=status)

    def delete_segment(self, segment_id: str) -> bool:
        return self.store.delete_segment(segment_id=segment_id)

    def save_clip_take(
        self,
        segment_id: str,
        output_path: str,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        make_active: bool = True,
    ) -> Dict[str, Any]:
        return self.store.add_clip_take(
            segment_id=segment_id,
            output_path=output_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            seed=seed,
            params=params,
            make_active=make_active,
        )

    def get_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_clip(clip_id=clip_id)

    def list_clips_for_segment(self, segment_id: str) -> List[Dict[str, Any]]:
        return self.store.list_clips_for_segment(segment_id=segment_id)

    def get_clips_for_segments(self, segment_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        return self.store.get_clips_for_segments(segment_ids=segment_ids)

    def delete_clip(self, clip_id: str) -> bool:
        return self.store.delete_clip(clip_id=clip_id)

    def set_active_take(self, segment_id: str, clip_id: str) -> None:
        self.store.set_active_clip(segment_id=segment_id, clip_id=clip_id)

    def add_beat(self, segment_id: str, beat_type: str, summary: str = "", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.store.add_beat(segment_id=segment_id, beat_type=beat_type, summary=summary, payload=payload)

    def delete_beat(self, beat_id: str) -> bool:
        return self.store.delete_beat(beat_id=beat_id)

    def add_entity_state(
        self,
        entity_name: str,
        t_start: float,
        t_end: float,
        state: Dict[str, Any],
        source_clip_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if t_end <= t_start:
            raise ValueError(f"t_end must be greater than t_start, got {t_start} -> {t_end}")
        return self.store.add_entity_state(
            entity_name=entity_name,
            t_start=t_start,
            t_end=t_end,
            state=state,
            source_clip_id=source_clip_id,
        )

    def list_entity_states(
        self,
        entity_name: Optional[str] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return self.store.list_entity_states(entity_name=entity_name, t_start=t_start, t_end=t_end)

    def delete_entity_state(self, state_id: str) -> bool:
        return self.store.delete_entity_state(state_id=state_id)

    def add_eval(
        self,
        clip_id: str,
        consistency_score: Optional[float] = None,
        story_match_score: Optional[float] = None,
        visual_score: Optional[float] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        return self.store.add_eval(
            clip_id=clip_id,
            consistency_score=consistency_score,
            story_match_score=story_match_score,
            visual_score=visual_score,
            note=note,
        )

    def list_evals(self, clip_id: str) -> List[Dict[str, Any]]:
        return self.store.list_evals(clip_id=clip_id)

    def add_artifact(
        self,
        kind: str,
        path: str,
        segment_id: Optional[str] = None,
        clip_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.store.add_artifact(kind=kind, path=path, segment_id=segment_id, clip_id=clip_id, meta=meta)

    def list_artifacts(
        self,
        segment_id: Optional[str] = None,
        clip_id: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.store.list_artifacts(segment_id=segment_id, clip_id=clip_id, kind=kind)

    def delete_artifact(self, artifact_id: str) -> bool:
        return self.store.delete_artifact(artifact_id=artifact_id)

    def get_latest_artifact(
        self,
        segment_id: Optional[str] = None,
        clip_id: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.store.get_latest_artifact(segment_id=segment_id, clip_id=clip_id, kind=kind)

    def get_last_frame(
        self,
        segment_id: Optional[str] = None,
        clip_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the most recent last-frame artifact for a clip or segment.
        If segment_id is provided and clip_id is not, prefer the active clip.
        """
        if clip_id:
            return self.store.get_latest_artifact(clip_id=clip_id, kind="last_frame")
        if segment_id:
            active = self.store.get_active_clips_for_segments([segment_id]).get(segment_id)
            if active:
                art = self.store.get_latest_artifact(clip_id=active["clip_id"], kind="last_frame")
                if art:
                    return art
            return self.store.get_latest_artifact(segment_id=segment_id, kind="last_frame")
        return None

    def get_entity_states_at(self, t: float) -> List[Dict[str, Any]]:
        return self.store.get_entity_states_at(t=t)

    def get_context_window(self, t_start: float, t_end: float, pad_sec: float = 8.0) -> Dict[str, Any]:
        """
        Return timeline context around [t_start, t_end].
        Includes:
        - segments in expanded window
        - active clip for each segment (if exists)
        - beats for those segments
        - entity states at center time
        """
        left = float(t_start) - float(pad_sec)
        right = float(t_end) + float(pad_sec)

        segments = self.store.get_segments_in_window(left, right)
        seg_ids = [s["segment_id"] for s in segments]
        active_clips = self.store.get_active_clips_for_segments(seg_ids)
        beats = self.store.get_beats_for_segments(seg_ids)
        center_t = (float(t_start) + float(t_end)) / 2.0
        entity_states = self.store.get_entity_states_at(center_t)

        return {
            "project_id": self.project_id,
            "window": {"t_start": left, "t_end": right, "pad_sec": float(pad_sec)},
            "focus": {"t_start": float(t_start), "t_end": float(t_end)},
            "segments": segments,
            "active_clips_by_segment": active_clips,
            "beats": beats,
            "entity_states_at_center": entity_states,
        }


def init_project_memory(project_id: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience initializer for external callers/tools.
    """
    db_path = db_path or default_project_db_path(project_id)
    svc = ProjectMemoryService.open(project_id=project_id, db_path=db_path)
    canonical = svc.project_id
    svc.close()
    return {"project_id": canonical, "db_path": str(db_path)}
