from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .service import ProjectMemoryService


def build_memory_context(
    project_id: str,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    pad_sec: float = 8.0,
    max_segments: int = 12,
) -> Dict[str, Any]:
    svc = ProjectMemoryService.open(project_id=project_id)
    try:
        if t_start is not None and t_end is not None:
            ctx = svc.get_context_window(t_start=t_start, t_end=t_end, pad_sec=pad_sec)
            segments = ctx.get("segments", [])
        else:
            segments = svc.list_segments()
            if max_segments > 0:
                segments = segments[-max_segments:]
            seg_ids = [s["segment_id"] for s in segments]
            active_clips = svc.store.get_active_clips_for_segments(seg_ids)
            beats = svc.store.get_beats_for_segments(seg_ids)
            center_t = None
            if segments:
                last = segments[-1]
                center_t = (float(last["t_start"]) + float(last["t_end"])) / 2.0
            entity_states = svc.get_entity_states_at(center_t) if center_t is not None else []
            ctx = {
                "project_id": svc.project_id,
                "window": {"t_start": None, "t_end": None, "pad_sec": float(pad_sec)},
                "focus": {"t_start": None, "t_end": None},
                "segments": segments,
                "active_clips_by_segment": active_clips,
                "beats": beats,
                "entity_states_at_center": entity_states,
            }

        # Attach last-frame artifact for each segment if available
        last_frames: Dict[str, Dict[str, Any]] = {}
        for seg in segments:
            seg_id = seg.get("segment_id")
            if not seg_id:
                continue
            art = svc.get_last_frame(segment_id=seg_id)
            if art:
                last_frames[seg_id] = art
        ctx["last_frames_by_segment"] = last_frames
        return ctx
    finally:
        svc.close()


def format_memory_context(ctx: Dict[str, Any]) -> str:
    project_id = ctx.get("project_id", "")
    window = ctx.get("window") or {}
    focus = ctx.get("focus") or {}
    segments = ctx.get("segments") or []
    active_clips = ctx.get("active_clips_by_segment") or {}
    beats = ctx.get("beats") or []
    entity_states = ctx.get("entity_states_at_center") or []
    last_frames = ctx.get("last_frames_by_segment") or {}

    lines: List[str] = []
    lines.append("MEMORY_CONTEXT")
    lines.append(f"project_id: {project_id}")
    lines.append(f"window: t_start={window.get('t_start')} t_end={window.get('t_end')} pad_sec={window.get('pad_sec')}")
    lines.append(f"focus: t_start={focus.get('t_start')} t_end={focus.get('t_end')}")

    lines.append("segments:")
    if not segments:
        lines.append("(none)")
    else:
        for seg in segments:
            seg_id = seg.get("segment_id")
            kind = seg.get("kind")
            status = seg.get("status")
            t_start = seg.get("t_start")
            t_end = seg.get("t_end")
            active = active_clips.get(seg_id)
            active_path = active.get("output_path") if active else None
            lines.append(
                f"- id={seg_id} t=[{t_start},{t_end}] kind={kind} status={status} active_clip={active.get('clip_id') if active else None} active_path={active_path}"
            )
            lf = last_frames.get(seg_id)
            if lf:
                lines.append(f"  last_frame_path={lf.get('path')}")

    if beats:
        lines.append("beats:")
        for b in beats:
            lines.append(f"- segment_id={b.get('segment_id')} type={b.get('beat_type')} summary={b.get('summary')}")

    if entity_states:
        lines.append("entity_states_at_center:")
        for es in entity_states:
            state = es.get("state") or {}
            lines.append(f"- entity={es.get('entity_name')} t=[{es.get('t_start')},{es.get('t_end')}] state={json.dumps(state, ensure_ascii=True)}")

    return "\n".join(lines)
