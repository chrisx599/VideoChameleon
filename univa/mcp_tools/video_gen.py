import yaml
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from mcp_tools.base import ToolResponse, setup_logger
from utils.video_process import merge_videos, storyboard_generate, save_last_frame_decord
from utils.query_llm import refine_gen_prompt, audio_prompt_gen
from utils.image_process import download_image
from utils.wavespeed_api import text_to_video_generate, image_to_video_generate, frame_to_frame_video, text_to_image_generate, image_to_image_generate, audio_gen, hailuo_i2v_pro

# Load configuration
os.chdir(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/mcp_tools_config")
config_path = os.path.join(config_dir, "config.yaml")
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "config.example.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

video_gen_config = config.get('video_gen', {})
image_gen_config = config.get('image_gen', {})
# base_output_path = video_gen_config.get('base_output_path', '/share/project/liangzy/liangzy2/UniVideo/generated_videos')

def _get_wavespeed_api_key() -> str:
    """API keys must come from environment variables (.env)."""
    key = os.getenv("WAVESPEED_API_KEY") or ""
    if not key:
        raise RuntimeError("Missing WAVESPEED_API_KEY (set it in .env or environment).")
    return key

# Configure logging
logger = setup_logger(__name__, "logs/mcp_tools", "video_gen.log")
logger.info(f"Loaded video_gen_config: {video_gen_config}")

mcp = FastMCP("Video_Generation_Server")

try:
    from memory.service import ProjectMemoryService
except Exception:
    ProjectMemoryService = None


def _maybe_open_memory(project_id: Optional[str]):
    if not project_id or ProjectMemoryService is None:
        return None
    try:
        return ProjectMemoryService.open(project_id=project_id)
    except Exception as exc:
        logger.warning(f"Failed to open memory DB for project_id={project_id}: {exc}")
        return None


def _resolve_segment_id(svc, segment_id: str, t_start: Optional[float], t_end: Optional[float], kind: str, status: str) -> Optional[str]:
    if not svc:
        return None
    if segment_id:
        try:
            existing = svc.get_segment(segment_id=segment_id)
        except Exception:
            existing = None
        if existing:
            return segment_id
        logger.warning(f"segment_id not found in memory: {segment_id}")
    if t_start is None or t_end is None:
        return None
    try:
        return svc.upsert_segment(t_start=t_start, t_end=t_end, kind=kind, status=status)
    except Exception as exc:
        logger.warning(f"Failed to upsert segment: {exc}")
        return None


def _save_last_frame_artifact(svc, segment_id: str, clip_id: str, video_path: str) -> Optional[str]:
    if not svc:
        return None
    if not video_path:
        return None
    try:
        last_frame_path = save_last_frame_decord(video_path)
    except Exception as exc:
        logger.warning(f"Failed to extract last frame: {exc}")
        return None
    if not last_frame_path:
        return None
    try:
        svc.add_artifact(
            kind="last_frame",
            path=last_frame_path,
            segment_id=segment_id,
            clip_id=clip_id,
            meta={"source": "video_gen", "video_path": video_path},
        )
    except Exception as exc:
        logger.warning(f"Failed to save last frame artifact: {exc}")
    return last_frame_path

@mcp.tool()
async def text2video_gen(
    prompt: str,
    project_id: str = "",
    segment_id: str = "",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    kind: str = "target",
    status: str = "planned",
    save_last_frame: bool = True,
) -> str:
    """
    Generates a short video (approx. 5 seconds) from a text description.
    If segment_id is provided, it updates the corresponding segment in the server timeline.
    Otherwise, it generates a video without adding it to the timeline.

    Args:
        prompt (str): The prompt to generate the video.
        project_id (str): Optional project id to persist memory.
        segment_id (str): Optional segment id to attach the clip to.
        t_start (float): Optional segment start time (seconds).
        t_end (float): Optional segment end time (seconds).
        kind (str): Segment kind if created (default: target).
        status (str): Segment status if created (default: planned).
        save_last_frame (bool): Whether to save last frame into memory artifacts.
        save_path (str): The path to save the generated video. Suggest numbering each video when naming them.

    Returns:
        dict: A dictionary containing the success status, output video path, and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str, optional): The path to the generated video if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = video_gen_config.get("text_to_video")

    if model == "seedance":
        api_key = _get_wavespeed_api_key()
        save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        _time = datetime.now().strftime("%m%d%H%M%S")
        save_path = f"{save_dir}/{_time}.mp4"
        return_dict = text_to_video_generate(api_key, prompt, save_path=save_path)

        svc = _maybe_open_memory(project_id)
        try:
            if svc and return_dict.get("success"):
                seg_id = _resolve_segment_id(svc, segment_id, t_start, t_end, kind, status)
                if seg_id:
                    try:
                        clip = svc.save_clip_take(
                            segment_id=seg_id,
                            output_path=return_dict.get("output_path"),
                            prompt=prompt,
                            model=model or "",
                            params={"tool": "text2video_gen"},
                            make_active=True,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to save clip take: {exc}")
                        clip = None
                    if clip:
                        return_dict["segment_id"] = seg_id
                        return_dict["clip_id"] = clip.get("clip_id")
                        if save_last_frame:
                            last_frame_path = _save_last_frame_artifact(
                                svc,
                                segment_id=seg_id,
                                clip_id=clip.get("clip_id"),
                                video_path=return_dict.get("output_path"),
                            )
                            if last_frame_path:
                                return_dict["last_frame_path"] = last_frame_path
        finally:
            if svc:
                svc.close()

        return return_dict

    return {"success": False, "error": f"Unsupported text_to_video model: {model}"}


@mcp.tool()
async def image2video_gen(
    prompt: str,
    image_path: str,
    project_id: str = "",
    segment_id: str = "",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    kind: str = "target",
    status: str = "planned",
    save_last_frame: bool = True,
) -> str:
    """
    Generates a short video (approx. 5 seconds) using a text prompt and an input image as a visual reference.
    This tool is useful for creating videos that maintain visual consistency with a provided image while incorporating new elements described in the prompt.

    Args:
        prompt (str): The prompt to generate the video.
        image_path (str): Input image path for use as video content reference, supporting common formats (.jpg/.png/.bmp, etc.).
        project_id (str): Optional project id to persist memory.
        segment_id (str): Optional segment id to attach the clip to.
        t_start (float): Optional segment start time (seconds).
        t_end (float): Optional segment end time (seconds).
        kind (str): Segment kind if created (default: target).
        status (str): Segment status if created (default: planned).
        save_last_frame (bool): Whether to save last frame into memory artifacts.

    Returns:
        dict: A dictionary containing the success status, output video path, and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str, optional): The path to the generated video if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = video_gen_config.get("image_to_video")

    if model == "seedance":
        api_key = _get_wavespeed_api_key()
        save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        _time = datetime.now().strftime("%m%d%H%M%S")
        save_path = f"{save_dir}/{_time}.mp4"
        return_dict = image_to_video_generate(api_key, prompt, image_path, save_path=save_path)

        svc = _maybe_open_memory(project_id)
        try:
            if svc and return_dict.get("success"):
                seg_id = _resolve_segment_id(svc, segment_id, t_start, t_end, kind, status)
                if seg_id:
                    try:
                        clip = svc.save_clip_take(
                            segment_id=seg_id,
                            output_path=return_dict.get("output_path"),
                            prompt=prompt,
                            model=model or "",
                            params={"tool": "image2video_gen", "image_path": image_path},
                            make_active=True,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to save clip take: {exc}")
                        clip = None
                    if clip:
                        return_dict["segment_id"] = seg_id
                        return_dict["clip_id"] = clip.get("clip_id")
                        if save_last_frame:
                            last_frame_path = _save_last_frame_artifact(
                                svc,
                                segment_id=seg_id,
                                clip_id=clip.get("clip_id"),
                                video_path=return_dict.get("output_path"),
                            )
                            if last_frame_path:
                                return_dict["last_frame_path"] = last_frame_path
        finally:
            if svc:
                svc.close()

        return return_dict

    return {"success": False, "error": f"Unsupported image_to_video model: {model}"}



@mcp.tool()
async def frame2frame_video_gen(
    prompt: str,
    first_frame_path: str,
    last_frame_path: str,
    project_id: str = "",
    segment_id: str = "",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    kind: str = "target",
    status: str = "planned",
    save_last_frame: bool = True,
) -> str:
    """
    Generates a short video (approx. 5 seconds) that transitions between a specified first frame and a last frame, guided by a text prompt.
    This tool is effective for creating dynamic action sequences or smooth transitions between two distinct visual states.

    Args:
        prompt (str): The prompt to generate the video.
        first_frame_path (str): The path to the first frame.
        last_frame_path (str): The path to the last frame.
        project_id (str): Optional project id to persist memory.
        segment_id (str): Optional segment id to attach the clip to.
        t_start (float): Optional segment start time (seconds).
        t_end (float): Optional segment end time (seconds).
        kind (str): Segment kind if created (default: target).
        status (str): Segment status if created (default: planned).
        save_last_frame (bool): Whether to save last frame into memory artifacts.
        save_path (str): The path to save the generated video. Suggest numbering each video when naming them.

    Returns:
        dict: A dictionary containing the success status and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = video_gen_config.get("frame_to_frame_video")

    if model == "wan_api":
        api_key = _get_wavespeed_api_key()
        save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        _time = datetime.now().strftime("%m%d%H%M%S")
        save_path = f"{save_dir}/{_time}.mp4"
        return_dict = hailuo_i2v_pro(api_key, prompt, first_frame_path, last_frame_path, save_path=save_path)

        svc = _maybe_open_memory(project_id)
        try:
            if svc and return_dict.get("success"):
                seg_id = _resolve_segment_id(svc, segment_id, t_start, t_end, kind, status)
                if seg_id:
                    try:
                        clip = svc.save_clip_take(
                            segment_id=seg_id,
                            output_path=return_dict.get("output_path"),
                            prompt=prompt,
                            model=model or "",
                            params={
                                "tool": "frame2frame_video_gen",
                                "first_frame_path": first_frame_path,
                                "last_frame_path": last_frame_path,
                            },
                            make_active=True,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to save clip take: {exc}")
                        clip = None
                    if clip:
                        return_dict["segment_id"] = seg_id
                        return_dict["clip_id"] = clip.get("clip_id")
                        if save_last_frame:
                            last_frame_path_out = _save_last_frame_artifact(
                                svc,
                                segment_id=seg_id,
                                clip_id=clip.get("clip_id"),
                                video_path=return_dict.get("output_path"),
                            )
                            if last_frame_path_out:
                                return_dict["last_frame_path"] = last_frame_path_out
        finally:
            if svc:
                svc.close()

        return return_dict

    return {"success": False, "error": f"Unsupported frame_to_frame_video model: {model}"}


@mcp.tool()
async def merge2videos(video_paths: list[str]):
    """
    Merges multiple video files into a single video file.

    Args:
        video_paths (list[str]): A list of paths to the video files to merge, or a folder path containing videos.

    Returns:
        dict: A dictionary containing the success status and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str): The path to the merged video.
              - 'message' (str): A success message.
    """
    save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    _time = datetime.now().strftime("%m%d%H%M%S")
    save_path = f"{save_dir}/{_time}.mp4"
    video_path = merge_videos(video_paths, output_file=save_path)

    return ToolResponse(
        success=True,
        output_path=video_path,
        message="Videos merged successfully."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
