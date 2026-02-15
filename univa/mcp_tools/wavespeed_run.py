import os
from typing import Any, Dict, Optional

from univa.mcp_tools.base import ToolResponse, setup_logger
from univa.utils.wavespeed_universal import UniversalRunner

logger = setup_logger(__name__)


def _get_wavespeed_api_key() -> str:
    key = os.getenv("WAVESPEED_API_KEY") or ""
    if not key:
        raise RuntimeError("Missing WAVESPEED_API_KEY (set it in .env or environment).")
    return key


def wavespeed_run(
    model_id: str,
    params: Dict[str, Any],
    save_path: Optional[str] = None,
    timeout_sec: int = 300,
):
    api_key = _get_wavespeed_api_key()
    runner = UniversalRunner(api_key=api_key)
    data = runner.run(model_id=model_id, params=params, save_path=save_path, timeout_sec=timeout_sec)
    return ToolResponse(
        success=True,
        message="WaveSpeed task completed.",
        output_url=data.get("output_url"),
        output_path=data.get("output_path"),
        content=data,
    )
