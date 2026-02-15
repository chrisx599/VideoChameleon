import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import requests


class SchemaValidationError(ValueError):
    pass


@dataclass
class ModelCache:
    data: list[dict]
    ts: float


class WaveSpeedClient:
    def __init__(self, api_key: str, base_url: str = "https://api.wavespeed.ai/api/v3", cache_ttl_sec: int = 180):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache_ttl_sec = cache_ttl_sec
        self._models: Optional[ModelCache] = None

    def list_models(self) -> list[dict]:
        if self._models and (time.time() - self._models.ts) < self.cache_ttl_sec:
            return self._models.data
        url = f"{self.base_url}/models"
        resp = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to list models: {resp.status_code} {resp.text}")
        data = resp.json().get("data", [])
        self._models = ModelCache(data=data, ts=time.time())
        return data

    def upload_media(self, path: str) -> str:
        url = f"{self.base_url}/media/upload/binary"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with open(path, "rb") as f:
            files = {"file": f}
            resp = requests.post(url, headers=headers, files=files)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return data.get("download_url") or data.get("url")
        url = f"{self.base_url}/upload"
        with open(path, "rb") as f:
            files = {"file": f}
            resp = requests.post(url, headers=headers, files=files)
        if resp.status_code != 200:
            raise RuntimeError(f"Upload failed: {resp.status_code} {resp.text}")
        data = resp.json().get("data", {})
        return data.get("url") or data.get("download_url")

    def run_model(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{model_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Submit failed: {resp.status_code} {resp.text}")
        data = resp.json().get("data", {})
        if isinstance(data.get("urls"), dict):
            data["result_url"] = data["urls"].get("get")
        return data

    def poll_result(
        self,
        task_id: str,
        result_url_hint: Optional[str] = None,
        timeout_sec: int = 300,
        poll_interval_sec: int = 2,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_sec
        url = result_url_hint or f"{self.base_url}/predictions/{task_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        while time.time() < deadline:
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Poll failed: {resp.status_code} {resp.text}")
            data = resp.json().get("data", {})
            status = data.get("status")
            if status == "completed":
                return data
            if status == "failed":
                raise RuntimeError(f"Task failed: {data.get('error')}")
            time.sleep(poll_interval_sec)
        raise TimeoutError(f"Timed out waiting for task {task_id}")


class SchemaAdapter:
    def __init__(self, schema: Optional[dict]):
        self.schema = schema or {}

    def validate(self, params: Dict[str, Any]) -> None:
        required = set(self.schema.get("required", []))
        props = self.schema.get("properties", {})
        missing = [k for k in required if k not in params]
        if missing:
            raise SchemaValidationError(f"Missing required fields: {', '.join(missing)}")
        for key, spec in props.items():
            if key not in params:
                continue
            expected = spec.get("type")
            if expected and not _matches_type(params[key], expected):
                raise SchemaValidationError(f"Field '{key}' should be {expected}")

    def prepare_params(self, params: Dict[str, Any], upload_media: Callable[[str], str]) -> Dict[str, Any]:
        self.validate(params)
        updated = dict(params)
        for key, value in list(updated.items()):
            if _is_local_path(value):
                updated[key] = upload_media(value)
            elif isinstance(value, list) and value and all(_is_local_path(v) for v in value):
                updated[key] = [upload_media(v) for v in value]
        return updated


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _is_local_path(value: Any) -> bool:
    return isinstance(value, str) and not value.startswith("http") and os.path.exists(value)
