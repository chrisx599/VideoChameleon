import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
