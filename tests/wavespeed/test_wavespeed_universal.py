import time
from unittest.mock import patch, MagicMock
import pytest

from univa.utils.wavespeed_universal import WaveSpeedClient, SchemaAdapter, SchemaValidationError


def _mock_resp(data, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"data": data}
    resp.text = "ok"
    return resp


def test_list_models_caches_response():
    client = WaveSpeedClient("key", cache_ttl_sec=60)
    with patch("univa.utils.wavespeed_universal.requests.get") as get:
        get.return_value = _mock_resp([
            {"id": "m1", "type": "text-to-image", "api_schema": {"type": "object"}},
        ])
        first = client.list_models()
        second = client.list_models()
        assert first == second
        assert get.call_count == 1


def test_schema_validation_required_and_types():
    schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "num_images": {"type": "integer"},
        },
        "required": ["prompt"],
    }
    adapter = SchemaAdapter(schema)
    with pytest.raises(SchemaValidationError):
        adapter.validate({"num_images": 1})
    adapter.validate({"prompt": "hi", "num_images": 2})
