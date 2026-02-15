import time
from unittest.mock import patch, MagicMock
import pytest

from univa.utils.wavespeed_universal import (
    WaveSpeedClient,
    SchemaAdapter,
    SchemaValidationError,
    UniversalRunner,
)


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


def test_prepare_params_uploads_local_file(tmp_path):
    local = tmp_path / "img.jpg"
    local.write_bytes(b"x")

    schema = {
        "type": "object",
        "properties": {"image": {"type": "string"}},
        "required": ["image"],
    }

    with patch("univa.utils.wavespeed_universal.WaveSpeedClient.upload_media") as upload:
        upload.return_value = "https://cdn.example.com/img.jpg"
        adapter = SchemaAdapter(schema)
        params = {"image": str(local)}
        prepared = adapter.prepare_params(params, upload_media=upload)
        assert prepared["image"].startswith("https://")
        upload.assert_called_once()


def test_run_and_poll_result_uses_result_url():
    client = WaveSpeedClient("key")

    with patch("univa.utils.wavespeed_universal.requests.post") as post, \
         patch("univa.utils.wavespeed_universal.requests.get") as get:
        post.return_value = _mock_resp({"id": "task1", "urls": {"get": "https://api.wavespeed.ai/api/v3/predictions/task1/result"}})
        get.return_value = _mock_resp({"status": "completed", "outputs": ["https://cdn.example.com/out.mp4"]})

        data = client.run_model("model-x", {"prompt": "hi"})
        result = client.poll_result(data["id"], result_url_hint=data.get("result_url"), timeout_sec=1, poll_interval_sec=0)
        assert result["outputs"][0].endswith("out.mp4")


def test_run_and_poll_result_fallbacks_to_prediction_url():
    client = WaveSpeedClient("key")

    with patch("univa.utils.wavespeed_universal.requests.post") as post, \
         patch("univa.utils.wavespeed_universal.requests.get") as get:
        post.return_value = _mock_resp({"id": "task2"})
        get.return_value = _mock_resp({"status": "completed", "outputs": ["https://cdn.example.com/out.png"]})

        data = client.run_model("model-y", {"prompt": "hi"})
        result = client.poll_result(data["id"], result_url_hint=None, timeout_sec=1, poll_interval_sec=0)
        assert result["outputs"][0].endswith("out.png")


def test_download_output_to_save_path(tmp_path):
    runner = UniversalRunner(api_key="key")
    url = "https://cdn.example.com/out.mp4"

    with patch("univa.utils.wavespeed_universal.requests.get") as get:
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_content.return_value = [b"data"]
        get.return_value = resp
        out = runner._download_output(url, save_path=str(tmp_path / "out.mp4"))
        assert out.endswith("out.mp4")


def test_runner_returns_output_path_when_save_path_provided():
    runner = UniversalRunner(api_key="key")

    with patch.object(runner.client, "list_models") as list_models, \
         patch.object(runner.client, "run_model") as run_model, \
         patch.object(runner.client, "poll_result") as poll_result, \
         patch.object(runner, "_download_output") as download:
        list_models.return_value = [{
            "id": "model-1",
            "api_schema": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
        }]
        run_model.return_value = {"id": "task-1"}
        poll_result.return_value = {"status": "completed", "outputs": ["https://cdn.example.com/out.mp4"]}
        download.return_value = "saved.mp4"

        result = runner.run("model-1", {"prompt": "hi"}, save_path="saved.mp4", timeout_sec=1)
        assert result["output_path"] == "saved.mp4"
        download.assert_called_once_with("https://cdn.example.com/out.mp4", save_path="saved.mp4")
