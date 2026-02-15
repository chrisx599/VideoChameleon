from unittest.mock import patch
import pytest

from univa.mcp_tools.wavespeed_run import wavespeed_run


def test_wavespeed_run_requires_api_key(monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        wavespeed_run("model-x", {"prompt": "hi"})


def test_wavespeed_run_returns_tool_response(monkeypatch):
    monkeypatch.setenv("WAVESPEED_API_KEY", "key")
    with patch("univa.mcp_tools.wavespeed_run.UniversalRunner") as runner:
        inst = runner.return_value
        inst.run.return_value = {"output_url": "https://cdn.example.com/out.mp4"}
        resp = wavespeed_run("model-x", {"prompt": "hi"})
        assert resp.success is True
        assert resp.output_url.endswith("out.mp4")
