from pathlib import Path
from unittest.mock import patch

import univa.mcp_tools.image_gen as image_gen


def test_text2image_saves_under_repo_results_dir(monkeypatch):
    monkeypatch.setitem(image_gen.image_gen_config, "text_to_image", "flux-kontext")
    monkeypatch.setitem(image_gen.image_gen_config, "base_output_path", "results/image")
    monkeypatch.setenv("WAVESPEED_API_KEY", "key")

    with patch("univa.mcp_tools.image_gen.text_to_image_generate") as gen, \
         patch("univa.mcp_tools.image_gen.download_image") as download:
        gen.return_value = "https://example.com/img.jpg"
        resp = image_gen.text2image_generate("cat")
        args, kwargs = download.call_args
        save_path = kwargs.get("save_path") or args[1]

    repo_root = Path(__file__).resolve().parents[2]
    expected_dir = repo_root / "results" / "image"
    assert str(save_path).startswith(str(expected_dir))
    assert resp.output_path and str(resp.output_path).startswith(str(expected_dir))
