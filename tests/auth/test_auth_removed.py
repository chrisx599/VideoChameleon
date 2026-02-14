from pathlib import Path


def test_config_has_no_auth_keys():
    root = Path(__file__).resolve().parents[2]
    config_py = root / "univa" / "config" / "config.py"
    config_toml = root / "univa" / "config" / "config.toml"

    config_py_text = config_py.read_text(encoding="utf-8")
    config_toml_text = config_toml.read_text(encoding="utf-8")

    assert "auth_enabled" not in config_py_text
    assert "auth_config_file" not in config_py_text
    assert "admin_access_code" not in config_py_text

    assert "auth_enabled" not in config_toml_text
    assert "auth_config_file" not in config_toml_text
    assert "admin_access_code" not in config_toml_text
