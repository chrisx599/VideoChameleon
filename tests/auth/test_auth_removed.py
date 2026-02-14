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


def test_backend_auth_middleware_and_routes_removed():
    root = Path(__file__).resolve().parents[2]
    server = root / "univa" / "univa_server.py"
    auth_dir = root / "univa" / "auth"

    server_text = server.read_text(encoding="utf-8")

    assert "AuthMiddleware" not in server_text
    assert "/admin/access-codes" not in server_text
    assert "/access-code/status" not in server_text
    assert "accessCode" not in server_text
    assert not auth_dir.exists()


def test_web_auth_dependencies_removed():
    root = Path(__file__).resolve().parents[2]
    package_json = root / "apps" / "web" / "package.json"
    pkg = package_json.read_text(encoding="utf-8")

    assert "\"@opencut/auth\"" not in pkg
    assert "\"better-auth\"" not in pkg
    assert not (root / "packages" / "auth").exists()
    assert not (root / "apps" / "web" / "src" / "app" / "api" / "auth").exists()


def test_access_code_ui_and_routes_removed():
    root = Path(__file__).resolve().parents[2]

    assert not (root / "apps" / "web" / "src" / "app" / "admin" / "access-codes").exists()
    assert not (root / "apps" / "web" / "src" / "app" / "api" / "admin" / "access-codes").exists()
    assert not (root / "apps" / "web" / "src" / "app" / "api" / "access-code").exists()
    assert not (root / "apps" / "web" / "src" / "components" / "chat" / "AccessCodeSettings.tsx").exists()
    assert not (root / "apps" / "web" / "src" / "components" / "chat-settings.tsx").exists()

    use_chat = (root / "apps" / "web" / "src" / "components" / "chat" / "useChat.ts").read_text(encoding="utf-8")
    settings_view = (root / "apps" / "web" / "src" / "components" / "editor" / "media-panel" / "views" / "settings.tsx").read_text(encoding="utf-8")

    assert "accessCode" not in use_chat
    assert "Access code" not in settings_view


def test_privacy_and_terms_no_auth_references():
    root = Path(__file__).resolve().parents[2]
    privacy = (root / "apps" / "web" / "src" / "app" / "privacy" / "page.tsx").read_text(encoding="utf-8")
    terms = (root / "apps" / "web" / "src" / "app" / "terms" / "page.tsx").read_text(encoding="utf-8")

    assert "Better Auth" not in privacy
    assert "Google OAuth" not in privacy

    assert "create an account" not in terms
    assert "delete your account" not in terms
    assert "Keep your account" not in terms
    assert "suspend accounts" not in terms
