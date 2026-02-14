import os
import toml
import yaml
from dotenv import load_dotenv

def get_default_config():
    """Get default configuration"""
    # Get project root directory (assuming config.py is in univa/config/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
    return {
        # Session configuration
        "session_timeout_minutes": 60,
        "max_sessions_per_user": 10,
        
        # Agent configuration paths
        "mcp_servers_config": os.path.join(project_root, "univa/config/mcp_configs.json"),
        "prompt_dir": os.path.join(project_root, "univa/prompts"),
        "mcp_tools_path": os.path.join(project_root, "univa"),  # Default UniVideo path
        
        # Model configuration for Plan Agent
        "plan_model_provider": "openai",
        "plan_model_id": "gpt-5-2025-08-07",
        "plan_model_base_url": "",
        "plan_model_extra_params": "",
        
        # Model configuration for Act Agent
        "act_model_provider": "openai",
        "act_model_id": "gpt-5-2025-08-07",
        "act_model_base_url": "",
        "act_model_extra_params": "",
        
        # Other settings
        "proxy_host": "",
        "proxy_port": "",
        
        # MCP Configuration (Merged from YAML + environment variables)
        "mcp_config": {}
    }

def load_mcp_config(project_root: str) -> dict:
    """
    Load MCP tools configuration from YAML and override secrets with environment variables.

    Keys/secrets MUST come from environment variables (typically via .env):
    - WAVESPEED_API_KEY
    - LLM_OPENAI_API_KEY (or OPENAI_API_KEY)
    """
    config_dir = os.path.join(project_root, "univa/config/mcp_tools_config")
    config_path = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "config.example.yaml")
    
    mcp_config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                mcp_config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load MCP config from {config_path}: {e}")

    # Helper to safely set nested dict values
    def set_config(section, key, value):
        if value:
            if section not in mcp_config:
                mcp_config[section] = {}
            mcp_config[section][key] = value

    # --- Override secrets with Environment Variables ---
    wavespeed_key = os.getenv("WAVESPEED_API_KEY") or ""
    if wavespeed_key:
        for section in ["image_gen", "video_editing", "video_gen", "audio_gen"]:
            # Ensure the section exists if the key is provided.
            if section in mcp_config:
                set_config(section, "wavespeed_api", wavespeed_key)
            elif section == "video_gen":
                set_config(section, "wavespeed_api", wavespeed_key)

    llm_key = os.getenv("LLM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if llm_key:
        set_config("llm", "openai_api_key", llm_key)

    return mcp_config

def load_config():
    """Load configuration from TOML file or create with defaults if it doesn't exist"""
    app_name = "univa"
    # Main app config is TOML (separate from mcp_tools YAML config).
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CONFIG_FILE = os.path.join(project_root, "univa", "config", "config.toml")
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)

    # Create default config if file doesn't exist
    if not os.path.exists(CONFIG_FILE):
        config = get_default_config()
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            toml.dump(config, f)
    else:
        # read existing config file
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = toml.load(f)
            # Merge with defaults to ensure all required fields exist
            defaults = get_default_config()
            for key, value in defaults.items():
                if key not in config:
                    config[key] = value

    # Load .env (secrets only). We intentionally do NOT override model choices from .env.
    try:
        univa_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_candidates = [
            os.path.join(univa_root, ".env"),
            os.path.join(project_root, ".env"),
        ]
        env_path = next((p for p in env_candidates if os.path.exists(p)), None)
        if env_path:
            load_dotenv(dotenv_path=env_path, override=False)

        # Load API keys from environment variables (preferred) with a single fallback OPENAI_API_KEY.
        openai_key = os.getenv("OPENAI_API_KEY") or ""
        # Secrets come from environment only (do not read/write secrets to config.toml).
        config["plan_model_api_key"] = os.getenv("PLAN_MODEL_API_KEY") or openai_key or ""
        config["act_model_api_key"] = os.getenv("ACT_MODEL_API_KEY") or openai_key or ""
        config["react_model_api_key"] = os.getenv("REACT_MODEL_API_KEY") or openai_key or ""

        # --- Load and Merge MCP Config (YAML + env secrets) ---
        config["mcp_config"] = load_mcp_config(project_root)

    except Exception as e:
        print(f"Error loading configuration: {e}")
        pass

    # Set up proxy settings if configured
    PROXY_HOST = config.get("proxy_host")
    PROXY_PORT = config.get("proxy_port")
    if PROXY_HOST and PROXY_PORT:
        os.environ["http_proxy"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
        os.environ["https_proxy"] = f"http://{PROXY_HOST}:{PROXY_PORT}"

    return CONFIG_FILE, config

CONFIG_FILE, config = load_config()
