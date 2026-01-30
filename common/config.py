import json
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = Path("config.json")
        if not config_path.exists():
            # Fallback to default if config.json not found
            # In a real scenario, might want to raise error or load from default dict
            self._config = {}
            return

        with open(config_path, "r") as f:
            self._config = json.load(f)
            
        # Ensure directories exist
        self._ensure_dirs()

    def _ensure_dirs(self):
        paths = self._config.get("paths", {})
        for path in paths.values():
            if isinstance(path, str) and not path.endswith(('db', 'json', 'txt')):
                Path(path).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

# Global accessor
config = Config()
