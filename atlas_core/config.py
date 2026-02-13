"""
Singleton configuration loader for atlas_core.

Reads config.json once and provides dot-notation access.  All required
parameters must be present — missing values raise AtlasConfigError.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from atlas_core.errors import AtlasConfigError
from atlas_core.logging import get_logger

logger = get_logger("atlas_core.config")


class Config:
    """Thread-safe singleton configuration backed by config.json."""

    _instance: Optional["Config"] = None
    _data: Dict[str, Any]

    def __new__(cls) -> "Config":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._data = {}
            inst._load()
            cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        config_path = Path("config.json")
        if not config_path.exists():
            logger.warning("config.json not found — configuration will be empty")
            self._data = {}
            return

        with open(config_path, "r") as fh:
            self._data = json.load(fh)

        logger.info("Loaded configuration from config.json")
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Best-effort directory creation for paths that don't look like files."""
        paths = self._data.get("paths", {})
        for value in paths.values():
            if isinstance(value, str) and not value.endswith((".db", ".json", ".txt", ".csv")):
                try:
                    Path(value).mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value by dot-separated key (e.g. ``"qdrant.url"``).

        Lookup order:
        1. Value from config.json (if present and not None).
        2. Caller-supplied *default*.

        If both are None, returns None.  Use :meth:`require` when a value
        **must** exist.
        """
        value = self._traverse(key)
        if value is not None:
            return value
        return default

    def require(self, key: str) -> Any:
        """
        Like :meth:`get` but raises :class:`AtlasConfigError` when the value
        is missing or None.
        """
        value = self._traverse(key)
        if value is None:
            logger.error("Missing required config key: %s", key)
            raise AtlasConfigError(key)
        return value

    def section(self, key: str) -> Dict[str, Any]:
        """
        Return an entire config section as a dict.

        Raises AtlasConfigError if the section does not exist or is not a dict.
        """
        value = self._traverse(key)
        if not isinstance(value, dict):
            raise AtlasConfigError(key, reason="section not found or not a dict")
        return value

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _traverse(self, key: str) -> Any:
        parts = key.split(".")
        node: Any = self._data
        for part in parts:
            if isinstance(node, dict):
                node = node.get(part)
            else:
                return None
            if node is None:
                return None
        return node

    def __repr__(self) -> str:
        return f"<Config keys={list(self._data.keys())}>"


# Module-level singleton
config = Config()
