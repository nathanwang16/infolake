"""Tests for common/config.py — schema defaults and validation."""

import pytest

from common.config import Config, CONFIG_SCHEMA


class TestConfigSchemaDefaults:
    def test_schema_has_database_path(self):
        assert "database.sqlite_path" in CONFIG_SCHEMA
        _type, default = CONFIG_SCHEMA["database.sqlite_path"]
        assert default == "data/atlas.db"

    def test_schema_has_workers(self):
        _type, default = CONFIG_SCHEMA["batch_processing.workers"]
        assert default == 8
        assert _type is int

    def test_schema_has_queue_sizes(self):
        _type, default = CONFIG_SCHEMA["batch_processing.url_queue_size"]
        assert default == 10000

        _type, default = CONFIG_SCHEMA["batch_processing.embed_queue_size"]
        assert default == 5000

    def test_schema_types_are_valid(self):
        valid_types = {str, int, float, bool, None}
        for key, (t, _default) in CONFIG_SCHEMA.items():
            assert t in valid_types or t is None, f"Invalid type for {key}: {t}"


class TestConfigGet:
    def test_get_returns_schema_default_when_no_config(self):
        """When config.json doesn't have the key and no caller default, use schema."""
        # Create a config with empty dict
        cfg = Config.__new__(Config)
        cfg._config = {}

        assert cfg.get("database.sqlite_path") == "data/atlas.db"
        assert cfg.get("batch_processing.workers") == 8

    def test_get_caller_default_overrides_schema(self):
        cfg = Config.__new__(Config)
        cfg._config = {}
        # Caller-provided default takes precedence over schema
        assert cfg.get("database.sqlite_path", "/custom/path.db") == "/custom/path.db"

    def test_get_config_value_overrides_all(self):
        cfg = Config.__new__(Config)
        cfg._config = {"database": {"sqlite_path": "/from/config.db"}}
        assert cfg.get("database.sqlite_path", "/caller/default") == "/from/config.db"

    def test_get_unknown_key_returns_none(self):
        cfg = Config.__new__(Config)
        cfg._config = {}
        assert cfg.get("nonexistent.key") is None

    def test_get_unknown_key_returns_caller_default(self):
        cfg = Config.__new__(Config)
        cfg._config = {}
        assert cfg.get("nonexistent.key", "fallback") == "fallback"


class TestConfigValidate:
    def test_validate_clean_config(self):
        cfg = Config.__new__(Config)
        cfg._config = {"database": {"sqlite_path": "data/test.db"}}
        warnings = cfg.validate()
        assert len(warnings) == 0

    def test_validate_type_mismatch(self):
        cfg = Config.__new__(Config)
        cfg._config = {"batch_processing": {"workers": "not_a_number"}}
        warnings = cfg.validate()
        assert any("workers" in w for w in warnings)
