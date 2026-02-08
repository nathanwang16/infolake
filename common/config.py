import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from common.logging.logger import get_logger

logger = get_logger("config")

# Centralized default values for all config keys used across the codebase.
# Each entry: (type, default_value)
# Types: str, int, float, bool, None (any)
CONFIG_SCHEMA: Dict[str, tuple] = {
    # Paths
    "paths.data_dir":               (str,   "data"),
    "paths.logs_dir":               (str,   "logs"),
    "paths.dumps_dir":              (str,   "./dataset"),
    "paths.archive_dir":            (str,   "data/archive"),
    "paths.parsed_archive_dir":     (str,   "data/parsed_archive"),
    "paths.golden_set_path":        (str,   "data/golden_set.db"),
    "paths.exports_dir":            (str,   "data/exports"),
    "paths.mappings_dir":           (str,   "data/mappings"),

    # Database
    "database.sqlite_path":         (str,   "data/atlas.db"),

    # Qdrant
    "qdrant.url":                   (str,   "http://localhost:6333"),
    "qdrant.collection":            (str,   "atlas_embeddings"),
    "qdrant.quantization":          (str,   "scalar"),
    "qdrant.on_disk":               (bool,  True),
    "qdrant.timeout_seconds":       (float, None),
    "qdrant.batch_size":            (int,   None),
    "qdrant.batch_timeout_seconds": (float, None),
    "qdrant.write_queue_size":      (int,   None),

    # Embedding
    "embedding.model":              (str,   "BAAI/bge-small-en-v1.5"),
    "embedding.device":             (str,   "cpu"),
    "embedding.batch_size":         (int,   32),
    "embedding.max_tokens":         (int,   512),

    # Batch processing
    "batch_processing.default_dump":            (str,   None),
    "batch_processing.workers":                 (int,   8),
    "batch_processing.worker_batch_size":       (int,   32),
    "batch_processing.quality_weight_alpha":    (float, 1.0),
    "batch_processing.novelty_threshold":       (float, 0.08),
    "batch_processing.quality_threshold":       (float, 0.3),
    "batch_processing.url_queue_size":          (int,   10000),
    "batch_processing.embed_queue_size":        (int,   5000),
    "batch_processing.lazy_greedy":             (bool,  True),
    "batch_processing.expected_documents":      (int,   1_000_000),
    "batch_processing.limit":                   (int,   0),
    "batch_processing.max_retries":             (int,   3),
    "batch_processing.fetch_concurrency":       (int,   100),
    "batch_processing.fetch_timeout":           (int,   15),
    "batch_processing.use_playwright":          (bool,  False),
    "batch_processing.playwright_concurrency":  (int,   4),
    "batch_processing.extract_processes":       (int,   8),
    "batch_processing.skip_scoring":            (bool,  True),
    "batch_processing.skip_url_filter":         (bool,  True),

    # Content extraction
    "content_extraction.extractor":             (str,   "trafilatura"),
    "content_extraction.fallback":              (str,   "readability"),
    "content_extraction.store_raw_html_hash":   (bool,  True),
    "content_extraction.min_length":            (int,   100),
    "content_extraction.max_length":            (int,   100000),

    # Language
    "language.target":              (str,   "en"),
    "language.use_langdetect":      (bool,  True),

    # Calibration
    "calibration.current_version":      (int,   1),
    "calibration.holdout_fraction":     (float, 0.2),
    "calibration.csv_path":             (str,   None),
    "calibration.default_content_type": (str,   "default"),

    # Deduplication
    "deduplication.simhash_threshold":              (int,   3),
    "deduplication.minhash_jaccard_threshold":       (float, 0.5),
    "deduplication.embedding_similarity_threshold":  (float, 0.95),

    # Mapping
    "mapping.input_source":         (str,   "qdrant"),
    "mapping.output_format":        (str,   "json"),
    "mapping.output_path":          (str,   "./data/mappings/latest.json"),
    "mapping.umap.neighbors":       (int,   15),
    "mapping.umap.min_dist":        (float, 0.1),
    "mapping.umap.metric":          (str,   "cosine"),
    "mapping.hdbscan.min_cluster_size": (int, 5),
    "mapping.sample_for_fit":       (int,   100000),

    # Visualizer
    "visualizer.host":              (str,   "localhost"),
    "visualizer.port":              (int,   8080),
    "visualizer.static_dir":        (str,   "./visualizer/static"),

    # Monitor
    "monitor.gini_alert_threshold":     (float, 0.6),
    "monitor.orphan_review_threshold":  (int,   100),
    "monitor.drift_alert_delta":        (float, 0.05),

    # Summarizer
    "summarizer.backend":           (str,   "auto"),
    "summarizer.ollama_model":      (str,   "llama3.2:1b"),
    "summarizer.max_length":        (int,   80),

    # LLM
    "llm.provider":                 (str,   "openai"),
    "llm.model":                    (str,   "gpt-4o-mini"),
    "llm.api_key":                  (str,   None),

    # Resource limits
    "resource_limits.max_rss_gb":              (float, None),
    "resource_limits.check_interval_seconds": (float, None),
}


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
            self._config = {}
            return

        with open(config_path, "r") as f:
            self._config = json.load(f)

        # Ensure directories exist (best-effort, don't fail on inaccessible paths)
        self._ensure_dirs()

    def _ensure_dirs(self):
        paths = self._config.get("paths", {})
        for path in paths.values():
            if isinstance(path, str) and not path.endswith(('db', 'json', 'txt')):
                try:
                    Path(path).mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass  # Skip inaccessible paths (e.g. unmounted volumes)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a config value by dot-separated key.

        Lookup order:
        1. Value from config.json (if present and not None)
        2. Caller-provided default (if not None)
        3. Schema default from CONFIG_SCHEMA
        4. None
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = None
                break
            if value is None:
                break

        # If found in config, return it
        if value is not None:
            return value

        # If caller provided an explicit default, use it
        if default is not None:
            return default

        # Fall back to schema default
        schema_entry = CONFIG_SCHEMA.get(key)
        if schema_entry is not None:
            return schema_entry[1]

        return None

    def validate(self) -> list:
        """
        Validates the loaded config against CONFIG_SCHEMA.

        Returns a list of warning strings for type mismatches.
        Does NOT raise -- config.json values always take precedence.
        """
        warnings = []
        for key, (expected_type, _default) in CONFIG_SCHEMA.items():
            if expected_type is None:
                continue
            value = self._get_raw(key)
            if value is not None and not isinstance(value, expected_type):
                warnings.append(
                    f"Config '{key}': expected {expected_type.__name__}, "
                    f"got {type(value).__name__} ({value!r})"
                )
        if warnings:
            for w in warnings:
                logger.warning(w)
        return warnings

    def _get_raw(self, key: str) -> Any:
        """Gets value from config.json without schema fallback."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
            if value is None:
                return None
        return value

    def require(self, key: str) -> Any:
        """
        Requires a config value to be explicitly set in config.json.

        Raises ValueError if missing.
        """
        value = self._get_raw(key)
        if value is None:
            logger.error(f"Missing required config key: {key}")
            raise ValueError(f"Missing required config key: {key}")
        return value


# Global accessor
config = Config()
