"""
Structured JSON logger for atlas_core.

Per-module JSONL logs to the configured logs directory.
Console: INFO+, File: DEBUG+, Rotating 10 MB / 5 backups.
"""

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Outputs each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
_BACKUP_COUNT = 5
_CONSOLE_FMT = "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s"

# Cache resolved log directory so we mkdir only once
_log_dir: Optional[Path] = None


def _resolve_log_dir() -> Path:
    """Resolve the logs directory from config.json, with a safe fallback."""
    global _log_dir
    if _log_dir is not None:
        return _log_dir

    # Attempt to read config.json for paths.logs_dir without importing
    # atlas_core.config (which itself uses this logger — avoid circular import).
    log_path = Path("logs")
    try:
        config_file = Path("config.json")
        if config_file.exists():
            import json as _json
            with open(config_file, "r") as fh:
                cfg = _json.load(fh)
            configured = cfg.get("paths", {}).get("logs_dir")
            if configured:
                log_path = Path(configured)
    except Exception:
        pass  # fall back to ./logs

    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_path = Path("logs")
        log_path.mkdir(parents=True, exist_ok=True)

    _log_dir = log_path
    return _log_dir


def get_logger(name: str, *, console: bool = True) -> logging.Logger:
    """
    Returns a logger configured with JSON file + optional console handlers.

    Idempotent: repeated calls with the same *name* return the same logger
    without adding duplicate handlers.

    Args:
        name: Logger name (typically module __name__ or a short label).
        console: Whether to attach a console (stdout) handler.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # --- File handler (DEBUG+) ---
    log_dir = _resolve_log_dir()
    file_handler = RotatingFileHandler(
        log_dir / f"{name}.jsonl",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    # --- Console handler (INFO+) ---
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(_CONSOLE_FMT))
        logger.addHandler(console_handler)

    return logger
