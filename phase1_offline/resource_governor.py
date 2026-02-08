"""
Resource governor utilities for Phase 1 pipeline.

Exits the process when memory cap is exceeded.
"""

import os
import signal
import threading
import time
from typing import Optional, Callable

from common.logging.logger import get_logger

logger = get_logger("resource_governor")


class MemoryGovernor(threading.Thread):
    """
    Monitors total RSS memory across the current process tree.

    When memory exceeds the configured limit, triggers a graceful shutdown.
    """

    def __init__(
        self,
        max_rss_gb: float,
        check_interval_seconds: float,
        on_limit: Optional[Callable[[], None]] = None,
    ):
        super().__init__(name="MemoryGovernor", daemon=True)

        if max_rss_gb is None:
            logger.error("max_rss_gb is required")
            raise ValueError("max_rss_gb is required")
        if check_interval_seconds is None:
            logger.error("check_interval_seconds is required")
            raise ValueError("check_interval_seconds is required")
        if max_rss_gb <= 0:
            logger.error(f"Invalid max_rss_gb: {max_rss_gb}")
            raise ValueError("max_rss_gb must be > 0")
        if check_interval_seconds <= 0:
            logger.error(f"Invalid check_interval_seconds: {check_interval_seconds}")
            raise ValueError("check_interval_seconds must be > 0")

        self.max_rss_bytes = int(max_rss_gb * 1024 ** 3)
        self.check_interval_seconds = check_interval_seconds
        self._on_limit = on_limit or self._default_on_limit

        self.running = True
        self._limit_triggered = False

    def run(self):
        """Main monitoring loop."""
        while self.running:
            try:
                total_rss = self._total_rss_bytes()
            except Exception as e:
                logger.error(f"MemoryGovernor failed to read RSS: {e}")
                time.sleep(self.check_interval_seconds)
                continue

            if total_rss >= self.max_rss_bytes and not self._limit_triggered:
                gb_used = total_rss / (1024 ** 3)
                logger.warning(
                    f"Memory cap reached: rss={gb_used:.2f}GB "
                    f"(limit={self.max_rss_bytes / (1024 ** 3):.2f}GB)"
                )
                self._limit_triggered = True
                self._on_limit()

            time.sleep(self.check_interval_seconds)

    def _total_rss_bytes(self) -> int:
        """Returns total RSS in bytes for current process tree."""
        try:
            import psutil
        except ImportError:
            logger.error("psutil is required for MemoryGovernor")
            raise RuntimeError("psutil dependency is required for memory monitoring")

        process = psutil.Process(os.getpid())
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                continue
        return total

    def stop(self):
        """Stops the governor thread."""
        self.running = False

    def _default_on_limit(self):
        """Default action: send SIGTERM to self for graceful shutdown."""
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            logger.error(f"Failed to signal shutdown on memory cap: {e}")
