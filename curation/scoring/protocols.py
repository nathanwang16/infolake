"""Abstract base classes for the scoring system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ScoringMetric(ABC):
    """Protocol for a single scoring metric."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric name (e.g. 'citation_quality')."""
        ...

    @property
    def default_weight(self) -> float:
        """Default weight when no calibration data exists."""
        return 0.1

    @abstractmethod
    def compute(
        self,
        text: str,
        words: List[str],
        sentences: List[str],
        metadata: Dict[str, Any],
    ) -> float:
        """
        Compute metric value for a document.

        Args:
            text: Raw document text.
            words: Pre-tokenized lowercase words.
            sentences: Pre-split sentences (stripped, non-empty).
            metadata: Document metadata dict (url, domain, title, ...).

        Returns:
            Score in [0.0, 1.0].
        """
        ...


class ScoreAggregator(ABC):
    """Protocol for combining per-metric scores into a final score."""

    @abstractmethod
    def aggregate(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """Return a single aggregate score in [0.0, 1.0]."""
        ...


class ContentTypeDetector(ABC):
    """Protocol for detecting document content type."""

    @abstractmethod
    def detect(self, text: str, metadata: Dict[str, Any]) -> str:
        """Return a content-type string (e.g. 'academic', 'news')."""
        ...
