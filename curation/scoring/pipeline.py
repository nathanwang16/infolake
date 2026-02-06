"""Scoring pipeline — orchestrates metrics, aggregation, and detection."""

import re
from typing import Dict, Any, List, Optional

from curation.scoring.protocols import ScoringMetric, ScoreAggregator, ContentTypeDetector
from curation.scoring.registry import MetricRegistry
from curation.scoring.aggregation import WeightedSigmoidAggregator, WilsonScoreComputer
from curation.scoring.detection import RuleBasedContentTypeDetector
from curation.scoring.metrics import BUILTIN_METRICS


class ScoringPipeline:
    """
    Orchestrates scoring: tokenizes once, runs all metrics, aggregates.

    Args:
        metrics: Optional list of metrics (defaults to BUILTIN_METRICS).
        aggregator: Optional aggregator (defaults to WeightedSigmoidAggregator).
        detector: Optional content-type detector (defaults to RuleBasedContentTypeDetector).
    """

    def __init__(
        self,
        metrics: Optional[List[ScoringMetric]] = None,
        aggregator: Optional[ScoreAggregator] = None,
        detector: Optional[ContentTypeDetector] = None,
    ):
        self.registry = MetricRegistry()
        for m in (metrics or BUILTIN_METRICS):
            self.registry.register(m)

        self.aggregator = aggregator or WeightedSigmoidAggregator()
        self.detector = detector or RuleBasedContentTypeDetector()
        self.wilson = WilsonScoreComputer()

        # Per-content-type weight overrides.
        # Keys are content-type strings; values are {metric_name: weight}.
        self._weight_overrides: Dict[str, Dict[str, float]] = {}

    # -- tokenization helpers (run once per document) -----------------------

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\b\w+\b', text)]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    # -- weight management --------------------------------------------------

    def get_weights(self, content_type: str = "default") -> Dict[str, float]:
        """Return effective weights for *content_type*."""
        return self._weight_overrides.get(
            content_type,
            self._weight_overrides.get("default", self.registry.default_weights),
        )

    def update_weights(self, content_type: str, weights: Dict[str, float]) -> None:
        self._weight_overrides[content_type] = weights

    # -- public API ---------------------------------------------------------

    def compute_raw_metrics(
        self, text: str, metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute all registered metrics for *text*."""
        names = self.registry.names
        if not text:
            return {n: 0.0 for n in names}

        words = self._tokenize_words(text)
        if not words:
            return {n: 0.0 for n in names}

        sentences = self._split_sentences(text)

        return {
            m.name: m.compute(text, words, sentences, metadata)
            for m in self.registry.metrics
        }

    def compute_score(
        self, metrics: Dict[str, float], content_type: str = "default"
    ) -> float:
        """Aggregate *metrics* into a single score using calibrated weights."""
        weights = self.get_weights(content_type)
        return self.aggregator.aggregate(metrics, weights)

    def compute_wilson_score(self, positive: int, total: int, z: float = 1.96) -> float:
        return self.wilson.compute(positive, total, z)

    def compute_document_wilson_score(self, metrics: Dict[str, float]) -> float:
        return self.wilson.compute_from_metrics(metrics)

    def detect_content_type(self, text: str, metadata: Dict[str, Any]) -> str:
        return self.detector.detect(text, metadata)
