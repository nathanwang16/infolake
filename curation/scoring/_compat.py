"""Backward-compatible QualityScorer facade over ScoringPipeline."""

from typing import Dict, Any, List

from curation.scoring.pipeline import ScoringPipeline
from curation.scoring.metrics.reputation import SourceReputationMetric
from curation.scoring.metrics.writing import STOP_WORDS
from curation.scoring.metrics.methodology import METHODOLOGY_KEYWORDS


class QualityScorer:
    """
    Drop-in replacement for the original monolithic QualityScorer.

    All public attributes (METRICS, STOP_WORDS, METHODOLOGY_KEYWORDS, weights)
    and methods (compute_raw_metrics, compute_score, compute_wilson_score,
    compute_document_wilson_score, update_weights, _get_domain) are preserved.
    """

    STOP_WORDS = STOP_WORDS
    METHODOLOGY_KEYWORDS = METHODOLOGY_KEYWORDS

    def __init__(self, pipeline: ScoringPipeline = None):
        self._pipeline = pipeline or ScoringPipeline()
        # Seed the pipeline with the original default weights so that
        # existing callers see the same behaviour.
        self._pipeline.update_weights("default", {
            "citation_quality": 0.2,
            "writing_quality": 0.2,
            "content_depth": 0.2,
            "methodology_transparency": 0.1,
            "specificity": 0.1,
            "source_reputation": 0.1,
            "structural_integrity": 0.1,
        })

    # -- properties for backward compat ------------------------------------

    @property
    def METRICS(self) -> List[str]:
        return self._pipeline.registry.names

    @property
    def weights(self) -> Dict[str, Dict[str, float]]:
        """Expose weight overrides as a mutable dict (legacy API)."""
        return self._pipeline._weight_overrides

    @weights.setter
    def weights(self, value: Dict[str, Dict[str, float]]):
        self._pipeline._weight_overrides = value

    # -- delegated methods --------------------------------------------------

    def compute_raw_metrics(self, text: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        return self._pipeline.compute_raw_metrics(text, metadata)

    def compute_score(self, metrics: Dict[str, float], content_type: str = "default") -> float:
        return self._pipeline.compute_score(metrics, content_type)

    def compute_wilson_score(self, positive: int, total: int, z: float = 1.96) -> float:
        return self._pipeline.compute_wilson_score(positive, total, z)

    def compute_document_wilson_score(self, metrics: Dict[str, float]) -> float:
        return self._pipeline.compute_document_wilson_score(metrics)

    def update_weights(self, content_type: str, new_weights: Dict[str, float]):
        self._pipeline.update_weights(content_type, new_weights)

    @staticmethod
    def _get_domain(url: str) -> str:
        return SourceReputationMetric._get_domain(url)
