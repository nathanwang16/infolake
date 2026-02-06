"""Methodology transparency metric."""

from typing import Dict, Any, List

from curation.scoring.protocols import ScoringMetric

METHODOLOGY_KEYWORDS = {
    "methodology", "methods", "experiment", "analysis", "data", "results",
    "conclusion", "study", "survey", "interview", "observation", "algorithm",
    "model", "framework", "approach", "validation", "metrics",
}


class MethodologyTransparencyMetric(ScoringMetric):
    """Scores methodology transparency via keyword density."""

    @property
    def name(self) -> str:
        return "methodology_transparency"

    @property
    def default_weight(self) -> float:
        return 0.1

    def compute(
        self,
        text: str,
        words: List[str],
        sentences: List[str],
        metadata: Dict[str, Any],
    ) -> float:
        total_words = len(words)
        method_hits = sum(1 for w in words if w in METHODOLOGY_KEYWORDS)
        return min(method_hits / (total_words * 0.01 + 1), 1.0)
