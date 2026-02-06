"""Specificity metric."""

from typing import Dict, Any, List

from curation.scoring.protocols import ScoringMetric


class SpecificityMetric(ScoringMetric):
    """Scores information density via ratio of long words."""

    @property
    def name(self) -> str:
        return "specificity"

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
        long_words = sum(1 for w in words if len(w) > 6)
        return min(long_words / total_words * 3, 1.0)
