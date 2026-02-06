"""Score aggregation strategies."""

import math
from typing import Dict

from curation.scoring.protocols import ScoreAggregator


class WeightedSigmoidAggregator(ScoreAggregator):
    """Weighted average with sigmoid smoothing (k=10, center=0.5)."""

    def __init__(self, k: float = 10, center: float = 0.5):
        self.k = k
        self.center = center

    def aggregate(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        score = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            if metric in weights:
                weight = weights[metric]
                score += value * weight
                total_weight += weight

        if total_weight > 0:
            raw_score = score / total_weight
            try:
                return 1 / (1 + math.exp(-self.k * (raw_score - self.center)))
            except OverflowError:
                return 0.0 if raw_score < self.center else 1.0
        return 0.0


class WilsonScoreComputer:
    """Computes Wilson Score Lower Bound for ranking hidden gems."""

    def compute(self, positive: int, total: int, z: float = 1.96) -> float:
        """
        Compute Wilson Score Lower Bound.

        Args:
            positive: Number of positive quality signals.
            total: Total number of applicable quality signals.
            z: Z-score for confidence level (1.96 = 95%).

        Returns:
            Lower bound of the confidence interval [0.0, 1.0].
        """
        if total == 0:
            return 0.0

        p = positive / total

        denominator = 1 + (z * z) / total
        center = p + (z * z) / (2 * total)
        spread = z * math.sqrt((p * (1 - p) + (z * z) / (4 * total)) / total)

        return (center - spread) / denominator

    def compute_from_metrics(self, metrics: Dict[str, float]) -> float:
        """
        Compute Wilson score by thresholding metrics into binary signals.

        Uses the same thresholds as the original QualityScorer.
        """
        signals = [
            (True, metrics.get("citation_quality", 0) > 0.5),
            (True, metrics.get("writing_quality", 0) > 0.5),
            (True, metrics.get("content_depth", 0) > 0.5),
            (True, metrics.get("methodology_transparency", 0) > 0.5),
            (True, metrics.get("specificity", 0) > 0.3),
            (True, metrics.get("source_reputation", 0) > 0.6),
            (True, metrics.get("structural_integrity", 0) > 0.8),
        ]

        positive_count = sum(1 for present, is_pos in signals if present and is_pos)
        total_count = sum(1 for present, _ in signals if present)

        return self.compute(positive_count, total_count)
