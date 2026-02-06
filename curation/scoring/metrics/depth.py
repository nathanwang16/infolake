"""Content depth metric."""

import math
from typing import Dict, Any, List

from curation.scoring.protocols import ScoringMetric


class ContentDepthMetric(ScoringMetric):
    """Scores content depth via log-normalized length and lexical diversity."""

    @property
    def name(self) -> str:
        return "content_depth"

    @property
    def default_weight(self) -> float:
        return 0.2

    def compute(
        self,
        text: str,
        words: List[str],
        sentences: List[str],
        metadata: Dict[str, Any],
    ) -> float:
        total_words = len(words)
        unique_words = len(set(words))

        length_score = min(math.log(total_words + 1) / math.log(4000), 1.0)
        diversity_score = unique_words / total_words if total_words > 0 else 0

        return (length_score * 0.7) + (diversity_score * 0.3)
