"""Citation quality metric."""

import re
from typing import Dict, Any, List

from curation.scoring.protocols import ScoringMetric


class CitationQualityMetric(ScoringMetric):
    """Scores citation presence: bracketed refs, parenthetical cites, links, keywords."""

    @property
    def name(self) -> str:
        return "citation_quality"

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
        has_brackets = len(re.findall(r'\[\d+\]', text))
        has_parens_cite = len(re.findall(r'\([A-Z][a-z]+, \d{4}\)', text))
        has_http = text.count("http://") + text.count("https://")
        has_keywords = sum(
            1 for k in ["references", "bibliography", "sources", "cited"]
            if k in text.lower()[-1000:]
        )
        return min(
            (has_brackets * 0.2)
            + (has_parens_cite * 0.3)
            + (has_http * 0.1)
            + (has_keywords * 0.4),
            1.0,
        )
