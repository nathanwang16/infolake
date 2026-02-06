"""Source reputation metric."""

from typing import Dict, Any, List

from curation.scoring.protocols import ScoringMetric


class SourceReputationMetric(ScoringMetric):
    """Scores source reputation based on domain TLD heuristics."""

    @property
    def name(self) -> str:
        return "source_reputation"

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
        domain = metadata.get("domain", "") or self._get_domain(
            metadata.get("url", "")
        )
        source_reputation = 0.5  # Neutral default

        if domain:
            if any(d in domain for d in [".edu", ".gov", ".mil"]):
                source_reputation = 0.9
            elif any(d in domain for d in [".org", ".ac.", ".sci"]):
                source_reputation = 0.8
            elif "wordpress" in domain or "blogspot" in domain:
                source_reputation = 0.4

        return source_reputation

    @staticmethod
    def _get_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return ""
