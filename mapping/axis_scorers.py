"""Built-in axis scorer implementations."""

from typing import Dict, Optional

import numpy as np

from mapping.protocols import AxisScorer


class DomainAuthorityAxisScorer(AxisScorer):
    """
    Z-axis importance scorer based on domain authority heuristics.

    Extracted from ImportanceScorer in mapper.py.
    """

    DOMAIN_AUTHORITY: Dict[str, float] = {
        # Academic/Research
        'arxiv.org': 0.95,
        'scholar.google.com': 0.90,
        'nature.com': 0.95,
        'science.org': 0.95,
        'ieee.org': 0.90,
        'acm.org': 0.90,
        # Educational
        '.edu': 0.85,
        '.ac.uk': 0.85,
        '.gov': 0.80,
        # Tech documentation
        'docs.python.org': 0.85,
        'developer.mozilla.org': 0.90,
        'docs.microsoft.com': 0.85,
        'cloud.google.com': 0.85,
        # Quality tech blogs
        'martinfowler.com': 0.80,
        'norvig.com': 0.85,
        'paulgraham.com': 0.80,
    }

    def __init__(self):
        self._cache: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "domain_authority"

    def compute(
        self,
        domain: str,
        quality_score: float,
        content_type: Optional[str] = None,
        inbound_links: Optional[int] = None,
        citations: Optional[int] = None,
    ) -> float:
        if domain is None:
            raise ValueError("domain is required")
        if quality_score is None:
            raise ValueError("quality_score is required")

        cache_key = f"{domain}:{quality_score:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        domain_score = self._get_domain_authority(domain)

        if inbound_links is not None:
            link_score = min(np.log10(inbound_links + 1) / 3.5, 1.0)
        else:
            link_score = domain_score * 0.8

        if citations is not None:
            citation_score = min(np.log10(citations + 1) / 2.5, 1.0)
        else:
            citation_score = 0.7 if content_type == 'scientific' else 0.3

        importance = (
            0.5 * domain_score
            + 0.3 * link_score
            + 0.2 * citation_score
        )

        importance = importance * 0.7 + quality_score * 0.3

        self._cache[cache_key] = importance
        return importance

    def _get_domain_authority(self, domain: str) -> float:
        domain = domain.lower()

        if domain in self.DOMAIN_AUTHORITY:
            return self.DOMAIN_AUTHORITY[domain]

        for pattern, score in self.DOMAIN_AUTHORITY.items():
            if pattern.startswith('.') and domain.endswith(pattern):
                return score
            if domain.endswith('.' + pattern):
                return score

        if '.edu' in domain or '.ac.' in domain:
            return 0.75
        if '.gov' in domain or '.mil' in domain:
            return 0.70
        if '.org' in domain:
            return 0.55

        return 0.40
