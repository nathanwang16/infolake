"""Structural integrity metric."""

from typing import Dict, Any, List

from curation.scoring.protocols import ScoringMetric


class StructuralIntegrityMetric(ScoringMetric):
    """Scores structural integrity via paragraph analysis."""

    @property
    def name(self) -> str:
        return "structural_integrity"

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
        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 50]
        if paragraphs:
            avg_para_len = sum(len(p) for p in paragraphs) / len(paragraphs)
            para_score = 1.0
            if avg_para_len < 100:
                para_score = avg_para_len / 100
            elif avg_para_len > 1000:
                para_score = max(0, 1.0 - (avg_para_len - 1000) / 1000)
            return para_score
        return 0.2  # Wall of text
