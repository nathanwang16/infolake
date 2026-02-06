"""Built-in scoring metrics."""

from curation.scoring.metrics.citation import CitationQualityMetric
from curation.scoring.metrics.writing import WritingQualityMetric
from curation.scoring.metrics.depth import ContentDepthMetric
from curation.scoring.metrics.methodology import MethodologyTransparencyMetric
from curation.scoring.metrics.specificity import SpecificityMetric
from curation.scoring.metrics.reputation import SourceReputationMetric
from curation.scoring.metrics.structure import StructuralIntegrityMetric

BUILTIN_METRICS = [
    CitationQualityMetric(),
    WritingQualityMetric(),
    ContentDepthMetric(),
    MethodologyTransparencyMetric(),
    SpecificityMetric(),
    SourceReputationMetric(),
    StructuralIntegrityMetric(),
]

__all__ = [
    "CitationQualityMetric",
    "WritingQualityMetric",
    "ContentDepthMetric",
    "MethodologyTransparencyMetric",
    "SpecificityMetric",
    "SourceReputationMetric",
    "StructuralIntegrityMetric",
    "BUILTIN_METRICS",
]
