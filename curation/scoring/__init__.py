"""
Scoring package — modular quality scoring system.

Backward-compatible exports:
    QualityScorer  — facade over ScoringPipeline
    scorer         — module-level singleton

New public API:
    ScoringPipeline, MetricRegistry, ScoringMetric, ScoreAggregator,
    ContentTypeDetector, WeightedSigmoidAggregator, WilsonScoreComputer,
    RuleBasedContentTypeDetector
"""

from curation.scoring._compat import QualityScorer
from curation.scoring.pipeline import ScoringPipeline
from curation.scoring.registry import MetricRegistry
from curation.scoring.protocols import ScoringMetric, ScoreAggregator, ContentTypeDetector
from curation.scoring.aggregation import WeightedSigmoidAggregator, WilsonScoreComputer
from curation.scoring.detection import RuleBasedContentTypeDetector

# Module-level singleton (preserves `from curation.scoring import scorer`)
scorer = QualityScorer()

__all__ = [
    # Backward compat
    "QualityScorer",
    "scorer",
    # Protocols
    "ScoringMetric",
    "ScoreAggregator",
    "ContentTypeDetector",
    # Pipeline
    "ScoringPipeline",
    "MetricRegistry",
    # Aggregation
    "WeightedSigmoidAggregator",
    "WilsonScoreComputer",
    # Detection
    "RuleBasedContentTypeDetector",
]
