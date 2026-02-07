"""
Scoring package — modular quality scoring system.

Public API:
    ScoringPipeline, MetricRegistry, ScoringMetric, ScoreAggregator,
    ContentTypeDetector, WeightedSigmoidAggregator, WilsonScoreComputer,
    RuleBasedContentTypeDetector
"""

from curation.scoring.pipeline import ScoringPipeline
from curation.scoring.registry import MetricRegistry
from curation.scoring.protocols import ScoringMetric, ScoreAggregator, ContentTypeDetector
from curation.scoring.aggregation import WeightedSigmoidAggregator, WilsonScoreComputer
from curation.scoring.detection import RuleBasedContentTypeDetector

__all__ = [
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
