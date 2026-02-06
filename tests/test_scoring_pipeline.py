"""Tests for the modular scoring pipeline, registry, and components."""

import pytest

from curation.scoring.protocols import ScoringMetric, ScoreAggregator, ContentTypeDetector
from curation.scoring.registry import MetricRegistry
from curation.scoring.pipeline import ScoringPipeline
from curation.scoring.aggregation import WeightedSigmoidAggregator, WilsonScoreComputer
from curation.scoring.detection import RuleBasedContentTypeDetector
from curation.scoring.metrics import BUILTIN_METRICS


# ── MetricRegistry ─────────────────────────────────────────────

class TestMetricRegistry:
    def test_register_and_get(self):
        reg = MetricRegistry()
        metric = BUILTIN_METRICS[0]
        reg.register(metric)
        assert reg.get(metric.name) is metric

    def test_names_order(self):
        reg = MetricRegistry()
        for m in BUILTIN_METRICS:
            reg.register(m)
        assert reg.names == [m.name for m in BUILTIN_METRICS]

    def test_unregister(self):
        reg = MetricRegistry()
        reg.register(BUILTIN_METRICS[0])
        reg.unregister(BUILTIN_METRICS[0].name)
        assert reg.get(BUILTIN_METRICS[0].name) is None
        assert len(reg) == 0

    def test_unregister_nonexistent_is_noop(self):
        reg = MetricRegistry()
        reg.unregister("does_not_exist")  # Should not raise

    def test_default_weights(self):
        reg = MetricRegistry()
        for m in BUILTIN_METRICS:
            reg.register(m)
        weights = reg.default_weights
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_contains(self):
        reg = MetricRegistry()
        reg.register(BUILTIN_METRICS[0])
        assert BUILTIN_METRICS[0].name in reg
        assert "nonexistent" not in reg


# ── Custom Metric ──────────────────────────────────────────────

class _ConstantMetric(ScoringMetric):
    """Test metric that always returns a fixed value."""

    def __init__(self, val: float, metric_name: str = "constant"):
        self._val = val
        self._name = metric_name

    @property
    def name(self):
        return self._name

    @property
    def default_weight(self):
        return 0.5

    def compute(self, text, words, sentences, metadata):
        return self._val


class TestCustomMetric:
    def test_custom_metric_in_pipeline(self):
        m = _ConstantMetric(0.42, "custom_score")
        pipe = ScoringPipeline(metrics=[m])
        raw = pipe.compute_raw_metrics("hello world", {})
        assert "custom_score" in raw
        assert raw["custom_score"] == pytest.approx(0.42)

    def test_register_replaces_existing(self):
        pipe = ScoringPipeline(metrics=[_ConstantMetric(0.1, "x")])
        pipe.registry.register(_ConstantMetric(0.9, "x"))
        raw = pipe.compute_raw_metrics("hello world", {})
        assert raw["x"] == pytest.approx(0.9)


# ── ScoringPipeline ───────────────────────────────────────────

class TestScoringPipeline:
    @pytest.fixture
    def pipeline(self):
        return ScoringPipeline()

    def test_builtin_metrics_registered(self, pipeline):
        assert len(pipeline.registry) == 7

    def test_empty_text_returns_zeros(self, pipeline):
        raw = pipeline.compute_raw_metrics("", {})
        assert all(v == 0.0 for v in raw.values())

    def test_compute_score_range(self, pipeline):
        raw = pipeline.compute_raw_metrics(
            "This is a test document about methodology and results. " * 20,
            {"url": "https://example.com"},
        )
        score = pipeline.compute_score(raw)
        assert 0.0 <= score <= 1.0

    def test_weight_override(self, pipeline):
        pipeline.update_weights("custom", {"citation_quality": 1.0})
        # Other metrics have zero weight under "custom"
        raw = {m: 0.5 for m in pipeline.registry.names}
        raw["citation_quality"] = 1.0
        score = pipeline.compute_score(raw, "custom")
        # With only citation_quality weighted at 1.0, raw_score should be 1.0
        assert score > 0.9

    def test_detect_content_type_academic(self, pipeline):
        ct = pipeline.detect_content_type("", {"url": "https://arxiv.org/paper", "title": ""})
        assert ct == "academic"

    def test_detect_content_type_technical(self, pipeline):
        ct = pipeline.detect_content_type("", {"url": "https://docs.python.org", "title": "API reference"})
        assert ct == "technical_code"

    def test_detect_content_type_news(self, pipeline):
        ct = pipeline.detect_content_type("", {"url": "https://news.example.com", "title": ""})
        assert ct == "news"


# ── WeightedSigmoidAggregator ─────────────────────────────────

class TestWeightedSigmoidAggregator:
    def test_zero_weights(self):
        agg = WeightedSigmoidAggregator()
        assert agg.aggregate({"a": 0.5}, {}) == 0.0

    def test_perfect_score(self):
        agg = WeightedSigmoidAggregator()
        assert agg.aggregate({"a": 1.0}, {"a": 1.0}) > 0.9

    def test_zero_score(self):
        agg = WeightedSigmoidAggregator()
        assert agg.aggregate({"a": 0.0}, {"a": 1.0}) < 0.1


# ── WilsonScoreComputer ───────────────────────────────────────

class TestWilsonScoreComputer:
    def test_zero_total(self):
        w = WilsonScoreComputer()
        assert w.compute(0, 0) == 0.0

    def test_all_positive(self):
        w = WilsonScoreComputer()
        assert w.compute(100, 100) > 0.95

    def test_from_metrics_all_high(self):
        w = WilsonScoreComputer()
        metrics = {
            "citation_quality": 0.9,
            "writing_quality": 0.9,
            "content_depth": 0.9,
            "methodology_transparency": 0.9,
            "specificity": 0.9,
            "source_reputation": 0.9,
            "structural_integrity": 0.9,
        }
        score = w.compute_from_metrics(metrics)
        assert score > 0.5

    def test_from_metrics_all_zero(self):
        w = WilsonScoreComputer()
        metrics = {m: 0.0 for m in [
            "citation_quality", "writing_quality", "content_depth",
            "methodology_transparency", "specificity",
            "source_reputation", "structural_integrity",
        ]}
        score = w.compute_from_metrics(metrics)
        assert score < 0.1


# ── RuleBasedContentTypeDetector ───────────────────────────────

class TestRuleBasedContentTypeDetector:
    def test_detect_technical(self):
        d = RuleBasedContentTypeDetector()
        assert d.detect("", {"url": "https://docs.python.org/tutorial", "title": ""}) == "technical_code"

    def test_detect_academic(self):
        d = RuleBasedContentTypeDetector()
        assert d.detect("", {"url": "https://arxiv.org/abs/123", "title": ""}) == "academic"

    def test_detect_news(self):
        d = RuleBasedContentTypeDetector()
        assert d.detect("", {"url": "https://news.bbc.com/article", "title": ""}) == "news"

    def test_detect_default(self):
        d = RuleBasedContentTypeDetector()
        ct = d.detect("", {"url": "https://example.com", "title": ""})
        assert isinstance(ct, str)
