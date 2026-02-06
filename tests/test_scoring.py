"""Tests for curation/scoring.py — quality scoring and Wilson score."""

import pytest

from curation.scoring import QualityScorer


@pytest.fixture
def scorer():
    return QualityScorer()


# ── Wilson Score ──────────────────────────────────────────────

class TestWilsonScore:
    def test_zero_total(self, scorer):
        assert scorer.compute_wilson_score(0, 0) == 0.0

    def test_all_positive(self, scorer):
        score = scorer.compute_wilson_score(100, 100)
        assert 0.95 < score <= 1.0

    def test_all_negative(self, scorer):
        score = scorer.compute_wilson_score(0, 100)
        assert score < 0.05

    def test_half_positive(self, scorer):
        score = scorer.compute_wilson_score(50, 100)
        assert 0.3 < score < 0.6

    def test_small_sample_conservative(self, scorer):
        """Wilson score should be conservative (lower) with small samples."""
        small = scorer.compute_wilson_score(1, 1)  # 100% but n=1
        large = scorer.compute_wilson_score(100, 100)  # 100% with n=100
        assert small < large


# ── compute_raw_metrics ───────────────────────────────────────

class TestRawMetrics:
    def test_empty_text_returns_zeros(self, scorer):
        metrics = scorer.compute_raw_metrics("", {})
        assert all(v == 0.0 for v in metrics.values())

    def test_returns_expected_keys(self, scorer):
        text = "This is a test document with enough words to pass the check."
        metrics = scorer.compute_raw_metrics(text, {"url": "https://example.com"})
        for key in scorer.METRICS:
            assert key in metrics

    def test_no_link_density_penalty_key(self, scorer):
        """link_density_penalty was removed in Issue 6."""
        text = "Some document text that is long enough to be processed by the scorer."
        metrics = scorer.compute_raw_metrics(text, {"url": "https://example.com"})
        assert "link_density_penalty" not in metrics

    def test_domain_reputation_edu(self, scorer):
        text = "Research paper on quantum computing with comprehensive methodology and results."
        metrics = scorer.compute_raw_metrics(text, {"url": "https://mit.edu/paper"})
        assert metrics["source_reputation"] == 0.9

    def test_domain_reputation_neutral(self, scorer):
        text = "Some random content about general topics and everyday matters."
        metrics = scorer.compute_raw_metrics(text, {"url": "https://somesite.com/page"})
        assert metrics["source_reputation"] == 0.5

    def test_metrics_range(self, scorer):
        """All metrics should be in [0, 1]."""
        text = (
            "This is a comprehensive document about machine learning. "
            "The methodology involves training neural networks [1]. "
            "Results show significant improvement (Smith, 2023). " * 20
        )
        metrics = scorer.compute_raw_metrics(
            text, {"url": "https://example.com"}
        )
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"


# ── compute_score ─────────────────────────────────────────────

class TestComputeScore:
    def test_zero_metrics(self, scorer):
        metrics = {m: 0.0 for m in scorer.METRICS}
        score = scorer.compute_score(metrics)
        assert score < 0.1  # Sigmoid of 0 → near 0

    def test_perfect_metrics(self, scorer):
        metrics = {m: 1.0 for m in scorer.METRICS}
        score = scorer.compute_score(metrics)
        assert score > 0.9  # Sigmoid of 1 → near 1

    def test_unknown_content_type_uses_default(self, scorer):
        metrics = {m: 0.5 for m in scorer.METRICS}
        score_default = scorer.compute_score(metrics, "default")
        score_unknown = scorer.compute_score(metrics, "nonexistent_type")
        assert score_default == score_unknown


# ── update_weights ────────────────────────────────────────────

class TestUpdateWeights:
    def test_update_creates_new_type(self, scorer):
        scorer.update_weights("custom_type", {"citation_quality": 1.0})
        assert "custom_type" in scorer.weights

    def test_update_overwrites_existing(self, scorer):
        scorer.update_weights("default", {"citation_quality": 0.5})
        assert scorer.weights["default"]["citation_quality"] == 0.5
