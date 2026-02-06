"""Tests for monitor/health.py — Gini coefficient and coverage metrics."""

import pytest

from monitor.health import CoverageMonitor


@pytest.fixture
def monitor(memory_db, metrics_repo):
    return CoverageMonitor(database=memory_db, metrics_repo=metrics_repo)


class TestGiniCoefficient:
    def test_perfect_equality(self, monitor):
        """All categories have same count → Gini = 0."""
        counts = [10, 10, 10, 10]
        assert monitor.gini_coefficient(counts) == 0.0

    def test_maximum_inequality(self, monitor):
        """One category has everything → Gini near 1."""
        counts = [0, 0, 0, 100]
        gini = monitor.gini_coefficient(counts)
        assert gini > 0.7

    def test_empty_list(self, monitor):
        assert monitor.gini_coefficient([]) == 0.0

    def test_all_zeros(self, monitor):
        assert monitor.gini_coefficient([0, 0, 0]) == 0.0

    def test_single_element(self, monitor):
        assert monitor.gini_coefficient([42]) == 0.0

    def test_range_zero_to_one(self, monitor):
        """Gini should always be in [0, 1]."""
        import random
        for _ in range(10):
            counts = [random.randint(0, 100) for _ in range(random.randint(2, 20))]
            gini = monitor.gini_coefficient(counts)
            assert 0.0 <= gini <= 1.0, f"Gini={gini} for counts={counts}"
