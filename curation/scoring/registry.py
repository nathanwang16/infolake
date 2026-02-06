"""Metric registry for discovery and management."""

from typing import Dict, List, Optional

from curation.scoring.protocols import ScoringMetric


class MetricRegistry:
    """Registry for scoring metrics — register, unregister, lookup by name."""

    def __init__(self):
        self._metrics: Dict[str, ScoringMetric] = {}

    def register(self, metric: ScoringMetric) -> None:
        """Register a metric (replaces existing with same name)."""
        self._metrics[metric.name] = metric

    def unregister(self, name: str) -> None:
        """Remove a metric by name. No-op if not found."""
        self._metrics.pop(name, None)

    def get(self, name: str) -> Optional[ScoringMetric]:
        """Look up a metric by name."""
        return self._metrics.get(name)

    @property
    def names(self) -> List[str]:
        """Ordered list of registered metric names."""
        return list(self._metrics.keys())

    @property
    def metrics(self) -> List[ScoringMetric]:
        """All registered metrics in insertion order."""
        return list(self._metrics.values())

    @property
    def default_weights(self) -> Dict[str, float]:
        """Dict of {name: default_weight} for all registered metrics."""
        return {m.name: m.default_weight for m in self._metrics.values()}

    def __len__(self) -> int:
        return len(self._metrics)

    def __contains__(self, name: str) -> bool:
        return name in self._metrics
