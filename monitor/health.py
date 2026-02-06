import math
import sqlite3
from typing import List, Dict, Any
from common.database import db
from common.models import CoverageMetrics
from common.repositories import MetricsRepository
from common.logging.logger import get_logger

logger = get_logger("monitor")

class CoverageMonitor:
    def __init__(self, database=None, metrics_repo=None):
        self._database = database or db
        self._metrics_repo = metrics_repo or MetricsRepository(self._database)

    def gini_coefficient(self, counts: List[int]) -> float:
        """
        Compute Gini coefficient for a list of category counts.
        0 = perfect equality, 1 = max inequality.
        """
        if not counts:
            return 0.0

        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        total = sum(sorted_counts)

        if total == 0:
            return 0.0

        gini_sum = 0
        for i, count in enumerate(sorted_counts):
            gini_sum += (2 * (i + 1) - n - 1) * count

        return gini_sum / (n * total)

    def compute_coverage_metrics(self) -> Dict[str, Any]:
        """
        Computes coverage metrics (Gini, orphans, etc.) and stores them in DB.
        """
        # Topic Coverage
        topic_counts = self._metrics_repo.get_topic_distribution()
        topic_gini = self.gini_coefficient(topic_counts)

        # Domain Coverage
        domain_counts = self._metrics_repo.get_domain_distribution()
        domain_gini = self.gini_coefficient(domain_counts)

        # Cluster Health
        cluster_counts = self._metrics_repo.get_cluster_distribution()

        total_docs = sum(cluster_counts.values()) or 1
        orphan_count = cluster_counts.get(-1, 0)
        orphan_rate = orphan_count / total_docs

        # High quality orphans
        high_quality_orphans = self._metrics_repo.get_high_quality_orphan_count()

        # Largest cluster pct
        non_orphan_counts = [c for cid, c in cluster_counts.items() if cid != -1]
        cluster_count = len(non_orphan_counts)
        max_cluster_size = max(non_orphan_counts) if non_orphan_counts else 0
        largest_cluster_pct = max_cluster_size / total_docs

        metrics = CoverageMetrics(
            topic_gini=topic_gini,
            domain_gini=domain_gini,
            orphan_rate=orphan_rate,
            high_quality_orphans=high_quality_orphans,
            cluster_count=cluster_count,
            largest_cluster_pct=largest_cluster_pct,
        )

        # Store in DB
        self._metrics_repo.insert_coverage_metrics(metrics)

        result = metrics.to_dict()
        logger.info(f"Computed Coverage Metrics: {result}")
        return result

    def check_health(self):
        """
        Checks metrics against thresholds and logs alerts.
        """
        metrics = self.compute_coverage_metrics()

        alerts = []
        if metrics["topic_gini"] > 0.6:
            alerts.append(f"WARNING: Topic coverage imbalanced (Gini={metrics['topic_gini']:.2f})")

        if metrics["high_quality_orphans"] > 100:
             alerts.append(f"INFO: {metrics['high_quality_orphans']} high-quality orphaned docs found.")

        if metrics["largest_cluster_pct"] > 0.3:
            alerts.append(f"WARNING: Largest cluster dominates {metrics['largest_cluster_pct']*100:.1f}% of atlas.")

        for alert in alerts:
            logger.warning(alert)

        return alerts

if __name__ == "__main__":
    monitor = CoverageMonitor()
    monitor.check_health()
