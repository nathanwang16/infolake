#!/usr/bin/env python3
"""
Health Check Script

Computes coverage metrics and checks atlas health.

All parameters read from config.json under "monitor" section.

Input:
    - Documents from SQLite
    
Output:
    - Health alerts and metrics to console
    - Metrics stored in coverage_metrics table

Usage:
    python scripts/run_health_check.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging.logger import setup_logger, get_logger
from common.config import config

logger = get_logger("health_check")


def main():
    parser = argparse.ArgumentParser(
        description="Health Check - Compute coverage metrics and check atlas health"
    )
    
    parser.add_argument(
        "--gini-threshold",
        type=float,
        default=config.get("monitor.gini_alert_threshold", 0.6),
        help="Gini coefficient alert threshold"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger("health_check", console_output=True)
    
    logger.info("=" * 60)
    logger.info("HEALTH CHECK")
    logger.info("=" * 60)
    
    from monitor.health import CoverageMonitor
    
    monitor = CoverageMonitor()
    
    # Compute metrics
    metrics = monitor.compute_coverage_metrics()
    
    # Check health
    alerts = monitor.check_health()
    
    # Summary
    print("\n" + "=" * 60)
    print("COVERAGE METRICS")
    print("=" * 60)
    print(f"Topic Gini: {metrics['topic_gini']:.3f}")
    print(f"Domain Gini: {metrics['domain_gini']:.3f}")
    print(f"Orphan Rate: {metrics['orphan_rate']:.1%}")
    print(f"High-Quality Orphans: {metrics['high_quality_orphans']}")
    print(f"Cluster Count: {metrics['cluster_count']}")
    print(f"Largest Cluster: {metrics['largest_cluster_pct']:.1%}")
    print("=" * 60)
    
    if alerts:
        print("\nALERTS:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("\nNo alerts. Atlas health is good.")


if __name__ == "__main__":
    main()
