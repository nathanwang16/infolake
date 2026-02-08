#!/usr/bin/env python3
"""
Meilisearch Sync Script

Backfills Meilisearch index with existing documents from SQLite.
Run this once after setting up Meilisearch to index documents
that were ingested before Meilisearch was added.

Usage:
    python scripts/meili_sync.py
    python scripts/meili_sync.py --batch-size 500
    python scripts/meili_sync.py --clear   # wipe index before sync
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging.logger import setup_logger, get_logger
from common.config import config
from common.database import db
from common.meilisearch_manager import MeilisearchManager

logger = get_logger("meili_sync")


def sync_documents(batch_size: int = 200, clear: bool = False):
    """Reads all active documents from SQLite and indexes them into Meilisearch."""

    meili = MeilisearchManager(create_if_missing=True)
    if not meili.available:
        logger.error("Meilisearch is not available. Ensure it is running.")
        sys.exit(1)

    if clear:
        logger.info("Clearing Meilisearch index...")
        try:
            task = meili.index.delete_all_documents()
            meili.client.wait_for_task(task.task_uid, timeout_in_ms=60000)
            logger.info("Index cleared")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            sys.exit(1)

    conn = db.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, canonical_url, title, summary, domain,
                   detected_content_type, quality_score, cluster_id
            FROM documents
            WHERE status = 'active'
            ORDER BY created_at
        """)

        total = 0
        batch = []
        start = time.time()

        for row in cursor:
            doc = {
                'id': row[0],
                'url': row[1],
                'title': row[2] or '',
                'summary': row[3] or '',
                'domain': row[4] or '',
                'content_type': row[5] or 'unscored',
                'quality_score': row[6] or 0.0,
                'cluster_id': row[7],
            }
            batch.append(doc)

            if len(batch) >= batch_size:
                meili.add_documents(batch)
                total += len(batch)
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0
                logger.info(f"Indexed {total:,} documents ({rate:.0f} docs/sec)")
                batch = []

        # Flush remaining
        if batch:
            meili.add_documents(batch)
            total += len(batch)

        elapsed = time.time() - start
        logger.info(f"Sync complete: {total:,} documents in {elapsed:.1f}s")

    finally:
        conn.close()

    # Log Meilisearch stats
    stats = meili.get_stats()
    logger.info(f"Meilisearch index: {stats.get('numberOfDocuments', 0):,} documents")


def main():
    parser = argparse.ArgumentParser(description="Sync SQLite documents to Meilisearch")
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Documents per batch (default: 200)"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear Meilisearch index before syncing"
    )
    args = parser.parse_args()

    setup_logger("meili_sync", console_output=True)

    logger.info("=" * 50)
    logger.info("MEILISEARCH SYNC")
    logger.info(f"Meilisearch: {config.get('meilisearch.url')}")
    logger.info(f"Index: {config.get('meilisearch.index')}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Clear first: {args.clear}")
    logger.info("=" * 50)

    sync_documents(batch_size=args.batch_size, clear=args.clear)


if __name__ == "__main__":
    main()
