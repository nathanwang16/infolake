#!/usr/bin/env python3
"""
Post-Processing Script: Deferred Scoring & Filtering

Runs quality scoring, deduplication, and content type detection
on documents stored by the batch pipeline with quality_profile_used='pending'.

Input:
    - Documents in SQLite with quality_profile_used='pending'
    - Full text in document_texts table

Output:
    - Updated quality_score, wilson_score, detected_content_type
    - Status flags for duplicates and low-quality documents

Usage:
    python scripts/post_process.py
    python scripts/post_process.py --batch-size 500
    python scripts/post_process.py --quality-threshold 0.2 --skip-dedup
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging.logger import setup_logger, get_logger
from common.config import config
from phase1_offline.post_processor import PostProcessor

logger = get_logger("post_process")


def main():
    parser = argparse.ArgumentParser(
        description="Post-Processing: Score and filter batch-processed documents"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of documents per processing batch (default: 1000)"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help=f"Minimum quality score (default: {config.get('batch_processing.quality_threshold')})"
    )
    parser.add_argument(
        "--skip-fps",
        action="store_true",
        help="Skip Farthest Point Sampling selection"
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip SimHash deduplication"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger("post_process", console_output=True)

    logger.info("=" * 60)
    logger.info("POST-PROCESSING: Scoring & Filtering")
    logger.info("=" * 60)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Quality threshold: {args.quality_threshold or config.get('batch_processing.quality_threshold')}")
    logger.info(f"Skip FPS: {args.skip_fps}")
    logger.info(f"Skip dedup: {args.skip_dedup}")
    logger.info("=" * 60)

    processor = PostProcessor()
    stats = processor.run(
        batch_size=args.batch_size,
        quality_threshold=args.quality_threshold,
        skip_fps=args.skip_fps,
        skip_dedup=args.skip_dedup,
    )

    # Summary
    print("\n" + "=" * 60)
    print("POST-PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total processed: {stats.get('total_processed', 0):,}")
    print(f"Scored: {stats.get('scored', 0):,}")
    print(f"Accepted (above threshold): {stats.get('accepted', 0):,}")
    print(f"Quality rejected: {stats.get('quality_rejected', 0):,}")
    print(f"Dedup rejected: {stats.get('dedup_rejected', 0):,}")
    print(f"Errors: {stats.get('errors', 0):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
