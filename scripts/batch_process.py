#!/usr/bin/env python3
"""
Phase 1: Batch Processing Script

Processes dump files through the producer-consumer pipeline:
    Producer → url_queue → Workers → embed_queue → Writer → Atlas

All parameters read from config.json under "batch_processing" section.

Input:
    - Dump file path (from config or --dump argument)
    
Output:
    - Documents in SQLite (data/atlas.db)
    - Embeddings in Qdrant (if available)
    - Parsed content archive (data/parsed_archive/)

Usage:
    python scripts/batch_process.py
    python scripts/batch_process.py --dump dataset/sample-test-50.tar
    python scripts/batch_process.py --limit 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging.logger import setup_logger, get_logger
from common.config import config
from phase1_offline.pipeline import BatchPipeline

logger = get_logger("batch_process")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Batch Processing - Process dump files into Atlas"
    )
    
    # All params default to config.json values
    parser.add_argument(
        "--dump",
        default=config.get("batch_processing.default_dump"),
        help=f"Dump file path (default: {config.get('batch_processing.default_dump')})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.get("batch_processing.workers", 8),
        help=f"Worker threads (default: {config.get('batch_processing.workers', 8)})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.get("batch_processing.limit", 0),
        help="Limit URLs to process (0=no limit)"
    )
    parser.add_argument(
        "--url-queue-size",
        type=int,
        default=config.get("batch_processing.url_queue_size", 5000),
        help="URL queue capacity"
    )
    parser.add_argument(
        "--embed-queue-size",
        type=int,
        default=config.get("batch_processing.embed_queue_size", 2000),
        help="Embedding queue capacity"
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.dump:
        parser.error("No dump file specified. Set batch_processing.default_dump in config.json or use --dump")
    
    dump_path = Path(args.dump)
    if not dump_path.exists():
        parser.error(f"Dump file not found: {dump_path}")
    
    # Setup logging
    setup_logger("batch_process", console_output=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 1: BATCH PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Dump: {args.dump}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Limit: {args.limit if args.limit > 0 else 'None'}")
    logger.info("=" * 60)
    
    # Run pipeline
    pipeline = BatchPipeline(
        dump_path=str(dump_path),
        num_workers=args.workers,
        limit=args.limit,
        url_queue_size=args.url_queue_size,
        embed_queue_size=args.embed_queue_size
    )
    
    stats = pipeline.run()
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Duration: {stats.duration_human}")
    print(f"Throughput: {stats.throughput:.1f} docs/sec")
    
    if stats.producer_stats:
        print(f"\nProducer: {stats.producer_stats.get('queued', 0):,} queued")
    
    if stats.writer_stats:
        print(f"Writer: {stats.writer_stats.get('accepted', 0):,} accepted")
        print(f"Acceptance rate: {stats.writer_stats.get('acceptance_rate', 0):.1%}")
    
    print("=" * 60)
    
    # Output location
    print(f"\nOutput:")
    print(f"  Database: {config.get('database.sqlite_path')}")
    print(f"  Archive: {config.get('paths.parsed_archive_dir')}")


if __name__ == "__main__":
    main()
