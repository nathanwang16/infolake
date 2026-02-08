#!/usr/bin/env python3
"""
Backfill document_texts table from the parsed archive.

Reads compressed JSONL archive files and inserts (doc_id, text)
into document_texts for any document that exists in the documents
table but is missing from document_texts.

Builds upon existing SQLite data without deleting anything.

Usage:
    python scripts/backfill_texts.py
    python scripts/backfill_texts.py --dry-run
    python scripts/backfill_texts.py --batch-size 500
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging.logger import setup_logger, get_logger
from common.config import config
from common.database import db

logger = get_logger("backfill_texts")


def decompress_archive(filepath: Path) -> list:
    """Decompresses a JSONL archive file and returns parsed records."""
    with open(filepath, 'rb') as f:
        raw = f.read()

    ext = ''.join(filepath.suffixes)

    if '.zst' in ext:
        try:
            import zstandard
            dctx = zstandard.ZstdDecompressor()
            data = dctx.decompress(raw, max_output_size=256 * 1024 * 1024)
        except ImportError:
            logger.error("zstandard not installed; cannot decompress .zst files")
            raise
    elif '.gz' in ext:
        data = zlib.decompress(raw, zlib.MAX_WBITS | 16)
    else:
        data = raw

    lines = data.decode('utf-8').strip().split('\n')
    records = []
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Backfill document_texts from parsed archive"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="SQLite insert batch size (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count matches without writing",
    )
    args = parser.parse_args()

    setup_logger("backfill_texts", console_output=True)

    archive_dir = Path(config.get("paths.parsed_archive_dir")).resolve()
    if not archive_dir.exists():
        logger.error(f"Parsed archive directory not found: {archive_dir}")
        sys.exit(1)

    # Collect archive files
    archive_files = sorted(archive_dir.rglob("*.jsonl.*"))
    if not archive_files:
        logger.error(f"No archive files found in {archive_dir}")
        sys.exit(1)

    logger.info(f"Found {len(archive_files)} archive files in {archive_dir}")

    # Load set of doc_ids that already have text
    conn = db.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id FROM document_texts")
        existing_text_ids = {row[0] for row in cursor.fetchall()}
        logger.info(f"Existing document_texts rows: {len(existing_text_ids)}")

        cursor.execute("SELECT id FROM documents")
        doc_ids_in_db = {row[0] for row in cursor.fetchall()}
        logger.info(f"Documents in database: {len(doc_ids_in_db)}")
    finally:
        conn.close()

    missing_ids = doc_ids_in_db - existing_text_ids
    logger.info(f"Documents missing text: {len(missing_ids)}")

    if not missing_ids:
        logger.info("All documents already have text. Nothing to backfill.")
        return

    # Scan archive files and collect texts for missing doc_ids
    start_time = time.time()
    total_inserted = 0
    total_scanned = 0
    batch_buffer = []

    conn = db.get_connection()
    try:
        cursor = conn.cursor()

        for i, archive_path in enumerate(archive_files):
            try:
                records = decompress_archive(archive_path)
            except Exception as e:
                logger.warning(f"Failed to read {archive_path.name}: {e}")
                continue

            total_scanned += len(records)

            for rec in records:
                doc_id = rec.get('id')
                text = rec.get('text')

                if not doc_id or not text:
                    continue
                if doc_id not in missing_ids:
                    continue
                if doc_id in existing_text_ids:
                    continue

                batch_buffer.append((doc_id, text))
                existing_text_ids.add(doc_id)

                if len(batch_buffer) >= args.batch_size:
                    if not args.dry_run:
                        cursor.executemany(
                            "INSERT OR IGNORE INTO document_texts (doc_id, text) VALUES (?, ?)",
                            batch_buffer,
                        )
                        conn.commit()
                    total_inserted += len(batch_buffer)
                    batch_buffer = []

            if (i + 1) % 20 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(archive_files)} files, "
                    f"{total_scanned:,} records scanned, "
                    f"{total_inserted:,} texts inserted"
                )

        # Flush remaining
        if batch_buffer:
            if not args.dry_run:
                cursor.executemany(
                    "INSERT OR IGNORE INTO document_texts (doc_id, text) VALUES (?, ?)",
                    batch_buffer,
                )
                conn.commit()
            total_inserted += len(batch_buffer)

    finally:
        conn.close()

    elapsed = time.time() - start_time

    still_missing = len(missing_ids) - total_inserted
    mode = "DRY RUN" if args.dry_run else "COMPLETE"

    print(f"\n{'='*60}")
    print(f"BACKFILL {mode}")
    print(f"{'='*60}")
    print(f"Archive files scanned: {len(archive_files)}")
    print(f"Archive records scanned: {total_scanned:,}")
    print(f"Texts inserted: {total_inserted:,}")
    print(f"Still missing (not in archive): {still_missing:,}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}")

    if still_missing > 0:
        logger.warning(
            f"{still_missing} documents have no text in the archive. "
            "These may be from runs that did not archive content, "
            "or from a different dump."
        )


if __name__ == "__main__":
    main()
