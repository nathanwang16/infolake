"""
Writer module for Phase 1 batch processing pipeline.

Single-threaded writer that:
1. Receives processed documents with embeddings
2. Generates document IDs
3. Saves to SQLite, document_texts table, and Qdrant
4. Archives parsed content

Scoring, FPS selection, and deduplication are deferred to the
post-processing stage (post_processor.py).

Key features:
- Single-threaded to avoid locks on database writes
- Simple store-only path for maximum throughput
- Text stored in document_texts table for deferred scoring
- Compressed JSONL archival
"""

import hashlib
import json
import threading
import time
import uuid
import zlib
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import numpy as np

from common.logging.logger import get_logger
from common.config import config
from common.database import db
from common.models import DocumentCreate
from common.repositories import DocumentRepository, DocumentTextRepository
from common.qdrant_manager import QdrantManager
from common.meilisearch_manager import MeilisearchManager

logger = get_logger("writer")


class ContentArchiver:
    """
    Archives parsed content to compressed JSONL files.

    Structure:
        parsed_archive/YYYY/MM/DD/batch_XXX.jsonl.zst

    Each record contains:
        url, raw_html_hash, text, title, author, date, fetch_date, extractor_version
    """

    def __init__(self, archive_dir: str, batch_size: int = 1000):
        if archive_dir is None:
            raise ValueError("archive_dir is required")

        self.archive_dir = Path(archive_dir)
        self.batch_size = batch_size

        self._buffer: List[Dict[str, Any]] = []
        self._current_batch = 0
        self._total_archived = 0

        # Try to use zstandard for better compression
        try:
            import zstandard
            self._zstd = zstandard
            self._compression = 'zstd'
        except ImportError:
            self._zstd = None
            self._compression = 'gzip'
            logger.warning("zstandard not installed, using gzip for archival")

    def archive(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Archives a record, returning the archive path when batch is flushed.

        Args:
            record: Document record to archive

        Returns:
            Archive path if batch was flushed, None otherwise
        """
        if record is None:
            raise ValueError("record is required")

        archive_record = {
            'id': record.get('id'),
            'url': record.get('url'),
            'raw_html_hash': record.get('raw_html_hash'),
            'text': record.get('text'),
            'title': record.get('metadata', {}).get('title'),
            'author': record.get('metadata', {}).get('author'),
            'publication_date': record.get('metadata', {}).get('date'),
            'fetch_date': datetime.now().isoformat(),
            'extractor_version': 'trafilatura-1.0',
        }

        self._buffer.append(archive_record)

        if len(self._buffer) >= self.batch_size:
            return self._flush()

        return None

    def _flush(self) -> str:
        """Flushes buffer to compressed file."""
        if not self._buffer:
            return ""

        # Create date-based directory structure
        now = datetime.now()
        date_dir = self.archive_dir / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
        date_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        self._current_batch += 1
        ext = 'jsonl.zst' if self._compression == 'zstd' else 'jsonl.gz'
        filepath = date_dir / f"batch_{self._current_batch:06d}.{ext}"

        # Write records
        try:
            content = '\n'.join(json.dumps(r) for r in self._buffer)
            content_bytes = content.encode('utf-8')

            if self._compression == 'zstd' and self._zstd:
                cctx = self._zstd.ZstdCompressor(level=3)
                compressed = cctx.compress(content_bytes)
            else:
                compressed = zlib.compress(content_bytes, level=6)

            with open(filepath, 'wb') as f:
                f.write(compressed)

            self._total_archived += len(self._buffer)
            logger.info(f"Archived {len(self._buffer)} records to {filepath}")

        except Exception as e:
            logger.error(f"Archive flush failed: {e}")
            filepath = Path("")

        self._buffer = []
        return str(filepath)

    def flush(self) -> Optional[str]:
        """Forces flush of remaining buffer."""
        if self._buffer:
            return self._flush()
        return None

    def get_stats(self) -> Dict[str, int]:
        """Returns archival statistics."""
        return {
            'total_archived': self._total_archived,
            'current_batch': self._current_batch,
            'buffer_size': len(self._buffer),
        }


class QdrantWriteWorker(threading.Thread):
    """Background worker for batched Qdrant upserts."""

    def __init__(
        self,
        qdrant_manager: QdrantManager,
        input_queue: Queue,
        batch_size: int,
        batch_timeout_seconds: float,
    ):
        super().__init__(name="QdrantWriteWorker", daemon=True)

        if qdrant_manager is None:
            logger.error("qdrant_manager is required")
            raise ValueError("qdrant_manager is required")
        if input_queue is None:
            logger.error("input_queue is required")
            raise ValueError("input_queue is required")
        if batch_size is None:
            logger.error("batch_size is required")
            raise ValueError("batch_size is required")
        if batch_timeout_seconds is None:
            logger.error("batch_timeout_seconds is required")
            raise ValueError("batch_timeout_seconds is required")

        self._qdrant_mgr = qdrant_manager
        self._queue = input_queue
        self._batch_size = batch_size
        self._batch_timeout_seconds = batch_timeout_seconds
        self.running = True

        self._stats = {
            'batches': 0,
            'points': 0,
            'errors': 0,
        }

    def run(self):
        batch = []
        last_flush = time.time()

        while self.running or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                item = None

            if item is not None:
                batch.append(item)
                self._queue.task_done()

            now = time.time()
            should_flush = (
                len(batch) >= self._batch_size or
                (batch and now - last_flush >= self._batch_timeout_seconds)
            )

            if should_flush and batch:
                self._flush_batch(batch)
                batch = []
                last_flush = now

        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[Dict[str, Any]]):
        if not self._qdrant_mgr.available:
            self._stats['errors'] += len(batch)
            return

        try:
            from qdrant_client.models import PointStruct

            points = []
            for item in batch:
                qdrant_id = str(uuid.UUID(item['doc_id']))
                points.append(
                    PointStruct(
                        id=qdrant_id,
                        vector=item['embedding'],
                        payload=item['payload'],
                    )
                )

            result = self._qdrant_mgr.upsert(points=points)
            if result is None:
                self._stats['errors'] += len(batch)
                return

            self._stats['batches'] += 1
            self._stats['points'] += len(points)
        except Exception as e:
            logger.debug(f"Qdrant batch upsert failed: {e}")
            self._stats['errors'] += len(batch)

    def stop(self):
        self.running = False

    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()


class Writer:
    """
    Single-threaded writer for the batch processing pipeline.

    Receives documents with embeddings and stores them.
    Scoring and selection are deferred to the post-processor.
    """

    def __init__(
        self,
        embed_queue: Queue,
        database=None,
        qdrant_manager=None,
        meilisearch_manager=None,
        doc_repo=None,
        text_repo=None,
    ):
        if embed_queue is None:
            raise ValueError("embed_queue is required")

        self.embed_queue = embed_queue
        self.running = False

        # DI
        self._database = database or db
        qdrant_timeout = config.require("qdrant.timeout_seconds")
        self._qdrant_mgr = qdrant_manager or QdrantManager(
            create_if_missing=True,
            timeout=qdrant_timeout,
        )
        self._meili_mgr = meilisearch_manager or MeilisearchManager(create_if_missing=True)
        self._doc_repo = doc_repo or DocumentRepository(self._database)
        self._text_repo = text_repo or DocumentTextRepository(self._database)

        # Database connection - created in start() for thread safety
        self.conn = None
        self._db_path = self._database.db_path

        # Initialize archiver
        archive_dir = config.get("paths.parsed_archive_dir")
        self.archiver = ContentArchiver(archive_dir)

        # Backwards-compatible properties
        self.qdrant = self._qdrant_mgr.client
        self.qdrant_collection = self._qdrant_mgr.collection_name
        self._qdrant_available = self._qdrant_mgr.available
        self._qdrant_error_logged = False
        self._qdrant_drop_logged = False

        self._qdrant_batch_size = config.require("qdrant.batch_size")
        self._qdrant_batch_timeout_seconds = config.require("qdrant.batch_timeout_seconds")
        self._qdrant_write_queue_size = config.require("qdrant.write_queue_size")
        self._qdrant_queue: Queue = Queue(maxsize=self._qdrant_write_queue_size)
        self._qdrant_worker: Optional[QdrantWriteWorker] = None

        # Statistics
        self._stats = {
            'received': 0,
            'accepted': 0,
            'db_errors': 0,
            'qdrant_errors': 0,
            'meili_errors': 0,
            'qdrant_enqueued': 0,
            'qdrant_dropped': 0,
        }

        logger.info("Writer initialized (store-only mode, scoring deferred)")

    def start(self):
        """Main writer loop."""
        self.running = True

        # Create database connection in the writer thread for thread safety
        import sqlite3
        self.conn = sqlite3.connect(self._db_path)

        self._start_qdrant_worker()

        logger.info("Writer started")

        while self.running or not self.embed_queue.empty():
            try:
                item = self.embed_queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                self._process_item(item)
            except Exception as e:
                logger.error(f"Writer error: {e}")
            finally:
                self.embed_queue.task_done()

        # Flush any remaining data before exiting (must be in writer thread)
        self._on_shutdown()

        # Flush archiver
        self.archiver.flush()

        logger.info(f"Writer stopped. Stats: {self._stats}")

    def _on_shutdown(self):
        """Called at end of writer thread to flush buffers. Override in subclasses."""
        self._shutdown_qdrant_worker()
        if self.conn:
            try:
                self.conn.commit()
            except Exception as e:
                logger.warning(f"Final commit failed: {e}")
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def _process_item(self, item: Dict[str, Any]):
        """Processes a single item from the embed queue."""
        self._stats['received'] += 1

        url = item['url']
        embedding = np.array(item['embedding'])
        text = item['text']
        metadata = item['metadata']
        raw_html_hash = item.get('raw_html_hash', '')

        # 1. Generate doc_id from URL
        doc_id = self._generate_doc_id(url)

        # 2. Save to SQLite (with quality_profile_used='pending', quality_score=0.0)
        self._save_to_db(
            doc_id=doc_id,
            url=url,
            text=text,
            metadata=metadata,
            raw_html_hash=raw_html_hash,
        )

        # 3. Save text to document_texts table
        try:
            self._text_repo.insert(doc_id, text, conn=self.conn)
        except Exception as e:
            logger.debug(f"Text insert failed for {url}: {e}")

        # 4. Commit document + text together
        try:
            self.conn.commit()
        except Exception as e:
            logger.error(f"Commit failed for {url}: {e}")

        # 5. Save embedding to Qdrant
        self._save_to_qdrant(doc_id, embedding, metadata)

        # 6. Index metadata in Meilisearch
        self._save_to_meilisearch(doc_id, url, text, metadata)

        # 7. Archive content
        archive_record = {
            'id': doc_id,
            'url': url,
            'text': text,
            'metadata': metadata,
            'raw_html_hash': raw_html_hash,
        }
        self.archiver.archive(archive_record)

        self._stats['accepted'] += 1

        if self._stats['accepted'] % 100 == 0:
            logger.info(f"Accepted {self._stats['accepted']} documents")

    def _generate_doc_id(self, url: str) -> str:
        """Generates document ID from URL."""
        # Use UUID5 for reproducible IDs from URLs
        namespace = uuid.UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')  # URL namespace
        return str(uuid.uuid5(namespace, url))

    def _save_to_db(
        self,
        doc_id: str,
        url: str,
        text: str,
        metadata: Dict,
        raw_html_hash: str,
    ):
        """Saves document to SQLite database."""
        try:
            doc = DocumentCreate(
                id=doc_id,
                url=url,
                title=metadata.get('title', ''),
                summary='',
                content_hash=hashlib.md5(text.encode()).hexdigest(),
                domain=self._get_domain(url),
                content_type='unscored',
                quality_score=0.0,
                quality_components='{}',
                quality_profile_used='pending',
                raw_html_hash=raw_html_hash,
                novelty_distance=0.0,
                source_phase='batch',
                source_dump=metadata.get('source_dump'),
                content_length=len(text),
            )
            self._doc_repo.insert(doc, conn=self.conn)

        except Exception as e:
            self._stats['db_errors'] += 1
            logger.error(f"Database write failed: {e}")

    def _save_to_qdrant(
        self,
        doc_id: str,
        embedding: np.ndarray,
        metadata: Dict,
    ):
        """Saves embedding to Qdrant."""
        if not self._qdrant_mgr.available:
            return
        if not self._qdrant_worker:
            self._start_qdrant_worker()
        if not self._qdrant_worker:
            return

        payload = {
            'title': metadata.get('title', ''),
            'url': metadata.get('url', ''),
            'quality_score': 0.0,
        }
        try:
            self._qdrant_queue.put_nowait({
                'doc_id': doc_id,
                'embedding': embedding.tolist(),
                'payload': payload,
            })
            self._stats['qdrant_enqueued'] += 1
        except Full:
            self._stats['qdrant_dropped'] += 1
            if not self._qdrant_drop_logged:
                logger.warning("Qdrant write queue full; dropping embeddings")
                self._qdrant_drop_logged = True

    def _save_to_meilisearch(
        self,
        doc_id: str,
        url: str,
        text: str,
        metadata: Dict,
    ):
        """Indexes document metadata in Meilisearch for full-text search."""
        if not self._meili_mgr.available:
            return

        try:
            self._meili_mgr.add_documents([{
                'id': doc_id,
                'title': metadata.get('title', ''),
                'url': url,
                'domain': self._get_domain(url),
                'content_type': 'unscored',
                'quality_score': 0.0,
                'summary': '',
                'cluster_id': None,
            }])
        except Exception as e:
            self._stats['meili_errors'] += 1
            if not self._meili_mgr._error_logged:
                logger.warning(f"Meilisearch index failed (further errors suppressed): {e}")

    def _get_domain(self, url: str) -> str:
        """Extracts domain from URL."""
        try:
            return urlparse(url).netloc
        except Exception:
            return ""

    def stop(self):
        """Signals writer to stop."""
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """Returns writer statistics."""
        qdrant_worker_stats = (
            self._qdrant_worker.get_stats()
            if self._qdrant_worker else {}
        )
        return {
            **self._stats,
            'archive_stats': self.archiver.get_stats(),
            'qdrant_worker': qdrant_worker_stats,
            'qdrant_errors_total': (
                self._stats.get('qdrant_errors', 0) +
                qdrant_worker_stats.get('errors', 0)
            ),
            'acceptance_rate': (
                self._stats['accepted'] / max(self._stats['received'], 1)
            ),
        }

    def _start_qdrant_worker(self):
        if not self._qdrant_mgr.available:
            return
        if self._qdrant_worker and self._qdrant_worker.is_alive():
            return
        self._qdrant_worker = QdrantWriteWorker(
            qdrant_manager=self._qdrant_mgr,
            input_queue=self._qdrant_queue,
            batch_size=self._qdrant_batch_size,
            batch_timeout_seconds=self._qdrant_batch_timeout_seconds,
        )
        self._qdrant_worker.start()

    def _shutdown_qdrant_worker(self):
        if not self._qdrant_worker:
            return
        self._qdrant_worker.stop()
        self._qdrant_worker.join(timeout=30.0)
        if self._qdrant_worker.is_alive():
            logger.warning("Qdrant worker did not finish in time")


class BatchWriter(Writer):
    """
    Extended writer with batch processing optimizations.

    Buffers writes for better performance with large dumps.
    """

    def __init__(self, embed_queue: Queue, db_batch_size: int = 100, **kwargs):
        super().__init__(embed_queue, **kwargs)
        self.db_batch_size = db_batch_size
        self._db_buffer: List[Dict] = []
        self._text_buffer: List[tuple] = []
        self._meili_buffer: List[Dict] = []

    def _save_to_db(self, **kwargs):
        """Buffers database writes."""
        self._db_buffer.append(kwargs)

        if len(self._db_buffer) >= self.db_batch_size:
            self._flush_db_buffer()

    def _flush_db_buffer(self):
        """Flushes database write buffer."""
        if not self._db_buffer:
            return

        try:
            docs = []
            text_items = []
            for kwargs in self._db_buffer:
                doc = DocumentCreate(
                    id=kwargs['doc_id'],
                    url=kwargs['url'],
                    title=kwargs['metadata'].get('title', ''),
                    summary='',
                    content_hash=hashlib.md5(kwargs['text'].encode()).hexdigest(),
                    domain=self._get_domain(kwargs['url']),
                    content_type='unscored',
                    quality_score=0.0,
                    quality_components='{}',
                    quality_profile_used='pending',
                    raw_html_hash=kwargs.get('raw_html_hash', ''),
                    novelty_distance=0.0,
                    source_phase='batch',
                    source_dump=kwargs['metadata'].get('source_dump'),
                    content_length=len(kwargs['text']),
                )
                docs.append(doc)
                text_items.append((kwargs['doc_id'], kwargs['text']))

            self._doc_repo.insert_batch(docs, conn=self.conn)
            self._text_repo.insert_batch(text_items, conn=self.conn)
            self.conn.commit()
            logger.debug(f"Flushed {len(self._db_buffer)} records to database")

        except Exception as e:
            self._stats['db_errors'] += len(self._db_buffer)
            logger.error(f"Batch database write failed: {e}")

        self._db_buffer = []

    def _process_item(self, item: Dict[str, Any]):
        """Processes a single item, buffering DB writes."""
        self._stats['received'] += 1

        url = item['url']
        embedding = np.array(item['embedding'])
        text = item['text']
        metadata = item['metadata']
        raw_html_hash = item.get('raw_html_hash', '')

        doc_id = self._generate_doc_id(url)

        # Buffer DB write (includes text)
        self._save_to_db(
            doc_id=doc_id,
            url=url,
            text=text,
            metadata=metadata,
            raw_html_hash=raw_html_hash,
        )

        # Save embedding to Qdrant
        self._save_to_qdrant(doc_id, embedding, metadata)

        # Buffer Meilisearch metadata
        self._buffer_meilisearch(doc_id, url, text, metadata)

        # Archive content
        archive_record = {
            'id': doc_id,
            'url': url,
            'text': text,
            'metadata': metadata,
            'raw_html_hash': raw_html_hash,
        }
        self.archiver.archive(archive_record)

        self._stats['accepted'] += 1

        if self._stats['accepted'] % 100 == 0:
            logger.info(f"Accepted {self._stats['accepted']} documents")

    def _buffer_meilisearch(self, doc_id: str, url: str, text: str, metadata: Dict):
        """Buffers a document for batch indexing into Meilisearch."""
        self._meili_buffer.append({
            'id': doc_id,
            'title': metadata.get('title', ''),
            'url': url,
            'domain': self._get_domain(url),
            'content_type': 'unscored',
            'quality_score': 0.0,
            'summary': '',
            'cluster_id': None,
        })
        if len(self._meili_buffer) >= self.db_batch_size:
            self._flush_meili_buffer()

    def _flush_meili_buffer(self):
        """Flushes buffered documents to Meilisearch."""
        if not self._meili_buffer:
            return
        self._meili_mgr.add_documents(self._meili_buffer)
        logger.debug(f"Flushed {len(self._meili_buffer)} documents to Meilisearch")
        self._meili_buffer = []

    def _on_shutdown(self):
        """Flushes buffers at end of writer thread, commits, and closes connection."""
        self._shutdown_qdrant_worker()
        self._flush_db_buffer()
        self._flush_meili_buffer()
        if self.conn:
            try:
                self.conn.commit()
            except Exception as e:
                logger.warning(f"Final commit failed: {e}")
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
