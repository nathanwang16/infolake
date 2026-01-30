"""
Writer module for Phase 1 batch processing pipeline.

Single-threaded writer that:
1. Receives processed documents from workers
2. Checks novelty against existing index
3. Applies FPS selection criterion
4. Scores documents using calibrated weights
5. Writes to SQLite and Qdrant
6. Archives parsed content

Key features:
- Single-threaded to avoid locks on database writes
- FPS-based coverage maximization
- Integration with calibrated quality scoring
- Content deduplication via SimHash
- Compressed JSONL archival
"""

import hashlib
import json
import time
import uuid
import zlib
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import numpy as np

from src.logging.logger import get_logger
from common.config import config
from common.database import db
from curation.scoring import scorer
from phase1_offline.fps_sampler import FarthestPointSampler
from phase1_offline.deduplication import Deduplicator, compute_content_hash

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


class Writer:
    """
    Single-threaded writer for the batch processing pipeline.
    
    Receives documents from workers, applies selection criteria,
    and writes to persistent storage.
    """
    
    def __init__(self, embed_queue: Queue):
        if embed_queue is None:
            raise ValueError("embed_queue is required")
        
        self.embed_queue = embed_queue
        self.running = False
        
        # Database connection - created in start() for thread safety
        self.conn = None
        self._db_path = config.get("database.sqlite_path", "data/atlas.db")
        
        # Configuration
        self.novelty_threshold = config.get("batch_processing.novelty_threshold", 0.08)
        self.quality_threshold = config.get("batch_processing.quality_threshold", 0.3)
        self.quality_alpha = config.get("batch_processing.quality_weight_alpha", 1.0)
        
        # Initialize FPS sampler
        self.fps_sampler = FarthestPointSampler(
            quality_weight_alpha=self.quality_alpha,
            use_qdrant=False  # Use local for now
        )
        
        # Initialize deduplicator
        expected_docs = config.get("batch_processing.expected_documents", 1_000_000)
        self.deduplicator = Deduplicator(expected_documents=expected_docs)
        
        # Initialize archiver
        archive_dir = config.get("paths.parsed_archive_dir", "data/parsed_archive")
        self.archiver = ContentArchiver(archive_dir)
        
        # Qdrant client
        self.qdrant = None
        self.qdrant_collection = config.get("qdrant.collection", "atlas_embeddings")
        self._qdrant_available = False
        self._qdrant_error_logged = False
        self._init_qdrant()
        
        # Statistics
        self._stats = {
            'received': 0,
            'novelty_rejected': 0,
            'quality_rejected': 0,
            'dedup_rejected': 0,
            'fps_rejected': 0,
            'accepted': 0,
            'db_errors': 0,
            'qdrant_errors': 0,
        }
        
        logger.info(f"Writer initialized (novelty_th={self.novelty_threshold}, "
                   f"quality_th={self.quality_threshold}, alpha={self.quality_alpha})")
    
    def _init_qdrant(self):
        """Initializes Qdrant client and collection."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            url = config.get("qdrant.url", "http://localhost:6333")
            self.qdrant = QdrantClient(url=url, timeout=5)
            
            # Check/create collection
            try:
                self.qdrant.get_collection(self.qdrant_collection)
                logger.info(f"Using existing Qdrant collection: {self.qdrant_collection}")
                self._qdrant_available = True
            except Exception:
                try:
                    logger.info(f"Creating Qdrant collection: {self.qdrant_collection}")
                    self.qdrant.create_collection(
                        collection_name=self.qdrant_collection,
                        vectors_config=VectorParams(
                            size=384,  # bge-small-en-v1.5 dimension
                            distance=Distance.COSINE
                        )
                    )
                    self._qdrant_available = True
                except Exception as e:
                    logger.warning(f"Qdrant not available: {e}. Running without vector search.")
                    self.qdrant = None
                
        except ImportError:
            logger.warning("qdrant-client not installed, running without vector search")
            self.qdrant = None
        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}. Running without vector search.")
            self.qdrant = None
    
    def start(self):
        """Main writer loop."""
        self.running = True
        
        # Create database connection in the writer thread for thread safety
        import sqlite3
        self.conn = sqlite3.connect(self._db_path)
        
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
        pass
    
    def _process_item(self, item: Dict[str, Any]):
        """Processes a single item from the embed queue."""
        self._stats['received'] += 1
        
        url = item['url']
        embedding = np.array(item['embedding'])
        text = item['text']
        metadata = item['metadata']
        raw_html_hash = item['raw_html_hash']
        
        # 1. Content deduplication (SimHash)
        content_hash = compute_content_hash(text)
        dedup_result = self.deduplicator.check_content(
            doc_id=content_hash,
            text=text,
            use_minhash=False  # SimHash is faster
        )
        
        if dedup_result.is_duplicate:
            self._stats['dedup_rejected'] += 1
            logger.debug(f"Dedup rejected: {url} (similar to {dedup_result.duplicate_of})")
            return
        
        # 2. Compute quality score using calibrated weights
        content_type = self._detect_content_type(text, metadata)
        raw_metrics = scorer.compute_raw_metrics(text, metadata)
        quality_score = scorer.compute_score(raw_metrics, content_type)
        
        if quality_score < self.quality_threshold:
            self._stats['quality_rejected'] += 1
            logger.debug(f"Quality rejected: {url} (score={quality_score:.3f})")
            return
        
        # 3. FPS selection (novelty + coverage)
        doc_id = self._generate_doc_id(url)
        should_select, distance, fps_score = self.fps_sampler.should_select(
            doc_id=doc_id,
            embedding=embedding,
            quality_score=quality_score,
            novelty_threshold=self.novelty_threshold
        )
        
        if not should_select:
            if distance < self.novelty_threshold:
                self._stats['novelty_rejected'] += 1
                logger.debug(f"Novelty rejected: {url} (distance={distance:.4f})")
            else:
                self._stats['fps_rejected'] += 1
                logger.debug(f"FPS rejected: {url}")
            return
        
        # 4. Register with deduplicator
        self.deduplicator.register_content(doc_id, text)
        self.deduplicator.register_url(url)
        
        # 5. Save to SQLite
        self._save_to_db(
            doc_id=doc_id,
            url=url,
            text=text,
            metadata=metadata,
            raw_metrics=raw_metrics,
            quality_score=quality_score,
            raw_html_hash=raw_html_hash,
            content_type=content_type,
            novelty_distance=distance
        )
        
        # 6. Save to Qdrant
        self._save_to_qdrant(doc_id, embedding, metadata, quality_score)
        
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
            logger.info(f"Accepted {self._stats['accepted']} documents "
                       f"(quality_rejected={self._stats['quality_rejected']}, "
                       f"novelty_rejected={self._stats['novelty_rejected']})")
    
    def _generate_doc_id(self, url: str) -> str:
        """Generates document ID from URL."""
        # Use UUID5 for reproducible IDs from URLs
        namespace = uuid.UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')  # URL namespace
        return str(uuid.uuid5(namespace, url))
    
    def _detect_content_type(self, text: str, metadata: Dict) -> str:
        """
        Detects content type for calibrated scoring.
        
        Types: technical_code, academic, news, blog, default
        """
        url = (metadata.get('url') or '').lower()
        title = (metadata.get('title') or '').lower()
        
        # Technical/code detection
        code_indicators = ['documentation', 'docs', 'api', 'reference', 'tutorial']
        if any(ind in url or ind in title for ind in code_indicators):
            return 'technical_code'
        
        # Academic detection
        academic_indicators = ['arxiv', 'journal', 'research', 'paper', 'study']
        if any(ind in url or ind in title for ind in academic_indicators):
            return 'academic'
        
        # News detection
        news_indicators = ['news', 'article', 'breaking', 'report']
        if any(ind in url or ind in title for ind in news_indicators):
            return 'news'
        
        return config.get("calibration.default_content_type", "default")
    
    def _save_to_db(
        self,
        doc_id: str,
        url: str,
        text: str,
        metadata: Dict,
        raw_metrics: Dict,
        quality_score: float,
        raw_html_hash: str,
        content_type: str,
        novelty_distance: float
    ):
        """Saves document to SQLite database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO documents 
                (id, canonical_url, title, content_hash, domain, 
                 detected_content_type, quality_score, quality_components,
                 quality_profile_used, raw_html_hash, novelty_distance,
                 source_phase, content_length, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'active')
            """, (
                doc_id,
                url,
                metadata.get('title', ''),
                hashlib.md5(text.encode()).hexdigest(),
                self._get_domain(url),
                content_type,
                quality_score,
                json.dumps(raw_metrics),
                content_type,
                raw_html_hash,
                novelty_distance,
                'batch',
                len(text),
            ))
            self.conn.commit()
            
        except Exception as e:
            self._stats['db_errors'] += 1
            logger.error(f"Database write failed: {e}")
    
    def _save_to_qdrant(
        self,
        doc_id: str,
        embedding: np.ndarray,
        metadata: Dict,
        quality_score: float
    ):
        """Saves embedding to Qdrant."""
        if not self.qdrant or not self._qdrant_available:
            return
        
        try:
            from qdrant_client.models import PointStruct
            
            # Convert string doc_id to UUID for Qdrant
            qdrant_id = str(uuid.UUID(doc_id))
            
            self.qdrant.upsert(
                collection_name=self.qdrant_collection,
                points=[
                    PointStruct(
                        id=qdrant_id,
                        vector=embedding.tolist(),
                        payload={
                            'title': metadata.get('title', ''),
                            'url': metadata.get('url', ''),
                            'quality_score': quality_score,
                        }
                    )
                ]
            )
            
        except Exception as e:
            self._stats['qdrant_errors'] += 1
            # Only log first error to avoid spam
            if not self._qdrant_error_logged:
                logger.warning(f"Qdrant write failed (further errors suppressed): {e}")
                self._qdrant_error_logged = True
                self._qdrant_available = False
    
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
        return {
            **self._stats,
            'fps_stats': self.fps_sampler.get_stats(),
            'dedup_stats': self.deduplicator.get_stats(),
            'archive_stats': self.archiver.get_stats(),
            'acceptance_rate': (
                self._stats['accepted'] / max(self._stats['received'], 1)
            ),
        }


class BatchWriter(Writer):
    """
    Extended writer with batch processing optimizations.
    
    Buffers writes for better performance with large dumps.
    """
    
    def __init__(self, embed_queue: Queue, db_batch_size: int = 100):
        super().__init__(embed_queue)
        self.db_batch_size = db_batch_size
        self._db_buffer: List[tuple] = []
        self._qdrant_buffer: List[Any] = []
    
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
            cursor = self.conn.cursor()
            
            for kwargs in self._db_buffer:
                cursor.execute("""
                    INSERT OR IGNORE INTO documents 
                    (id, canonical_url, title, content_hash, domain, 
                     detected_content_type, quality_score, quality_components,
                     quality_profile_used, raw_html_hash, novelty_distance,
                     source_phase, content_length, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'active')
                """, (
                    kwargs['doc_id'],
                    kwargs['url'],
                    kwargs['metadata'].get('title', ''),
                    hashlib.md5(kwargs['text'].encode()).hexdigest(),
                    self._get_domain(kwargs['url']),
                    kwargs['content_type'],
                    kwargs['quality_score'],
                    json.dumps(kwargs['raw_metrics']),
                    kwargs['content_type'],
                    kwargs['raw_html_hash'],
                    kwargs['novelty_distance'],
                    'batch',
                    len(kwargs['text']),
                ))
            
            self.conn.commit()
            logger.debug(f"Flushed {len(self._db_buffer)} records to database")
            
        except Exception as e:
            self._stats['db_errors'] += len(self._db_buffer)
            logger.error(f"Batch database write failed: {e}")
        
        self._db_buffer = []
    
    def _on_shutdown(self):
        """Flushes buffers at end of writer thread."""
        self._flush_db_buffer()
