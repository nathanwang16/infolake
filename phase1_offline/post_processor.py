"""
Post-processor for deferred scoring, filtering, FPS selection, and deduplication.

Runs as a standalone stage after the batch pipeline has stored documents.
Processes unscored documents (quality_profile_used='pending') in batches.

Workflow:
1. Query unscored documents from database
2. Load full text from document_texts table
3. Detect content type
4. Compute quality scores using calibrated weights
5. Compute Wilson scores
6. Run SimHash dedup
7. Optionally run FPS selection
8. Update documents table with scores
"""

import json
import time
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from common.logging.logger import get_logger
from common.config import config
from common.database import db as _default_db
from common.repositories import DocumentRepository, DocumentTextRepository
from common.qdrant_manager import QdrantManager
from curation.scoring import scorer as _default_scorer
from phase1_offline.deduplication import Deduplicator, compute_content_hash
from phase1_offline.fps_sampler import FarthestPointSampler

logger = get_logger("post_processor")


class PostProcessor:
    """Standalone scoring/filtering/FPS/dedup stage for batch-processed documents."""

    def __init__(
        self,
        database=None,
        qdrant_manager=None,
        quality_scorer=None,
        doc_repo=None,
        text_repo=None,
    ):
        self._database = database or _default_db
        self._qdrant_mgr = qdrant_manager or QdrantManager(create_if_missing=False, timeout=5)
        self._scorer = quality_scorer or _default_scorer
        self._doc_repo = doc_repo or DocumentRepository(self._database)
        self._text_repo = text_repo or DocumentTextRepository(self._database)

        # Configuration
        self.quality_threshold = config.get("batch_processing.quality_threshold")
        self.quality_alpha = config.get("batch_processing.quality_weight_alpha")
        self.novelty_threshold = config.get("batch_processing.novelty_threshold")

        # Deduplicator
        expected_docs = config.get("batch_processing.expected_documents")
        self.deduplicator = Deduplicator(expected_documents=expected_docs)

        # FPS sampler
        self.fps_sampler = FarthestPointSampler(
            quality_weight_alpha=self.quality_alpha,
            use_qdrant=False,
        )

        # Statistics
        self._stats = {
            'total_processed': 0,
            'scored': 0,
            'quality_rejected': 0,
            'dedup_rejected': 0,
            'accepted': 0,
            'errors': 0,
        }

    def run(
        self,
        batch_size: int = 1000,
        quality_threshold: Optional[float] = None,
        skip_fps: bool = False,
        skip_dedup: bool = False,
    ):
        """
        Process unscored documents in batches.

        Args:
            batch_size: Number of documents to process per batch
            quality_threshold: Override minimum quality score (None = use config)
            skip_fps: Skip FPS selection step
            skip_dedup: Skip SimHash dedup step
        """
        if quality_threshold is not None:
            self.quality_threshold = quality_threshold

        logger.info(
            f"PostProcessor starting: batch_size={batch_size}, "
            f"quality_th={self.quality_threshold}, skip_fps={skip_fps}, "
            f"skip_dedup={skip_dedup}"
        )

        start_time = time.time()
        batch_num = 0

        while True:
            # Fetch next batch of unscored documents
            batch = self._text_repo.get_unscored_batch(batch_size)
            if not batch:
                logger.info("No more unscored documents to process")
                break

            batch_num += 1
            logger.info(f"Processing batch {batch_num}: {len(batch)} documents")

            for doc_id, text in batch:
                try:
                    self._process_document(
                        doc_id, text,
                        skip_fps=skip_fps,
                        skip_dedup=skip_dedup,
                    )
                    self._stats['total_processed'] += 1
                except Exception as e:
                    logger.error(f"Error processing {doc_id}: {e}")
                    self._stats['errors'] += 1

            # Log batch progress
            elapsed = time.time() - start_time
            rate = self._stats['total_processed'] / max(elapsed, 1)
            logger.info(
                f"Batch {batch_num} complete: "
                f"processed={self._stats['total_processed']}, "
                f"scored={self._stats['scored']}, "
                f"rejected={self._stats['quality_rejected']}, "
                f"rate={rate:.1f} docs/s"
            )

        elapsed = time.time() - start_time
        logger.info(
            f"PostProcessor finished in {elapsed:.1f}s: {self._stats}"
        )
        return self._stats

    def _process_document(
        self,
        doc_id: str,
        text: str,
        skip_fps: bool = False,
        skip_dedup: bool = False,
    ):
        """Process a single document: score, dedup, optionally FPS."""
        # Get document metadata from DB
        doc = self._doc_repo.get_by_id(doc_id)
        metadata = {}
        if doc:
            metadata = {
                'url': doc.url,
                'title': doc.title,
                'domain': doc.domain,
            }

        # 1. Detect content type
        content_type = self._detect_content_type(text, metadata)

        # 2. Compute quality score
        raw_metrics = self._scorer.compute_raw_metrics(text, metadata)
        quality_score = self._scorer.compute_score(raw_metrics, content_type)

        # 3. SimHash dedup check
        if not skip_dedup:
            content_hash = compute_content_hash(text)
            dedup_result = self.deduplicator.check_content(
                doc_id=content_hash, text=text, use_minhash=False
            )
            if dedup_result.is_duplicate:
                self._stats['dedup_rejected'] += 1
                self._update_document_score(
                    doc_id, quality_score, raw_metrics, content_type,
                    status='duplicate'
                )
                return

            # Register for future dedup checks
            self.deduplicator.register_content(doc_id, text)

        # 4. Update document with scores
        self._update_document_score(
            doc_id, quality_score, raw_metrics, content_type,
            status='active'
        )
        self._stats['scored'] += 1

        # 5. Quality threshold check (flag but don't delete)
        if quality_score < self.quality_threshold:
            self._stats['quality_rejected'] += 1
        else:
            self._stats['accepted'] += 1

    def _detect_content_type(self, text: str, metadata: Dict) -> str:
        """Detects content type for calibrated scoring."""
        url = (metadata.get('url') or '').lower()
        title = (metadata.get('title') or '').lower()

        code_indicators = ['documentation', 'docs', 'api', 'reference', 'tutorial']
        if any(ind in url or ind in title for ind in code_indicators):
            return 'technical_code'

        academic_indicators = ['arxiv', 'journal', 'research', 'paper', 'study']
        if any(ind in url or ind in title for ind in academic_indicators):
            return 'academic'

        news_indicators = ['news', 'article', 'breaking', 'report']
        if any(ind in url or ind in title for ind in news_indicators):
            return 'news'

        return config.get("calibration.default_content_type")

    def _update_document_score(
        self,
        doc_id: str,
        quality_score: float,
        raw_metrics: Dict,
        content_type: str,
        status: str = 'active',
    ):
        """Update document with computed quality scores."""
        conn = self._database.get_connection()
        try:
            cursor = conn.cursor()

            # Compute Wilson score approximation
            # Use quality_score as proxy: positive = score * 100, total = 100
            positive = int(quality_score * 100)
            wilson = self._scorer.compute_wilson_score(positive, 100)

            cursor.execute("""
                UPDATE documents
                SET quality_score = ?,
                    quality_components = ?,
                    quality_profile_used = ?,
                    detected_content_type = ?,
                    wilson_score = ?,
                    status = ?
                WHERE id = ?
            """, (
                quality_score,
                json.dumps(raw_metrics),
                content_type,
                content_type,
                wilson,
                status,
                doc_id,
            ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update document {doc_id}: {e}")
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Returns post-processing statistics."""
        return {
            **self._stats,
            'dedup_stats': self.deduplicator.get_stats(),
        }
