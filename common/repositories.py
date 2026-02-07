"""
Repository layer for Truth Atlas.

Each repository class accepts an optional Database instance,
defaulting to the module-level singleton when not provided.
Thread-safe: each method opens its own connection via db.get_connection().
"""

import json
from typing import Dict, List, Optional, Tuple, Any

from common.database import db as _default_db
from common.models import (
    Document,
    DocumentListItem,
    DocumentCreate,
    ClusterInfo,
    CoverageMetrics,
    GoldenSetEntry,
    DumpJob,
)


class DocumentRepository:
    """Replaces raw SQL in atlas_store, writer, mapper, gaps, server."""

    def __init__(self, database=None):
        self._db = database or _default_db

    # ---- reads ----

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, canonical_url, title, summary, domain,
                       detected_content_type, quality_score, wilson_score,
                       importance_score, cluster_id, created_at
                FROM documents WHERE id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Document.from_row(row)
        finally:
            conn.close()

    def get_list(
        self,
        limit: int = 100,
        offset: int = 0,
        content_type: Optional[str] = None,
        min_quality: Optional[float] = None,
        cluster_id: Optional[int] = None,
        order_by: str = 'quality_score DESC',
    ) -> List[DocumentListItem]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            query = """
                SELECT id, canonical_url, title, domain,
                       detected_content_type, quality_score, wilson_score,
                       importance_score, cluster_id, created_at
                FROM documents
                WHERE status = 'active'
            """
            params: list = []

            if content_type:
                query += " AND detected_content_type = ?"
                params.append(content_type)
            if min_quality is not None:
                query += " AND quality_score >= ?"
                params.append(min_quality)
            if cluster_id is not None:
                query += " AND cluster_id = ?"
                params.append(cluster_id)

            query += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return [DocumentListItem.from_row(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_count(
        self,
        content_type: Optional[str] = None,
        min_quality: Optional[float] = None,
    ) -> int:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) FROM documents WHERE status = 'active'"
            params: list = []

            if content_type:
                query += " AND detected_content_type = ?"
                params.append(content_type)
            if min_quality is not None:
                query += " AND quality_score >= ?"
                params.append(min_quality)

            cursor.execute(query, params)
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def get_cluster_stats(self) -> List[ClusterInfo]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cluster_id,
                       COUNT(*) as doc_count,
                       AVG(quality_score) as avg_quality,
                       MIN(quality_score) as min_quality,
                       MAX(quality_score) as max_quality
                FROM documents
                WHERE status = 'active'
                GROUP BY cluster_id
                ORDER BY doc_count DESC
            """)
            return [ClusterInfo.from_row(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_atlas_summary_stats(self) -> Dict[str, Any]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'active'")
            total_docs = cursor.fetchone()[0]

            cursor.execute("""
                SELECT detected_content_type, COUNT(*)
                FROM documents
                WHERE status = 'active'
                GROUP BY detected_content_type
            """)
            content_types = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("""
                SELECT
                    COUNT(CASE WHEN quality_score >= 0.7 THEN 1 END) as high,
                    COUNT(CASE WHEN quality_score >= 0.4 AND quality_score < 0.7 THEN 1 END) as medium,
                    COUNT(CASE WHEN quality_score < 0.4 THEN 1 END) as low
                FROM documents WHERE status = 'active'
            """)
            quality_dist = cursor.fetchone()

            cursor.execute("""
                SELECT COUNT(DISTINCT cluster_id)
                FROM documents
                WHERE status = 'active' AND cluster_id != -1
            """)
            cluster_count = cursor.fetchone()[0]

            return {
                'total_docs': total_docs,
                'content_types': content_types,
                'quality_dist': {
                    'high': quality_dist[0] if quality_dist else 0,
                    'medium': quality_dist[1] if quality_dist else 0,
                    'low': quality_dist[2] if quality_dist else 0,
                },
                'cluster_count': cluster_count,
            }
        finally:
            conn.close()

    def search_text(self, query: str, limit: int = 20) -> List[DocumentListItem]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, canonical_url, title, domain,
                       detected_content_type, quality_score, wilson_score,
                       importance_score, cluster_id, created_at
                FROM documents
                WHERE status = 'active'
                AND (title LIKE ? OR canonical_url LIKE ?)
                ORDER BY quality_score DESC
                LIMIT ?
            """, (f'%{query}%', f'%{query}%', limit))
            return [DocumentListItem.from_row(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_metadata_for_mapping(self, doc_id: str) -> Optional[Tuple]:
        """Returns (domain, quality_score, content_type) for a document."""
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT domain, quality_score, detected_content_type
                FROM documents WHERE id = ?
            """, (doc_id,))
            return cursor.fetchone()
        finally:
            conn.close()

    def get_quality_score(self, doc_id: str) -> Optional[float]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT quality_score FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def get_export_fields(self, doc_id: str) -> Optional[Dict]:
        """Returns url, title, domain, content_type for JSON export."""
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT canonical_url, title, domain, detected_content_type
                FROM documents WHERE id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                'url': row[0],
                'title': row[1],
                'domain': row[2],
                'content_type': row[3],
            }
        finally:
            conn.close()

    def get_random_ids(self, limit: int = 1000) -> List[str]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM documents ORDER BY RANDOM() LIMIT ?", (limit,))
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    # ---- writes ----

    def insert(self, doc: DocumentCreate, conn=None) -> None:
        """Insert a single document. Uses provided conn or opens a new one."""
        owns_conn = conn is None
        if owns_conn:
            conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO documents
                (id, canonical_url, title, summary, content_hash, domain,
                 detected_content_type, quality_score, quality_components,
                 quality_profile_used, raw_html_hash, novelty_distance,
                 source_phase, content_length, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'active')
            """, (
                doc.id, doc.url, doc.title, doc.summary,
                doc.content_hash, doc.domain, doc.content_type,
                doc.quality_score, doc.quality_components,
                doc.quality_profile_used, doc.raw_html_hash,
                doc.novelty_distance, doc.source_phase, doc.content_length,
            ))
            if owns_conn:
                conn.commit()
        except Exception:
            if owns_conn:
                conn.rollback()
            raise
        finally:
            if owns_conn:
                conn.close()

    def insert_batch(self, docs: List[DocumentCreate], conn=None) -> None:
        """Insert multiple documents in one transaction."""
        owns_conn = conn is None
        if owns_conn:
            conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            for doc in docs:
                cursor.execute("""
                    INSERT OR IGNORE INTO documents
                    (id, canonical_url, title, summary, content_hash, domain,
                     detected_content_type, quality_score, quality_components,
                     quality_profile_used, raw_html_hash, novelty_distance,
                     source_phase, content_length, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'active')
                """, (
                    doc.id, doc.url, doc.title, doc.summary,
                    doc.content_hash, doc.domain, doc.content_type,
                    doc.quality_score, doc.quality_components,
                    doc.quality_profile_used, doc.raw_html_hash,
                    doc.novelty_distance, doc.source_phase, doc.content_length,
                ))
            if owns_conn:
                conn.commit()
        except Exception:
            if owns_conn:
                conn.rollback()
            raise
        finally:
            if owns_conn:
                conn.close()

    def update_mapping(self, doc_id: str, cluster_id: int, importance_score: float) -> None:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET cluster_id = ?, importance_score = ?
                WHERE id = ?
            """, (cluster_id, importance_score, doc_id))
            conn.commit()
        finally:
            conn.close()

    def update_mappings_batch(self, updates: List[Tuple[int, float, str]]) -> None:
        """Updates cluster_id and importance_score for many docs.

        Args:
            updates: list of (cluster_id, importance_score, doc_id) tuples
        """
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany("""
                UPDATE documents
                SET cluster_id = ?, importance_score = ?
                WHERE id = ?
            """, updates)
            conn.commit()
        finally:
            conn.close()

    def update_score(
        self,
        doc_id: str,
        quality_score: float,
        quality_components: str,
        content_type: str,
        wilson_score: float,
        status: str = 'active',
    ) -> None:
        """Update quality scores for a single document.

        Args:
            doc_id: Document ID
            quality_score: Computed quality score (0.0-1.0)
            quality_components: JSON string of raw metrics
            content_type: Detected content type
            wilson_score: Wilson confidence score
            status: Document status ('active', 'duplicate', etc.)
        """
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
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
                quality_components,
                content_type,
                content_type,
                wilson_score,
                status,
                doc_id,
            ))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def update_scores_batch(self, updates: List[Tuple]) -> None:
        """Batch update quality scores for multiple documents.

        Args:
            updates: List of (doc_id, quality_score, quality_components_json,
                     content_type, wilson_score, status) tuples
        """
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany("""
                UPDATE documents
                SET quality_score = ?,
                    quality_components = ?,
                    quality_profile_used = ?,
                    detected_content_type = ?,
                    wilson_score = ?,
                    status = ?
                WHERE id = ?
            """, [(
                quality_score,
                quality_components_json,
                content_type,
                content_type,
                wilson_score,
                status,
                doc_id,
            ) for doc_id, quality_score, quality_components_json,
                  content_type, wilson_score, status in updates])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class MetricsRepository:
    """Replaces raw SQL in monitor/health.py."""

    def __init__(self, database=None):
        self._db = database or _default_db

    def get_topic_distribution(self) -> List[int]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT detected_content_type, COUNT(*) FROM documents GROUP BY detected_content_type"
            )
            return [count for _, count in cursor.fetchall()]
        finally:
            conn.close()

    def get_domain_distribution(self) -> List[int]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT domain, COUNT(*) FROM documents GROUP BY domain")
            return [count for _, count in cursor.fetchall()]
        finally:
            conn.close()

    def get_cluster_distribution(self) -> Dict[Optional[int], int]:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT cluster_id, COUNT(*) FROM documents GROUP BY cluster_id")
            return {cid: count for cid, count in cursor.fetchall()}
        finally:
            conn.close()

    def get_high_quality_orphan_count(self) -> int:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM documents WHERE cluster_id = -1 AND quality_score > 0.7"
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def insert_coverage_metrics(self, metrics: CoverageMetrics) -> None:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO coverage_metrics (
                    topic_gini, domain_gini, orphan_rate,
                    high_quality_orphans, cluster_count, largest_cluster_pct
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metrics.topic_gini, metrics.domain_gini, metrics.orphan_rate,
                metrics.high_quality_orphans, metrics.cluster_count,
                metrics.largest_cluster_pct,
            ))
            conn.commit()
        finally:
            conn.close()


class GoldenSetRepository:
    """Replaces raw SQL in calibrate.py."""

    def __init__(self, database=None):
        self._db = database or _default_db

    def upsert(self, entry: GoldenSetEntry) -> None:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO golden_set (url, label, content_type, raw_metrics, notes, version, domain)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    label=excluded.label,
                    content_type=excluded.content_type,
                    notes=excluded.notes,
                    raw_metrics=excluded.raw_metrics,
                    domain=excluded.domain
            """, (
                entry.url, entry.label, entry.content_type,
                entry.raw_metrics, entry.notes, entry.version, entry.domain,
            ))
            conn.commit()
        finally:
            conn.close()

    def get_for_training(self, content_type: str) -> List[Tuple[str, Dict, str]]:
        """Returns (label, raw_metrics_dict, domain) rows for training."""
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT label, raw_metrics, domain FROM golden_set
                WHERE content_type = ? AND raw_metrics IS NOT NULL
            """, (content_type,))
            result = []
            for label_str, metrics_json, domain in cursor.fetchall():
                result.append((label_str, json.loads(metrics_json), domain))
            return result
        finally:
            conn.close()

    def get_for_validation(self, content_type: str) -> List[Tuple[str, Dict]]:
        """Returns (label, raw_metrics_dict) rows for validation."""
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT label, raw_metrics FROM golden_set
                WHERE content_type = ? AND raw_metrics IS NOT NULL
            """, (content_type,))
            result = []
            for label_str, metrics_json in cursor.fetchall():
                result.append((label_str, json.loads(metrics_json)))
            return result
        finally:
            conn.close()


class DocumentTextRepository:
    """Repository for document full text storage (deferred scoring)."""

    def __init__(self, database=None):
        self._db = database or _default_db

    def insert(self, doc_id: str, text: str, conn=None) -> None:
        """Insert a single document text."""
        owns_conn = conn is None
        if owns_conn:
            conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO document_texts (doc_id, text) VALUES (?, ?)",
                (doc_id, text),
            )
            if owns_conn:
                conn.commit()
        except Exception:
            if owns_conn:
                conn.rollback()
            raise
        finally:
            if owns_conn:
                conn.close()

    def insert_batch(self, items: List[tuple], conn=None) -> None:
        """Insert multiple (doc_id, text) tuples in one transaction."""
        owns_conn = conn is None
        if owns_conn:
            conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO document_texts (doc_id, text) VALUES (?, ?)",
                items,
            )
            if owns_conn:
                conn.commit()
        except Exception:
            if owns_conn:
                conn.rollback()
            raise
        finally:
            if owns_conn:
                conn.close()

    def get_text(self, doc_id: str) -> Optional[str]:
        """Get text for a single document."""
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM document_texts WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def get_unscored_batch(self, batch_size: int = 1000) -> List[tuple]:
        """Get batch of (doc_id, text) for documents pending scoring."""
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT dt.doc_id, dt.text
                FROM document_texts dt
                JOIN documents d ON d.id = dt.doc_id
                WHERE d.quality_profile_used = 'pending'
                LIMIT ?
            """, (batch_size,))
            return cursor.fetchall()
        finally:
            conn.close()


class JobRepository:
    """Replaces raw SQL in producer.py."""

    def __init__(self, database=None):
        self._db = database or _default_db

    def register_job(self, job: DumpJob) -> None:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dump_processing_jobs
                (id, dump_name, dump_path, status, started_at)
                VALUES (?, ?, ?, 'running', CURRENT_TIMESTAMP)
            """, (job.id, job.dump_name, job.dump_path))
            conn.commit()
        finally:
            conn.close()

    def finalize_job(
        self,
        job_id: str,
        status: str,
        total_urls: int,
        filtered_urls: int,
        error: Optional[str] = None,
    ) -> None:
        conn = self._db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE dump_processing_jobs
                SET status = ?,
                    total_urls = ?,
                    filtered_urls = ?,
                    completed_at = CURRENT_TIMESTAMP,
                    error_message = ?
                WHERE id = ?
            """, (status, total_urls, filtered_urls, error, job_id))
            conn.commit()
        finally:
            conn.close()
