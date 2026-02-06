"""
Unified storage layer for Atlas data.

Provides a single interface for:
- SQLite (metadata, documents, golden set)
- Qdrant (embeddings, vector search)
- Parquet (exports, mappings)

Usage:
    store = AtlasStore()

    # Get documents
    docs = store.get_documents(limit=100)

    # Search by embedding
    results = store.search_similar(embedding, k=10)

    # Export for visualization
    store.export_for_visualization("output/")
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from common.logging.logger import get_logger
from common.config import config
from common.database import db
from common.repositories import DocumentRepository
from common.qdrant_manager import QdrantManager
from storage.parquet_store import ParquetStore

logger = get_logger("atlas_store")


class AtlasStore:
    """
    Unified storage interface for the Atlas.

    Abstracts over SQLite, Qdrant, and Parquet storage.
    """

    def __init__(self, database=None, qdrant_manager=None, doc_repo=None):
        self._database = database or db
        self._qdrant_mgr = qdrant_manager or QdrantManager()
        self._doc_repo = doc_repo or DocumentRepository(self._database)
        self.parquet_store = ParquetStore()

        logger.info("AtlasStore initialized")

    @property
    def qdrant(self):
        """Backwards-compatible: returns QdrantClient or None."""
        return self._qdrant_mgr.client

    def get_connection(self) -> sqlite3.Connection:
        """Gets SQLite connection."""
        return self._database.get_connection()

    # ==================== Document Operations ====================

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets a single document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document dictionary or None
        """
        if doc_id is None:
            raise ValueError("doc_id is required")

        doc = self._doc_repo.get_by_id(doc_id)
        if not doc:
            return None
        return doc.to_dict()

    def get_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        content_type: Optional[str] = None,
        min_quality: Optional[float] = None,
        cluster_id: Optional[int] = None,
        order_by: str = 'quality_score DESC'
    ) -> List[Dict[str, Any]]:
        """
        Gets documents with filtering and pagination.

        Args:
            limit: Maximum documents to return
            offset: Pagination offset
            content_type: Filter by content type
            min_quality: Minimum quality score
            cluster_id: Filter by cluster
            order_by: Sort order

        Returns:
            List of document dictionaries
        """
        items = self._doc_repo.get_list(
            limit=limit,
            offset=offset,
            content_type=content_type,
            min_quality=min_quality,
            cluster_id=cluster_id,
            order_by=order_by,
        )
        return [item.to_dict() for item in items]

    def get_document_count(
        self,
        content_type: Optional[str] = None,
        min_quality: Optional[float] = None
    ) -> int:
        """Gets total document count with optional filters."""
        return self._doc_repo.get_count(
            content_type=content_type,
            min_quality=min_quality,
        )

    # ==================== Vector Operations ====================

    def search_similar(
        self,
        embedding: np.ndarray,
        k: int = 10,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar documents by embedding.

        Args:
            embedding: Query embedding vector
            k: Number of results
            min_score: Minimum similarity score

        Returns:
            List of results with id, score, and metadata
        """
        if embedding is None:
            raise ValueError("embedding is required")

        if not self._qdrant_mgr.available:
            logger.warning("Qdrant not available for similarity search")
            return []

        try:
            query_vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            results = self._qdrant_mgr.search(query_vector=query_vector, limit=k)

            search_results = []
            for r in results:
                if r.score >= min_score:
                    # Fetch full document from SQLite
                    doc = self.get_document(str(r.id))
                    if doc:
                        doc['similarity_score'] = r.score
                        search_results.append(doc)

            return search_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Gets embedding for a document."""
        if not self._qdrant_mgr.available:
            return None

        try:
            result = self._qdrant_mgr.retrieve(ids=[doc_id], with_vectors=True)
            if result:
                return np.array(result[0].vector)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")

        return None

    # ==================== Cluster Operations ====================

    def get_cluster_stats(self) -> List[Dict[str, Any]]:
        """Gets statistics for all clusters."""
        clusters = self._doc_repo.get_cluster_stats()
        return [c.to_dict() for c in clusters]

    def get_cluster_documents(
        self,
        cluster_id: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Gets documents in a specific cluster."""
        return self.get_documents(
            cluster_id=cluster_id,
            limit=limit,
            order_by='quality_score DESC'
        )

    # ==================== Search Operations ====================

    def search_text_documents(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Text search across document titles and URLs."""
        items = self._doc_repo.search_text(query, limit)
        return [item.to_dict() for item in items]

    # ==================== Export Operations ====================

    def export_for_visualization(
        self,
        output_dir: str,
        include_mappings: bool = True,
        include_documents: bool = True
    ) -> Dict[str, str]:
        """
        Exports data for visualization.

        Args:
            output_dir: Output directory
            include_mappings: Export coordinate mappings
            include_documents: Export document metadata

        Returns:
            Dictionary of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exports = {}

        if include_mappings:
            # Load latest mappings and save
            mappings = self.parquet_store.load_mappings()
            if mappings:
                filepath = output_dir / "mappings.json"
                with open(filepath, 'w') as f:
                    json.dump(mappings, f)
                exports['mappings'] = str(filepath)

        if include_documents:
            # Export document metadata
            docs = self.get_documents(limit=100000)  # Reasonable limit
            filepath = output_dir / "documents.json"
            with open(filepath, 'w') as f:
                json.dump(docs, f)
            exports['documents'] = str(filepath)

        # Export cluster stats
        clusters = self.get_cluster_stats()
        filepath = output_dir / "clusters.json"
        with open(filepath, 'w') as f:
            json.dump(clusters, f)
        exports['clusters'] = str(filepath)

        logger.info(f"Exported visualization data to {output_dir}")
        return exports

    def get_atlas_stats(self) -> Dict[str, Any]:
        """Gets overall atlas statistics."""
        summary = self._doc_repo.get_atlas_summary_stats()

        # Qdrant stats
        qdrant_count = 0
        info = self._qdrant_mgr.get_collection_info()
        if info:
            try:
                qdrant_count = info.points_count
            except Exception:
                pass

        return {
            'total_documents': summary['total_docs'],
            'vector_count': qdrant_count,
            'cluster_count': summary['cluster_count'],
            'content_type_distribution': summary['content_types'],
            'quality_distribution': summary['quality_dist'],
            'storage': self.parquet_store.get_stats(),
        }
