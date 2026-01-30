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

from src.logging.logger import get_logger
from common.config import config
from common.database import db
from storage.parquet_store import ParquetStore

logger = get_logger("atlas_store")


class AtlasStore:
    """
    Unified storage interface for the Atlas.
    
    Abstracts over SQLite, Qdrant, and Parquet storage.
    """
    
    def __init__(self):
        self.db_path = config.get("database.sqlite_path", "data/atlas.db")
        self.parquet_store = ParquetStore()
        
        # Qdrant client (lazy init)
        self._qdrant = None
        self._qdrant_collection = config.get("qdrant.collection", "atlas_embeddings")
        
        logger.info("AtlasStore initialized")
    
    @property
    def qdrant(self):
        """Lazy-initializes Qdrant client."""
        if self._qdrant is None:
            try:
                from qdrant_client import QdrantClient
                url = config.get("qdrant.url", "http://localhost:6333")
                self._qdrant = QdrantClient(url=url, timeout=5)
                
                # Verify connection
                self._qdrant.get_collection(self._qdrant_collection)
                
            except Exception as e:
                logger.warning(f"Qdrant not available: {e}")
                self._qdrant = None
        
        return self._qdrant
    
    def get_connection(self) -> sqlite3.Connection:
        """Gets SQLite connection."""
        return db.get_connection()
    
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
        
        conn = self.get_connection()
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
        
        return {
            'id': row[0],
            'url': row[1],
            'title': row[2],
            'summary': row[3],
            'domain': row[4],
            'content_type': row[5],
            'quality_score': row[6],
            'wilson_score': row[7],
            'importance_score': row[8],
            'cluster_id': row[9],
            'created_at': row[10],
        }
    
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
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT id, canonical_url, title, domain,
                   detected_content_type, quality_score, wilson_score,
                   importance_score, cluster_id, created_at
            FROM documents
            WHERE status = 'active'
        """
        params = []
        
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
        
        documents = []
        for row in cursor.fetchall():
            documents.append({
                'id': row[0],
                'url': row[1],
                'title': row[2],
                'domain': row[3],
                'content_type': row[4],
                'quality_score': row[5],
                'wilson_score': row[6],
                'importance_score': row[7],
                'cluster_id': row[8],
                'created_at': row[9],
            })
        
        return documents
    
    def get_document_count(
        self,
        content_type: Optional[str] = None,
        min_quality: Optional[float] = None
    ) -> int:
        """Gets total document count with optional filters."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT COUNT(*) FROM documents WHERE status = 'active'"
        params = []
        
        if content_type:
            query += " AND detected_content_type = ?"
            params.append(content_type)
        
        if min_quality is not None:
            query += " AND quality_score >= ?"
            params.append(min_quality)
        
        cursor.execute(query, params)
        return cursor.fetchone()[0]
    
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
        
        if not self.qdrant:
            logger.warning("Qdrant not available for similarity search")
            return []
        
        try:
            results = self.qdrant.search(
                collection_name=self._qdrant_collection,
                query_vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                limit=k
            )
            
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
        if not self.qdrant:
            return None
        
        try:
            result = self.qdrant.retrieve(
                collection_name=self._qdrant_collection,
                ids=[doc_id],
                with_vectors=True
            )
            
            if result:
                return np.array(result[0].vector)
            
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
        
        return None
    
    # ==================== Cluster Operations ====================
    
    def get_cluster_stats(self) -> List[Dict[str, Any]]:
        """Gets statistics for all clusters."""
        conn = self.get_connection()
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
        
        clusters = []
        for row in cursor.fetchall():
            clusters.append({
                'cluster_id': row[0],
                'doc_count': row[1],
                'avg_quality': row[2],
                'min_quality': row[3],
                'max_quality': row[4],
                'is_orphaned': row[0] == -1,
            })
        
        return clusters
    
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
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Document counts
        cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'active'")
        total_docs = cursor.fetchone()[0]
        
        # Content type distribution
        cursor.execute("""
            SELECT detected_content_type, COUNT(*) 
            FROM documents 
            WHERE status = 'active'
            GROUP BY detected_content_type
        """)
        content_types = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Quality distribution
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN quality_score >= 0.7 THEN 1 END) as high,
                COUNT(CASE WHEN quality_score >= 0.4 AND quality_score < 0.7 THEN 1 END) as medium,
                COUNT(CASE WHEN quality_score < 0.4 THEN 1 END) as low
            FROM documents WHERE status = 'active'
        """)
        quality_dist = cursor.fetchone()
        
        # Cluster count
        cursor.execute("""
            SELECT COUNT(DISTINCT cluster_id) 
            FROM documents 
            WHERE status = 'active' AND cluster_id != -1
        """)
        cluster_count = cursor.fetchone()[0]
        
        # Qdrant stats
        qdrant_count = 0
        if self.qdrant:
            try:
                info = self.qdrant.get_collection(self._qdrant_collection)
                qdrant_count = info.points_count
            except Exception:
                pass
        
        return {
            'total_documents': total_docs,
            'vector_count': qdrant_count,
            'cluster_count': cluster_count,
            'content_type_distribution': content_types,
            'quality_distribution': {
                'high': quality_dist[0] if quality_dist else 0,
                'medium': quality_dist[1] if quality_dist else 0,
                'low': quality_dist[2] if quality_dist else 0,
            },
            'storage': self.parquet_store.get_stats(),
        }
