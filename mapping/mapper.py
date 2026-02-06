"""
Atlas Mapper - Computes visualization coordinates and metrics.

Implements Section 6.4 of Technical Engineering Guide:
- Semantic mapping (UMAP with cosine metric)
- Importance scoring (domain authority for Z-axis)
- Topic clustering (HDBSCAN)
- Parquet export for coordinate persistence

Usage:
    python -m mapping.mapper --type all
    python -m mapping.mapper --type semantic --recompute
    python -m mapping.mapper --incremental
"""

import argparse
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from common.logging.logger import get_logger, setup_logger
from common.config import config
from common.database import db
from common.repositories import DocumentRepository
from common.qdrant_manager import QdrantManager
from mapping.axis_scorers import DomainAuthorityAxisScorer
from mapping.pipeline import MappingPipeline
from mapping.projectors import UMAPProjector
from mapping.clusterers import HDBSCANClusterer

logger = get_logger("mapper")


@dataclass
class MappingResult:
    """Result of a mapping operation."""
    doc_id: str
    x: float  # UMAP x-coordinate
    y: float  # UMAP y-coordinate
    z: float  # Importance score (domain authority)
    cluster_id: int  # HDBSCAN cluster (-1 = orphaned)
    quality_score: float

    def to_dict(self) -> Dict[str, Any]:
        # Convert numpy types to native Python types for JSON serialization
        return {
            'doc_id': str(self.doc_id),
            'x': float(self.x),
            'y': float(self.y),
            'z': float(self.z),
            'cluster_id': int(self.cluster_id),
            'quality_score': float(self.quality_score) if self.quality_score else 0.0,
        }


class ImportanceScorer:
    """
    Backward-compatible facade over DomainAuthorityAxisScorer.

    Importance = weighted combination of:
    - Domain authority (0.5 weight)
    - Inbound links (0.3 weight) - if available
    - Academic citations (0.2 weight) - if available

    For MVP: Uses domain-based heuristics when link data unavailable.
    """

    def __init__(self, axis_scorer: Optional[DomainAuthorityAxisScorer] = None):
        self._scorer = axis_scorer or DomainAuthorityAxisScorer()

    # Expose DOMAIN_AUTHORITY for backward compat
    @property
    def DOMAIN_AUTHORITY(self) -> Dict[str, float]:
        return self._scorer.DOMAIN_AUTHORITY

    @property
    def _cache(self) -> Dict[str, float]:
        return self._scorer._cache

    def compute_importance(
        self,
        domain: str,
        quality_score: float,
        content_type: Optional[str] = None,
        inbound_links: Optional[int] = None,
        citations: Optional[int] = None
    ) -> float:
        return self._scorer.compute(
            domain=domain,
            quality_score=quality_score,
            content_type=content_type,
            inbound_links=inbound_links,
            citations=citations,
        )

    def _get_domain_authority(self, domain: str) -> float:
        return self._scorer._get_domain_authority(domain)


class AtlasMapper:
    """
    Main mapper class for computing atlas coordinates and clusters.

    Implements the Mapper module from Technical Guide Section 6.4.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        hdbscan_min_cluster_size: int = 5,  # Lower for small test datasets
        sample_for_fit: int = 100000,
        database=None,
        qdrant_manager=None,
        doc_repo=None,
        pipeline: Optional[MappingPipeline] = None,
    ):
        self.output_dir = Path(output_dir or config.get("paths.data_dir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DI
        self._database = database or db
        self._qdrant_mgr = qdrant_manager or QdrantManager()
        self._doc_repo = doc_repo or DocumentRepository(self._database)

        # UMAP config (kept for backward compat)
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.sample_for_fit = sample_for_fit

        # HDBSCAN config (kept for backward compat)
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size

        # Pipeline — if not provided, build one from the per-component params
        if pipeline is not None:
            self._pipeline = pipeline
        else:
            self._pipeline = MappingPipeline(
                projector=UMAPProjector(
                    n_neighbors=umap_neighbors,
                    min_dist=umap_min_dist,
                    sample_for_fit=sample_for_fit,
                ),
                clusterer=HDBSCANClusterer(min_cluster_size=hdbscan_min_cluster_size),
                axis_scorer=DomainAuthorityAxisScorer(),
            )

        # Backward-compatible ImportanceScorer facade
        self.importance_scorer = ImportanceScorer(
            axis_scorer=self._pipeline.axis_scorer
            if isinstance(self._pipeline.axis_scorer, DomainAuthorityAxisScorer)
            else DomainAuthorityAxisScorer()
        )

        # State
        self._umap_model = None
        self._embeddings: List[np.ndarray] = []
        self._doc_ids: List[str] = []
        self._mappings: List[MappingResult] = []

        # Backwards-compatible properties
        self.qdrant = self._qdrant_mgr.client
        self.qdrant_collection = self._qdrant_mgr.collection_name

        # Stats
        self._stats = {
            'documents_mapped': 0,
            'clusters_found': 0,
            'orphaned_docs': 0,
            'mapping_time_seconds': 0,
        }

        logger.info(f"AtlasMapper initialized (neighbors={umap_neighbors}, "
                   f"min_dist={umap_min_dist}, min_cluster={hdbscan_min_cluster_size})")

    def load_embeddings_from_qdrant(self, limit: Optional[int] = None) -> int:
        """
        Loads embeddings from Qdrant.

        Args:
            limit: Maximum documents to load (None = all)

        Returns:
            Number of embeddings loaded
        """
        if not self._qdrant_mgr.available:
            raise RuntimeError("Qdrant client not available")

        logger.info("Loading embeddings from Qdrant...")

        try:
            # Scroll through all points
            offset = None
            batch_size = 1000

            while True:
                results = self._qdrant_mgr.scroll(
                    offset=offset,
                    limit=batch_size,
                    with_vectors=True
                )

                points, offset = results

                if not points:
                    break

                for point in points:
                    self._doc_ids.append(str(point.id))
                    self._embeddings.append(np.array(point.vector))

                    if limit and len(self._embeddings) >= limit:
                        break

                if limit and len(self._embeddings) >= limit:
                    break

                if offset is None:
                    break

            logger.info(f"Loaded {len(self._embeddings)} embeddings from Qdrant")
            return len(self._embeddings)

        except Exception as e:
            logger.error(f"Failed to load embeddings from Qdrant: {e}")
            raise

    def load_embeddings_from_array(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray
    ):
        """
        Loads embeddings from numpy arrays.

        Args:
            doc_ids: List of document IDs
            embeddings: Array of embeddings (n_docs x dim)
        """
        if doc_ids is None:
            raise ValueError("doc_ids is required")
        if embeddings is None:
            raise ValueError("embeddings is required")
        if len(doc_ids) != len(embeddings):
            raise ValueError("doc_ids and embeddings must have same length")

        self._doc_ids = list(doc_ids)
        self._embeddings = [embeddings[i] for i in range(len(embeddings))]

        logger.info(f"Loaded {len(self._embeddings)} embeddings from array")

    def compute_umap_projection(self, force_recompute: bool = False) -> np.ndarray:
        """
        Computes 2D UMAP projection of embeddings.

        Args:
            force_recompute: Recompute even if model exists

        Returns:
            Array of (x, y) coordinates
        """
        if not self._embeddings:
            raise RuntimeError("No embeddings loaded. Call load_embeddings_* first.")

        embeddings_matrix = np.array(self._embeddings)
        n_docs = len(embeddings_matrix)

        logger.info(f"Computing UMAP projection for {n_docs} documents...")
        start_time = time.time()

        coordinates = self._pipeline.project(embeddings_matrix, force_refit=force_recompute)

        elapsed = time.time() - start_time
        logger.info(f"UMAP projection complete in {elapsed:.1f}s")

        return coordinates

    def compute_clusters(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Computes clusters on UMAP coordinates.

        Args:
            coordinates: 2D coordinates from UMAP

        Returns:
            Array of cluster labels (-1 = orphaned/noise)
        """
        if coordinates is None:
            raise ValueError("coordinates is required")

        labels = self._pipeline.cluster(coordinates)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_orphaned = int(np.sum(labels == -1))

        self._stats['clusters_found'] = n_clusters
        self._stats['orphaned_docs'] = n_orphaned

        return labels

    def compute_importance_scores(self) -> List[float]:
        """
        Computes importance scores for all documents.

        Returns:
            List of importance scores (Z-axis values)
        """
        logger.info("Computing importance scores...")

        importance_scores = []

        for doc_id in self._doc_ids:
            row = self._doc_repo.get_metadata_for_mapping(doc_id)

            if row:
                domain, quality_score, content_type = row
                importance = self._pipeline.score_importance(
                    domain=domain or 'unknown',
                    quality_score=quality_score or 0.5,
                    content_type=content_type,
                )
            else:
                importance = 0.5  # Default for missing documents

            importance_scores.append(importance)

        logger.info(f"Computed {len(importance_scores)} importance scores")
        return importance_scores

    def run_full_mapping(self, recompute: bool = False) -> List[MappingResult]:
        """
        Runs the complete mapping pipeline.

        Args:
            recompute: Force recomputation even if cached

        Returns:
            List of MappingResult objects
        """
        start_time = time.time()

        # Load embeddings if not already loaded
        if not self._embeddings:
            if self._qdrant_mgr.available:
                self.load_embeddings_from_qdrant()
            else:
                raise RuntimeError("No embeddings loaded and Qdrant not available")

        if not self._embeddings:
            logger.warning("No embeddings to map")
            return []

        # 1. Compute UMAP projection
        coordinates = self.compute_umap_projection(force_recompute=recompute)

        # 2. Compute clusters
        cluster_labels = self.compute_clusters(coordinates)

        # 3. Compute importance scores
        importance_scores = self.compute_importance_scores()

        # 4. Fetch quality scores from database
        quality_scores = []
        for doc_id in self._doc_ids:
            qs = self._doc_repo.get_quality_score(doc_id)
            quality_scores.append(qs if qs is not None else 0.5)

        # 5. Build mapping results
        self._mappings = []
        for i, doc_id in enumerate(self._doc_ids):
            result = MappingResult(
                doc_id=doc_id,
                x=float(coordinates[i, 0]),
                y=float(coordinates[i, 1]),
                z=importance_scores[i],
                cluster_id=int(cluster_labels[i]),
                quality_score=quality_scores[i]
            )
            self._mappings.append(result)

        self._stats['documents_mapped'] = len(self._mappings)
        self._stats['mapping_time_seconds'] = time.time() - start_time

        logger.info(f"Full mapping complete: {len(self._mappings)} documents in "
                   f"{self._stats['mapping_time_seconds']:.1f}s")

        return self._mappings

    def update_database_coordinates(self):
        """Updates database with computed coordinates and cluster assignments."""
        if not self._mappings:
            raise RuntimeError("No mappings computed. Call run_full_mapping first.")

        logger.info("Updating database with mapping coordinates...")

        updates = [
            (mapping.cluster_id, mapping.z, mapping.doc_id)
            for mapping in self._mappings
        ]
        self._doc_repo.update_mappings_batch(updates)

        logger.info(f"Updated {len(self._mappings)} documents in database")

    def export_to_parquet(self, filepath: Optional[str] = None) -> str:
        """
        Exports mappings to Parquet file.

        Args:
            filepath: Output path (default: data/mappings_TIMESTAMP.parquet)

        Returns:
            Path to exported file
        """
        if not self._mappings:
            raise RuntimeError("No mappings to export. Call run_full_mapping first.")

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow not installed. Run: pip install pyarrow")

        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f"mappings_{timestamp}.parquet"
        else:
            filepath = Path(filepath)

        # Build table
        data = {
            'doc_id': [m.doc_id for m in self._mappings],
            'x': [m.x for m in self._mappings],
            'y': [m.y for m in self._mappings],
            'z': [m.z for m in self._mappings],
            'cluster_id': [m.cluster_id for m in self._mappings],
            'quality_score': [m.quality_score for m in self._mappings],
        }

        table = pa.table(data)
        pq.write_table(table, filepath)

        logger.info(f"Exported {len(self._mappings)} mappings to {filepath}")
        return str(filepath)

    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Exports mappings to JSON file (for visualization).

        Args:
            filepath: Output path

        Returns:
            Path to exported file
        """
        if not self._mappings:
            raise RuntimeError("No mappings to export. Call run_full_mapping first.")

        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f"mappings_{timestamp}.json"
        else:
            filepath = Path(filepath)

        # Add document metadata for visualization
        export_data = []
        for m in self._mappings:
            fields = self._doc_repo.get_export_fields(m.doc_id)

            doc_data = m.to_dict()
            if fields:
                doc_data.update(fields)

            export_data.append(doc_data)

        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(filepath, 'w') as f:
            json.dump({
                'mappings': export_data,
                'stats': self._stats,
                'generated_at': datetime.now().isoformat(),
            }, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Exported {len(self._mappings)} mappings to {filepath}")
        return str(filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Returns mapper statistics."""
        return self._stats.copy()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Atlas Mapper - Compute visualization coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full mapping
    python -m mapping.mapper --type all

    # Only compute UMAP projection
    python -m mapping.mapper --type semantic

    # Force recompute existing mappings
    python -m mapping.mapper --type all --recompute

    # Export to specific format
    python -m mapping.mapper --output mappings.parquet
"""
    )

    parser.add_argument(
        "--type",
        choices=['semantic', 'importance', 'clusters', 'all'],
        default='all',
        help="Mapping type to compute"
    )
    parser.add_argument(
        "--recompute",
        action='store_true',
        help="Force recompute even if mappings exist"
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: auto-generated)"
    )
    parser.add_argument(
        "--format",
        choices=['parquet', 'json', 'both'],
        default='both',
        help="Output format"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of documents to map"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger("mapper", console_output=True)

    # Run mapper
    mapper = AtlasMapper()

    try:
        if mapper._qdrant_mgr.available:
            mapper.load_embeddings_from_qdrant(limit=args.limit)
        else:
            logger.error("Qdrant not available and no embeddings provided")
            return 1

        mappings = mapper.run_full_mapping(recompute=args.recompute)

        if not mappings:
            logger.warning("No mappings generated")
            return 0

        # Update database
        mapper.update_database_coordinates()

        # Export
        if args.format in ['parquet', 'both']:
            output_path = args.output if args.output and args.output.endswith('.parquet') else None
            mapper.export_to_parquet(output_path)

        if args.format in ['json', 'both']:
            output_path = args.output if args.output and args.output.endswith('.json') else None
            mapper.export_to_json(output_path)

        # Print summary
        stats = mapper.get_stats()
        print("\n" + "=" * 60)
        print("MAPPING SUMMARY")
        print("=" * 60)
        print(f"Documents mapped: {stats['documents_mapped']}")
        print(f"Clusters found: {stats['clusters_found']}")
        print(f"Orphaned documents: {stats['orphaned_docs']}")
        print(f"Time: {stats['mapping_time_seconds']:.1f}s")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Mapping failed: {e}")
        raise


if __name__ == "__main__":
    main()
