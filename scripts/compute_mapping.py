#!/usr/bin/env python3
"""
Phase 2: Compute Mapping Script

Computes UMAP coordinates and HDBSCAN clusters for visualization.

All parameters read from config.json under "mapping" section.

Input:
    - Embeddings from Qdrant (mapping.input_source)
    - Document metadata from SQLite

Output:
    - Mapping file (mapping.output_path)
    - Updated cluster_id and importance_score in SQLite

Usage:
    python scripts/compute_mapping.py
    python scripts/compute_mapping.py --format json
    python scripts/compute_mapping.py --output data/mappings/custom.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging.logger import setup_logger, get_logger
from common.config import config
from common.qdrant_manager import QdrantManager
from common.repositories import DocumentRepository
from mapping import MappingPipeline
from mapping.projectors import UMAPProjector
from mapping.clusterers import HDBSCANClusterer

logger = get_logger("compute_mapping")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_embeddings(qdrant_mgr: QdrantManager, limit=None):
    """Load embeddings from Qdrant via scroll.

    Returns:
        Tuple of (doc_ids, embeddings_matrix)
    """
    logger.info("Loading embeddings from Qdrant...")

    doc_ids = []
    embeddings = []
    offset = None
    batch_size = 1000

    while True:
        points, offset = qdrant_mgr.scroll(
            offset=offset,
            limit=batch_size,
            with_vectors=True,
        )

        if not points:
            break

        for point in points:
            doc_ids.append(str(point.id))
            embeddings.append(np.array(point.vector))

            if limit and len(embeddings) >= limit:
                break

        if limit and len(embeddings) >= limit:
            break

        if offset is None:
            break

    logger.info(f"Loaded {len(embeddings)} embeddings from Qdrant")

    if not embeddings:
        return [], np.array([])

    return doc_ids, np.array(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Compute Mapping - UMAP projection and clustering"
    )

    # All params default to config.json values
    parser.add_argument(
        "--output",
        default=config.get("mapping.output_path"),
        help="Output mapping file path"
    )
    parser.add_argument(
        "--format",
        choices=['json', 'parquet', 'both'],
        default=config.get("mapping.output_format"),
        help="Output format"
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=config.get("mapping.umap.neighbors"),
        help="UMAP neighbors parameter"
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=config.get("mapping.umap.min_dist"),
        help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=config.get("mapping.hdbscan.min_cluster_size"),
        help="HDBSCAN min_cluster_size"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute even if mapping exists"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit documents to map (for testing)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger("compute_mapping", console_output=True)

    logger.info("=" * 60)
    logger.info("PHASE 2: COMPUTE MAPPING")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output}")
    logger.info(f"Format: {args.format}")
    logger.info(f"UMAP: neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}")
    logger.info(f"HDBSCAN: min_cluster_size={args.min_cluster_size}")
    logger.info("=" * 60)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize components
    qdrant_mgr = QdrantManager()
    doc_repo = DocumentRepository()
    pipeline = MappingPipeline(
        projector=UMAPProjector(
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        ),
        clusterer=HDBSCANClusterer(
            min_cluster_size=args.min_cluster_size,
        ),
    )

    # Check Qdrant
    if not qdrant_mgr.available:
        logger.error("Qdrant not available. Run batch processing first or start Qdrant.")
        sys.exit(1)

    # Load embeddings
    start_time = time.time()
    try:
        doc_ids, embeddings = load_embeddings(qdrant_mgr, limit=args.limit)
        if len(doc_ids) == 0:
            logger.error("No embeddings found in Qdrant. Run batch processing first.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        sys.exit(1)

    # 1. UMAP projection
    logger.info(f"Computing UMAP projection for {len(doc_ids)} documents...")
    coordinates = pipeline.project(embeddings, force_refit=args.recompute)

    # 2. HDBSCAN clustering
    logger.info("Computing HDBSCAN clusters...")
    cluster_labels = pipeline.cluster(coordinates)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_orphaned = int(np.sum(cluster_labels == -1))

    # 3. Importance scores
    logger.info("Computing importance scores...")
    importance_scores = []
    for doc_id in doc_ids:
        row = doc_repo.get_metadata_for_mapping(doc_id)
        if row:
            domain, quality_score, content_type = row
            importance = pipeline.score_importance(
                domain=domain or 'unknown',
                quality_score=quality_score or 0.5,
                content_type=content_type,
            )
        else:
            importance = 0.5
        importance_scores.append(importance)
    importance_scores = np.array(importance_scores)

    # 4. Update database
    logger.info("Updating database with mapping coordinates...")
    updates = [
        (int(cluster_labels[i]), float(importance_scores[i]), doc_ids[i])
        for i in range(len(doc_ids))
    ]
    doc_repo.update_mappings_batch(updates)
    logger.info(f"Updated {len(doc_ids)} documents in database")

    # 5. Fetch quality scores for export
    quality_scores = []
    for doc_id in doc_ids:
        qs = doc_repo.get_quality_score(doc_id)
        quality_scores.append(qs if qs is not None else 0.5)

    elapsed = time.time() - start_time

    # Stats
    stats = {
        'documents_mapped': len(doc_ids),
        'clusters_found': n_clusters,
        'orphaned_docs': n_orphaned,
        'mapping_time_seconds': round(elapsed, 1),
    }

    # Export
    if args.format in ['json', 'both']:
        json_path = args.output if args.output.endswith('.json') else str(output_path.with_suffix('.json'))

        # Build export data with document metadata
        export_data = []
        for i, doc_id in enumerate(doc_ids):
            entry = {
                'doc_id': doc_id,
                'x': float(coordinates[i, 0]),
                'y': float(coordinates[i, 1]),
                'z': float(importance_scores[i]),
                'cluster_id': int(cluster_labels[i]),
                'quality_score': float(quality_scores[i]),
            }
            fields = doc_repo.get_export_fields(doc_id)
            if fields:
                entry.update(fields)
            export_data.append(entry)

        with open(json_path, 'w') as f:
            json.dump({
                'mappings': export_data,
                'stats': stats,
                'generated_at': datetime.now().isoformat(),
            }, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Exported {len(export_data)} mappings to {json_path}")

    if args.format in ['parquet', 'both']:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            parquet_path = args.output if args.output.endswith('.parquet') else str(output_path.with_suffix('.parquet'))
            data = {
                'doc_id': doc_ids,
                'x': coordinates[:, 0].tolist(),
                'y': coordinates[:, 1].tolist(),
                'z': [float(s) for s in importance_scores],
                'cluster_id': [int(l) for l in cluster_labels],
                'quality_score': quality_scores,
            }
            table = pa.table(data)
            pq.write_table(table, parquet_path)
            logger.info(f"Exported {len(doc_ids)} mappings to {parquet_path}")
        except ImportError:
            logger.warning("pyarrow not installed, skipping Parquet export")

    # Summary
    print("\n" + "=" * 60)
    print("MAPPING COMPLETE")
    print("=" * 60)
    print(f"Documents mapped: {stats['documents_mapped']}")
    print(f"Clusters found: {stats['clusters_found']}")
    print(f"Orphaned docs: {stats['orphaned_docs']}")
    print(f"Time: {stats['mapping_time_seconds']:.1f}s")
    print("=" * 60)
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
