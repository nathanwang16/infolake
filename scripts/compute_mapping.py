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
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging.logger import setup_logger, get_logger
from common.config import config

logger = get_logger("compute_mapping")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Compute Mapping - UMAP projection and clustering"
    )
    
    # All params default to config.json values
    parser.add_argument(
        "--output",
        default=config.get("mapping.output_path", "./data/mappings/latest.json"),
        help="Output mapping file path"
    )
    parser.add_argument(
        "--format",
        choices=['json', 'parquet', 'both'],
        default=config.get("mapping.output_format", "json"),
        help="Output format"
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=config.get("mapping.umap.neighbors", 15),
        help="UMAP neighbors parameter"
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=config.get("mapping.umap.min_dist", 0.1),
        help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=config.get("mapping.hdbscan.min_cluster_size", 5),
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
    
    # Import here to avoid slow startup
    from mapping.mapper import AtlasMapper
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize mapper
    mapper = AtlasMapper(
        output_dir=str(output_path.parent),
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster_size=args.min_cluster_size
    )
    
    # Check Qdrant
    if not mapper.qdrant:
        logger.error("Qdrant not available. Run batch processing first or start Qdrant.")
        sys.exit(1)
    
    # Load embeddings
    try:
        count = mapper.load_embeddings_from_qdrant(limit=args.limit)
        if count == 0:
            logger.error("No embeddings found in Qdrant. Run batch processing first.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        sys.exit(1)
    
    # Run mapping
    mappings = mapper.run_full_mapping(recompute=args.recompute)
    
    if not mappings:
        logger.warning("No mappings generated")
        sys.exit(1)
    
    # Update database
    mapper.update_database_coordinates()
    
    # Export
    if args.format in ['json', 'both']:
        json_path = args.output if args.output.endswith('.json') else str(output_path.with_suffix('.json'))
        mapper.export_to_json(json_path)
    
    if args.format in ['parquet', 'both']:
        try:
            parquet_path = args.output if args.output.endswith('.parquet') else str(output_path.with_suffix('.parquet'))
            mapper.export_to_parquet(parquet_path)
        except ImportError:
            logger.warning("pyarrow not installed, skipping Parquet export")
    
    # Summary
    stats = mapper.get_stats()
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
