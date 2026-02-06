"""
Parquet storage for Atlas mappings and exports.

Provides efficient columnar storage for:
- UMAP coordinates
- Cluster assignments
- Document metadata snapshots

Usage:
    store = ParquetStore("data/exports")
    store.save_mappings(mappings)
    mappings = store.load_latest_mappings()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from common.logging.logger import get_logger
from common.config import config

logger = get_logger("parquet_store")


class ParquetStore:
    """
    Parquet-based storage for atlas mappings and exports.
    
    Structure:
        exports/
        ├── mappings/
        │   ├── mappings_20250129_120000.parquet
        │   └── latest -> mappings_20250129_120000.parquet
        ├── documents/
        │   └── documents_export_20250129.parquet
        └── manifest.json
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or config.get("paths.data_dir")) / "exports"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.mappings_dir = self.base_dir / "mappings"
        self.documents_dir = self.base_dir / "documents"
        
        self.mappings_dir.mkdir(exist_ok=True)
        self.documents_dir.mkdir(exist_ok=True)
        
        # Manifest tracking
        self.manifest_path = self.base_dir / "manifest.json"
        self._manifest = self._load_manifest()
        
        logger.info(f"ParquetStore initialized at {self.base_dir}")
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Loads or creates manifest file."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            'created_at': datetime.now().isoformat(),
            'mappings': [],
            'documents': [],
        }
    
    def _save_manifest(self):
        """Saves manifest file."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2)
    
    def save_mappings(
        self,
        mappings: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Saves mappings to Parquet file.
        
        Args:
            mappings: List of mapping dictionaries with x, y, z, cluster_id, etc.
            metadata: Optional metadata to store with export
            
        Returns:
            Path to saved file
        """
        if mappings is None:
            raise ValueError("mappings is required")
        
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow not installed. Run: pip install pyarrow")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.mappings_dir / f"mappings_{timestamp}.parquet"
        
        # Build column data
        columns = {}
        if mappings:
            for key in mappings[0].keys():
                columns[key] = [m.get(key) for m in mappings]
        
        table = pa.table(columns)
        pq.write_table(table, filepath)
        
        # Update manifest
        entry = {
            'filename': filepath.name,
            'created_at': datetime.now().isoformat(),
            'doc_count': len(mappings),
            'metadata': metadata or {},
        }
        self._manifest['mappings'].append(entry)
        self._manifest['latest_mappings'] = filepath.name
        self._save_manifest()
        
        logger.info(f"Saved {len(mappings)} mappings to {filepath}")
        return str(filepath)
    
    def load_mappings(self, filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Loads mappings from Parquet file.
        
        Args:
            filename: Specific file to load, or None for latest
            
        Returns:
            List of mapping dictionaries
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow not installed. Run: pip install pyarrow")
        
        if filename is None:
            filename = self._manifest.get('latest_mappings')
        
        if filename is None:
            logger.warning("No mappings file available")
            return []
        
        filepath = self.mappings_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Mappings file not found: {filepath}")
        
        table = pq.read_table(filepath)
        df = table.to_pandas()
        
        mappings = df.to_dict('records')
        logger.info(f"Loaded {len(mappings)} mappings from {filepath}")
        
        return mappings
    
    def save_documents_export(
        self,
        documents: List[Dict[str, Any]],
        include_embeddings: bool = False
    ) -> str:
        """
        Exports documents to Parquet for analysis.
        
        Args:
            documents: List of document dictionaries
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            Path to saved file
        """
        if documents is None:
            raise ValueError("documents is required")
        
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow not installed")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.documents_dir / f"documents_{timestamp}.parquet"
        
        # Filter out embeddings if not requested (they're large)
        if not include_embeddings:
            documents = [
                {k: v for k, v in doc.items() if k != 'embedding'}
                for doc in documents
            ]
        
        # Build table
        columns = {}
        if documents:
            for key in documents[0].keys():
                columns[key] = [d.get(key) for d in documents]
        
        table = pa.table(columns)
        pq.write_table(table, filepath)
        
        # Update manifest
        entry = {
            'filename': filepath.name,
            'created_at': datetime.now().isoformat(),
            'doc_count': len(documents),
            'includes_embeddings': include_embeddings,
        }
        self._manifest['documents'].append(entry)
        self._save_manifest()
        
        logger.info(f"Saved {len(documents)} documents to {filepath}")
        return str(filepath)
    
    def list_exports(self) -> Dict[str, List[Dict]]:
        """Lists all available exports."""
        return {
            'mappings': self._manifest.get('mappings', []),
            'documents': self._manifest.get('documents', []),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns storage statistics."""
        mapping_files = list(self.mappings_dir.glob("*.parquet"))
        document_files = list(self.documents_dir.glob("*.parquet"))
        
        total_size = sum(f.stat().st_size for f in mapping_files + document_files)
        
        return {
            'mapping_files': len(mapping_files),
            'document_files': len(document_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'latest_mappings': self._manifest.get('latest_mappings'),
        }
