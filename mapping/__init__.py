"""
Mapping module for Truth Atlas.

Computes visualization coordinates and diversity metrics:
- Semantic projection (UMAP)
- Importance scoring (Z-axis)
- Topic clustering (HDBSCAN)
"""

from mapping.mapper import AtlasMapper, MappingResult

__all__ = ['AtlasMapper', 'MappingResult']
