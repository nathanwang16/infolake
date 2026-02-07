"""
Mapping module for Truth Atlas.

Computes visualization coordinates and diversity metrics:
- Semantic projection (UMAP)
- Importance scoring (Z-axis)
- Topic clustering (HDBSCAN)
"""

from mapping.protocols import Projector, Clusterer, AxisScorer
from mapping.pipeline import MappingPipeline
from mapping.registry import ComponentRegistry
from mapping.projectors import UMAPProjector
from mapping.clusterers import HDBSCANClusterer
from mapping.axis_scorers import DomainAuthorityAxisScorer

__all__ = [
    # Protocols
    'Projector',
    'Clusterer',
    'AxisScorer',
    # Pipeline
    'MappingPipeline',
    'ComponentRegistry',
    # Built-in components
    'UMAPProjector',
    'HDBSCANClusterer',
    'DomainAuthorityAxisScorer',
]
