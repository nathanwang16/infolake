"""
atlas_core — minimal core library for Truth Atlas.

Every pipeline module imports from this package.  It provides:
- Domain types (DocumentID, URL, Embedding, etc.)
- Protocol definitions (Embedder, Scorer, Projector, etc.)
- GPU tensor operations (score, dedup, project, cluster, etc.)
- Hypergraph incidence matrix construction and query
- Base functor class for source adapters
- Singleton configuration loader
- Custom exception hierarchy
- Structured JSON logger

This package contains **zero** pipeline logic — only primitives and
contracts.
"""

# Errors — import first, no internal deps
from atlas_core.errors import (
    AtlasConfigError,
    AtlasEmbeddingError,
    AtlasError,
    AtlasFunctorError,
    AtlasHypergraphError,
    AtlasPipelineError,
    AtlasStorageError,
    AtlasTensorError,
)

# Logging
from atlas_core.logging import get_logger

# Configuration
from atlas_core.config import Config, config

# Domain types
from atlas_core.types import (
    ContentType,
    DocumentID,
    DocumentStatus,
    Embedding,
    HyperedgeLabel,
    QualityProfile,
    Record,
    ScoredDocument,
    SourcePhase,
    TensorCheckpoint,
    URL,
)

# Protocols
from atlas_core.protocols import (
    AxisScorer,
    Clusterer,
    Deduplicator,
    Embedder,
    Projector,
    Scorer,
    ScoringMetric,
    SourceFunctor,
)

# Tensor operations
from atlas_core import tensor_ops

# Hypergraph
from atlas_core.hypergraph import Hypergraph

# Functors
from atlas_core.functors import BaseFunctor

__all__ = [
    # Errors
    "AtlasError",
    "AtlasConfigError",
    "AtlasTensorError",
    "AtlasStorageError",
    "AtlasEmbeddingError",
    "AtlasPipelineError",
    "AtlasHypergraphError",
    "AtlasFunctorError",
    # Logging
    "get_logger",
    # Config
    "Config",
    "config",
    # Types
    "DocumentID",
    "URL",
    "Embedding",
    "HyperedgeLabel",
    "ContentType",
    "DocumentStatus",
    "QualityProfile",
    "SourcePhase",
    "Record",
    "ScoredDocument",
    "TensorCheckpoint",
    # Protocols
    "Embedder",
    "Scorer",
    "ScoringMetric",
    "Deduplicator",
    "Projector",
    "Clusterer",
    "AxisScorer",
    "SourceFunctor",
    # Tensor ops (namespace)
    "tensor_ops",
    # Hypergraph
    "Hypergraph",
    # Functors
    "BaseFunctor",
]
