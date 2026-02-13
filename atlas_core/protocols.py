"""
Protocol definitions for atlas_core.

All pipeline modules program against these protocols.  Implementations are
swappable — any object satisfying the structural contract is accepted.

Uses typing.Protocol (PEP 544) for structural subtyping: classes do NOT
need to inherit from these protocols, they just need matching signatures.
"""

from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable

import numpy as np
import torch

from atlas_core.types import ContentType, Record, URL


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    """Encodes a batch of texts into dense vectors on GPU."""

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode *texts* into an embedding matrix.

        Args:
            texts: List of document texts.

        Returns:
            Float32 tensor of shape (N, D) on the model device.
        """
        ...


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@runtime_checkable
class Scorer(Protocol):
    """Computes quality scores via weighted feature projection."""

    def score(self, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, F) feature matrix on GPU.
            weights:  (F,)   weight vector for a content type.

        Returns:
            (N,) score tensor.
        """
        ...


@runtime_checkable
class ScoringMetric(Protocol):
    """A single quality-scoring metric (runs on CPU, per-document)."""

    @property
    def name(self) -> str:
        """Unique metric identifier (e.g. ``'citation_quality'``)."""
        ...

    def compute(
        self,
        text: str,
        words: List[str],
        sentences: List[str],
        metadata: Dict[str, Any],
    ) -> float:
        """
        Compute metric value for one document.

        Returns:
            Score in [0.0, 1.0].
        """
        ...


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

@runtime_checkable
class Deduplicator(Protocol):
    """Identifies duplicate documents via embedding similarity."""

    def find_duplicates(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N, D) float32 tensor.

        Returns:
            Boolean mask of shape (N,) — True for duplicates to remove.
        """
        ...


# ---------------------------------------------------------------------------
# Projection / Clustering / Axis scoring
# ---------------------------------------------------------------------------

@runtime_checkable
class Projector(Protocol):
    """Dimensionality reduction to 2-D coordinates."""

    def fit_transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N, D) float32 tensor.

        Returns:
            (N, 2) float32 coordinate tensor.
        """
        ...


@runtime_checkable
class Clusterer(Protocol):
    """Clusters projected coordinates."""

    def fit_predict(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) float32 coordinates.

        Returns:
            (N,) int64 cluster labels (-1 = noise).
        """
        ...


@runtime_checkable
class AxisScorer(Protocol):
    """Computes per-document importance / Z-axis score."""

    def score(
        self,
        domain: str,
        quality_score: float,
        content_type: Optional[str] = None,
    ) -> float:
        """
        Returns:
            Importance in [0.0, 1.0].
        """
        ...


# ---------------------------------------------------------------------------
# Source adapters (functors)
# ---------------------------------------------------------------------------

@runtime_checkable
class SourceFunctor(Protocol):
    """
    Maps a heterogeneous data source into the atlas's uniform Record type.

    Each dump adapter implements this protocol.
    """

    def read(self, path: str) -> Iterator[Record]:
        """
        Yield Records from the source at *path*.

        Raises:
            AtlasFunctorError: On unrecoverable read failures.
        """
        ...
