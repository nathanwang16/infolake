"""
Hypergraph construction and query via sparse incidence matrix.

A hypergraph connects *sets* of nodes (URLs) through hyperedges (topics,
domains, date-bins).  Physically it is a sparse incidence matrix H where
rows = documents and columns = attribute labels.

Composite tensor:
    H = [ H_topics (Sparse) | H_domains (Sparse) | H_dates (Dense) | H_embeddings (Dense) ]

All heavy operations are expressed as tensor linear algebra so they run
on GPU.
"""

from typing import Dict, List, Optional, Tuple

import torch

from atlas_core.config import config
from atlas_core.errors import AtlasHypergraphError
from atlas_core.logging import get_logger
from atlas_core.types import DocumentID, HyperedgeLabel

logger = get_logger("atlas_core.hypergraph")


class Hypergraph:
    """
    Sparse incidence matrix backed by PyTorch sparse tensors.

    Parameters
    ----------
    n_docs : int
        Number of documents (rows).
    """

    def __init__(self, n_docs: int) -> None:
        if n_docs < 1:
            raise AtlasHypergraphError(f"n_docs must be >= 1, got {n_docs}")

        self.n_docs = n_docs

        # Sparse sub-matrices — built incrementally via add_* methods
        self._topic_indices: List[Tuple[int, int]] = []
        self._topic_labels: List[HyperedgeLabel] = []
        self._topic_label_to_col: Dict[HyperedgeLabel, int] = {}

        self._domain_indices: List[Tuple[int, int]] = []
        self._domain_labels: List[HyperedgeLabel] = []
        self._domain_label_to_col: Dict[HyperedgeLabel, int] = {}

        # Dense sub-matrices — set in bulk
        self._date_matrix: Optional[torch.Tensor] = None   # (N, date_bins)
        self._embeddings: Optional[torch.Tensor] = None     # (N, D)

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add_topic(self, doc_idx: int, label: HyperedgeLabel) -> None:
        """Assign document *doc_idx* to topic hyperedge *label*."""
        if doc_idx < 0 or doc_idx >= self.n_docs:
            raise AtlasHypergraphError(f"doc_idx {doc_idx} out of range [0, {self.n_docs})")
        if label not in self._topic_label_to_col:
            col = len(self._topic_labels)
            self._topic_labels.append(label)
            self._topic_label_to_col[label] = col
        self._topic_indices.append((doc_idx, self._topic_label_to_col[label]))

    def add_domain(self, doc_idx: int, label: HyperedgeLabel) -> None:
        """Assign document *doc_idx* to domain hyperedge *label*."""
        if doc_idx < 0 or doc_idx >= self.n_docs:
            raise AtlasHypergraphError(f"doc_idx {doc_idx} out of range [0, {self.n_docs})")
        if label not in self._domain_label_to_col:
            col = len(self._domain_labels)
            self._domain_labels.append(label)
            self._domain_label_to_col[label] = col
        self._domain_indices.append((doc_idx, self._domain_label_to_col[label]))

    def set_dates(self, date_matrix: torch.Tensor) -> None:
        """
        Set the dense date-bin sub-matrix.

        Args:
            date_matrix: (N, B) float32 where B = number of date bins.
        """
        if date_matrix.shape[0] != self.n_docs:
            raise AtlasHypergraphError(
                f"date_matrix rows ({date_matrix.shape[0]}) != n_docs ({self.n_docs})"
            )
        self._date_matrix = date_matrix.float()

    def set_embeddings(self, embeddings: torch.Tensor) -> None:
        """
        Set the dense embedding sub-matrix.

        Args:
            embeddings: (N, D) float32.
        """
        if embeddings.shape[0] != self.n_docs:
            raise AtlasHypergraphError(
                f"embeddings rows ({embeddings.shape[0]}) != n_docs ({self.n_docs})"
            )
        self._embeddings = embeddings.float()

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def build_topic_matrix(self, device: torch.device) -> torch.Tensor:
        """Return sparse (N, T) topic incidence matrix on *device*."""
        n_topics = len(self._topic_labels)
        if n_topics == 0:
            return torch.sparse_coo_tensor(
                size=(self.n_docs, 0), device=device
            ).float()

        rows, cols = zip(*self._topic_indices)
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(rows), dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, size=(self.n_docs, n_topics)
        ).coalesce().to(device)

    def build_domain_matrix(self, device: torch.device) -> torch.Tensor:
        """Return sparse (N, D_dom) domain incidence matrix on *device*."""
        n_domains = len(self._domain_labels)
        if n_domains == 0:
            return torch.sparse_coo_tensor(
                size=(self.n_docs, 0), device=device
            ).float()

        rows, cols = zip(*self._domain_indices)
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(rows), dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, size=(self.n_docs, n_domains)
        ).coalesce().to(device)

    def build_composite(self, device: torch.device) -> torch.Tensor:
        """
        Build the full composite incidence tensor on *device*.

        Concatenates: [H_topics | H_domains | H_dates | H_embeddings]

        The sparse sub-matrices are converted to dense for concatenation.
        For corpora that exceed VRAM, callers should work with the
        individual sub-matrices instead.

        Returns:
            Dense (N, C) float32 tensor.

        Raises:
            AtlasHypergraphError: If embeddings have not been set.
        """
        if self._embeddings is None:
            raise AtlasHypergraphError("embeddings must be set before building composite")

        parts: List[torch.Tensor] = []

        H_topics = self.build_topic_matrix(device).to_dense()
        H_domains = self.build_domain_matrix(device).to_dense()
        parts.append(H_topics)
        parts.append(H_domains)

        if self._date_matrix is not None:
            parts.append(self._date_matrix.to(device))

        parts.append(self._embeddings.to(device))

        composite = torch.cat(parts, dim=1)
        logger.info(
            "Composite hypergraph: shape=%s (topics=%d, domains=%d, dates=%s, embed=%d)",
            tuple(composite.shape),
            H_topics.shape[1],
            H_domains.shape[1],
            self._date_matrix.shape[1] if self._date_matrix is not None else 0,
            self._embeddings.shape[1],
        )
        return composite

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query(
        self,
        composite: torch.Tensor,
        query_vector: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dot-product query against the composite tensor.

        Args:
            composite:    (N, C) composite incidence matrix.
            query_vector: (C,)   query vector.
            top_k:        Number of results to return.

        Returns:
            (top_k,) scores tensor, (top_k,) index tensor.

        Raises:
            AtlasHypergraphError: On dimension mismatch.
        """
        if composite.shape[1] != query_vector.shape[0]:
            raise AtlasHypergraphError(
                f"query dimension mismatch: composite has {composite.shape[1]} cols, "
                f"query has {query_vector.shape[0]} elements"
            )

        scores = composite @ query_vector       # (N,)
        k = min(top_k, scores.shape[0])
        top_scores, top_indices = scores.topk(k)
        return top_scores, top_indices

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_topics(self) -> int:
        return len(self._topic_labels)

    @property
    def n_domains(self) -> int:
        return len(self._domain_labels)

    @property
    def topic_labels(self) -> List[HyperedgeLabel]:
        return list(self._topic_labels)

    @property
    def domain_labels(self) -> List[HyperedgeLabel]:
        return list(self._domain_labels)

    def __repr__(self) -> str:
        return (
            f"<Hypergraph docs={self.n_docs} topics={self.n_topics} "
            f"domains={self.n_domains} has_dates={self._date_matrix is not None} "
            f"has_embeddings={self._embeddings is not None}>"
        )
