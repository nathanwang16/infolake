"""Abstract base classes for the mapping system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np


class Projector(ABC):
    """Protocol for dimensionality reduction / 2D projection."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique projector name (e.g. 'umap')."""
        ...

    @property
    def n_components(self) -> int:
        return 2

    @abstractmethod
    def fit(self, embeddings: np.ndarray, **kwargs) -> None:
        """Fit the model on *embeddings*."""
        ...

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project *embeddings* to low-dimensional space."""
        ...

    def fit_transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Convenience: fit then transform."""
        self.fit(embeddings, **kwargs)
        return self.transform(embeddings)


class Clusterer(ABC):
    """Protocol for clustering projected coordinates."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit_predict(self, coordinates: np.ndarray) -> np.ndarray:
        """Return integer cluster labels (-1 = noise)."""
        ...


class AxisScorer(ABC):
    """Protocol for computing a per-document importance/Z-axis score."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute(
        self,
        domain: str,
        quality_score: float,
        content_type: Optional[str] = None,
        inbound_links: Optional[int] = None,
        citations: Optional[int] = None,
    ) -> float:
        """Return an importance score in [0, 1]."""
        ...
