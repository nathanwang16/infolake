"""Built-in projector implementations."""

import time
from typing import Optional

import numpy as np

from common.logging.logger import get_logger
from mapping.protocols import Projector

logger = get_logger("mapper")


class UMAPProjector(Projector):
    """UMAP-based 2D projection (extracted from AtlasMapper)."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        sample_for_fit: int = 100000,
    ):
        self._n_neighbors = n_neighbors
        self._min_dist = min_dist
        self._sample_for_fit = sample_for_fit
        self._model = None

    @property
    def name(self) -> str:
        return "umap"

    def fit(self, embeddings: np.ndarray, **kwargs) -> None:
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn not installed. Run: pip install umap-learn")

        n_docs = len(embeddings)
        logger.info(f"Fitting UMAP on {n_docs} documents...")
        start = time.time()

        if n_docs > self._sample_for_fit:
            sample_idx = np.random.choice(n_docs, self._sample_for_fit, replace=False)
            fit_data = embeddings[sample_idx]
            neighbors = self._n_neighbors
        else:
            fit_data = embeddings
            neighbors = min(self._n_neighbors, n_docs - 1)

        self._model = umap.UMAP(
            n_neighbors=neighbors,
            min_dist=self._min_dist,
            metric='cosine',
            n_components=self.n_components,
            random_state=42,
        )
        self._model.fit(fit_data)
        logger.info(f"UMAP fit complete in {time.time() - start:.1f}s")

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Projector not fitted. Call fit() first.")
        return self._model.transform(embeddings)
