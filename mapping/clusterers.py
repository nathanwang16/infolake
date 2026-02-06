"""Built-in clusterer implementations."""

import numpy as np

from common.logging.logger import get_logger
from mapping.protocols import Clusterer

logger = get_logger("mapper")


class HDBSCANClusterer(Clusterer):
    """HDBSCAN clustering on 2D coordinates (extracted from AtlasMapper)."""

    def __init__(self, min_cluster_size: int = 5):
        self._min_cluster_size = min_cluster_size

    @property
    def name(self) -> str:
        return "hdbscan"

    def fit_predict(self, coordinates: np.ndarray) -> np.ndarray:
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan not installed. Run: pip install hdbscan")

        n_docs = len(coordinates)
        min_cluster = min(self._min_cluster_size, max(2, n_docs // 10))

        logger.info(f"Computing HDBSCAN clusters for {n_docs} documents...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster,
            metric='euclidean',
            cluster_selection_method='eom',
        )
        labels = clusterer.fit_predict(coordinates)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_orphaned = int(np.sum(labels == -1))
        logger.info(f"Found {n_clusters} clusters, {n_orphaned} orphaned documents")

        return labels
