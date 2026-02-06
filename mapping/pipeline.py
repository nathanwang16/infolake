"""Mapping pipeline — orchestrates projection, clustering, and scoring."""

from typing import List, Optional

import numpy as np

from common.logging.logger import get_logger
from mapping.protocols import Projector, Clusterer, AxisScorer
from mapping.projectors import UMAPProjector
from mapping.clusterers import HDBSCANClusterer
from mapping.axis_scorers import DomainAuthorityAxisScorer

logger = get_logger("mapper")


class MappingPipeline:
    """
    Orchestrates the full mapping pipeline: project -> cluster -> score.

    Each component can be swapped independently.

    Args:
        projector: Dimensionality reduction strategy (default: UMAP).
        clusterer: Clustering strategy (default: HDBSCAN).
        axis_scorer: Importance/Z-axis scoring strategy (default: DomainAuthority).
    """

    def __init__(
        self,
        projector: Optional[Projector] = None,
        clusterer: Optional[Clusterer] = None,
        axis_scorer: Optional[AxisScorer] = None,
    ):
        self.projector = projector or UMAPProjector()
        self.clusterer = clusterer or HDBSCANClusterer()
        self.axis_scorer = axis_scorer or DomainAuthorityAxisScorer()

    def project(
        self, embeddings: np.ndarray, force_refit: bool = False
    ) -> np.ndarray:
        """Project embeddings to 2D coordinates."""
        if force_refit or not hasattr(self.projector, '_model') or getattr(self.projector, '_model', None) is None:
            return self.projector.fit_transform(embeddings)
        return self.projector.transform(embeddings)

    def cluster(self, coordinates: np.ndarray) -> np.ndarray:
        """Cluster 2D coordinates, returning integer labels."""
        return self.clusterer.fit_predict(coordinates)

    def score_importance(
        self,
        domain: str,
        quality_score: float,
        content_type: Optional[str] = None,
        inbound_links: Optional[int] = None,
        citations: Optional[int] = None,
    ) -> float:
        """Compute importance score for a single document."""
        return self.axis_scorer.compute(
            domain=domain,
            quality_score=quality_score,
            content_type=content_type,
            inbound_links=inbound_links,
            citations=citations,
        )
