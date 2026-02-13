"""
Farthest Point Sampling (FPS) with Lazy Greedy optimization.

Implements quality-weighted farthest point sampling for maximizing coverage
across embedding space while considering document quality scores.

Algorithm:
    1. Maintain upper bounds on selection scores (lazy evaluation)
    2. Use priority queue to efficiently find candidates
    3. Only recompute distances when upper bound is competitive
    
Complexity: O(n log n) build + O(k × log n × log |selected|) amortized
            vs O(n × k) for naive FPS

Reference: Technical Engineering Guide Section 3.3-3.4
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from atlas_core import get_logger

logger = get_logger("fps_sampler")


@dataclass(order=True)
class FPSCandidate:
    """
    Candidate document for FPS selection.
    Ordered by negative score (max-heap via min-heap with negation).
    """
    priority: float  # Negative of score for max-heap behavior
    doc_id: str = field(compare=False)
    embedding: np.ndarray = field(compare=False, repr=False)
    quality_score: float = field(compare=False)
    upper_bound: float = field(compare=False, default=float('inf'))
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)


class FarthestPointSampler:
    """
    Quality-weighted Farthest Point Sampling with Lazy Greedy optimization.
    
    Selection criterion:
        score(c) = min_distance(c) × quality_score(c)^α
        
    Where:
        α = 0.0: Pure coverage, ignores quality
        α = 0.5: Mild quality preference
        α = 1.0: Balanced (recommended)
        α = 2.0: Strong quality preference
    """
    
    def __init__(
        self,
        quality_weight_alpha: float,
        use_qdrant: bool = False,
        qdrant_client: Optional[Any] = None,
        qdrant_collection: Optional[str] = None
    ):
        if quality_weight_alpha is None:
            raise ValueError("quality_weight_alpha is required")
        
        self.alpha = quality_weight_alpha
        self.use_qdrant = use_qdrant
        self.qdrant = qdrant_client
        self.qdrant_collection = qdrant_collection
        
        # Selected documents
        self._selected_embeddings: List[np.ndarray] = []
        self._selected_ids: List[str] = []
        
        # Statistics
        self._stats = {
            'total_considered': 0,
            'distance_computations': 0,
            'lazy_skips': 0,
        }
    
    def _compute_distance(self, embedding: np.ndarray) -> float:
        """
        Computes minimum cosine distance to any selected embedding.
        
        Returns:
            float: Distance (1 - similarity), higher = more novel
        """
        if not self._selected_embeddings:
            return float('inf')
        
        self._stats['distance_computations'] += 1
        
        if self.use_qdrant and self.qdrant and len(self._selected_ids) >= 100:
            # Use Qdrant for approximate NN when selected set is large
            try:
                results = self.qdrant.search(
                    collection_name=self.qdrant_collection,
                    query_vector=embedding.tolist(),
                    limit=1
                )
                if results:
                    # Qdrant returns similarity; convert to distance
                    return 1.0 - results[0].score
            except Exception as e:
                logger.warning(f"Qdrant search failed, falling back to brute force: {e}")
        
        # Brute force for small selected sets (fast with numpy)
        selected_matrix = np.array(self._selected_embeddings)
        
        # Cosine similarity: dot product of normalized vectors
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        selected_norms = selected_matrix / (np.linalg.norm(selected_matrix, axis=1, keepdims=True) + 1e-10)
        
        similarities = np.dot(selected_norms, embedding_norm)
        max_similarity = np.max(similarities)
        
        # Distance = 1 - similarity
        return 1.0 - max_similarity
    
    def _compute_selection_score(self, distance: float, quality: float) -> float:
        """
        Computes quality-weighted selection score.
        
        Args:
            distance: Minimum distance to selected set (1 - similarity)
            quality: Document quality score [0, 1]
            
        Returns:
            Selection score (higher is better)
        """
        if distance <= 0:
            return 0.0
        
        # Avoid zero quality causing issues
        quality = max(quality, 0.01)
        
        return distance * (quality ** self.alpha)
    
    def should_select(
        self,
        doc_id: str,
        embedding: np.ndarray,
        quality_score: float,
        novelty_threshold: float
    ) -> Tuple[bool, float, float]:
        """
        Determines if a document should be selected using FPS criterion.
        
        This is the streaming/online version for batch processing where
        we evaluate each document as it arrives.
        
        Args:
            doc_id: Document identifier
            embedding: Document embedding vector
            quality_score: Document quality score [0, 1]
            novelty_threshold: Minimum distance threshold for novelty
            
        Returns:
            Tuple of (should_select, distance, selection_score)
        """
        if doc_id is None:
            raise ValueError("doc_id is required")
        if embedding is None:
            raise ValueError("embedding is required")
        if quality_score is None:
            raise ValueError("quality_score is required")
        if novelty_threshold is None:
            raise ValueError("novelty_threshold is required")
        
        self._stats['total_considered'] += 1
        
        # First document is always selected
        if not self._selected_embeddings:
            self._selected_embeddings.append(embedding)
            self._selected_ids.append(doc_id)
            return True, float('inf'), float('inf')
        
        # Compute distance to nearest selected
        distance = self._compute_distance(embedding)
        
        # Check novelty threshold
        if distance < novelty_threshold:
            return False, distance, 0.0
        
        # Compute selection score
        score = self._compute_selection_score(distance, quality_score)
        
        # In streaming mode, we accept if score is positive
        # (The FPS guarantee comes from the novelty threshold)
        if score > 0:
            self._selected_embeddings.append(embedding)
            self._selected_ids.append(doc_id)
            return True, distance, score
        
        return False, distance, score
    
    def batch_select(
        self,
        candidates: List[Dict[str, Any]],
        k: int,
        min_quality: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Selects k documents from candidates using Lazy Greedy FPS.
        
        This is the batch version for selecting from a pre-collected set
        of candidates (e.g., from a dump processing pass).
        
        Args:
            candidates: List of dicts with 'id', 'embedding', 'quality_score'
            k: Number of documents to select
            min_quality: Minimum quality threshold
            
        Returns:
            List of selected candidate dicts
        """
        if candidates is None:
            raise ValueError("candidates is required")
        if k is None:
            raise ValueError("k is required")
        
        # Filter by quality
        filtered = [c for c in candidates if c.get('quality_score', 0) >= min_quality]
        
        if not filtered:
            logger.warning("No candidates pass quality threshold")
            return []
        
        if len(filtered) <= k:
            logger.info(f"Only {len(filtered)} candidates, returning all")
            return filtered
        
        logger.info(f"Selecting {k} from {len(filtered)} candidates using Lazy Greedy FPS")
        
        # Initialize priority queue with upper bounds
        # Use negative scores for max-heap behavior
        pq = []
        candidates_by_id = {}
        
        for c in filtered:
            doc_id = c['id']
            quality = c.get('quality_score', 0.5)
            embedding = np.array(c['embedding'])
            
            # Initial upper bound is infinity (could be farthest)
            upper_bound = float('inf') * (quality ** self.alpha)
            
            candidates_by_id[doc_id] = FPSCandidate(
                priority=-upper_bound,
                doc_id=doc_id,
                embedding=embedding,
                quality_score=quality,
                upper_bound=upper_bound,
                metadata=c.get('metadata', {})
            )
            heapq.heappush(pq, candidates_by_id[doc_id])
        
        # Start with highest quality document
        best_quality_id = max(filtered, key=lambda c: c.get('quality_score', 0))['id']
        first = candidates_by_id[best_quality_id]
        
        selected = [first]
        self._selected_embeddings = [first.embedding]
        self._selected_ids = [first.doc_id]
        
        logger.info(f"Initial selection: {first.doc_id} (quality={first.quality_score:.3f})")
        
        # Lazy greedy selection
        iterations = 0
        max_iterations = len(filtered) * k  # Safety limit
        
        while len(selected) < k and pq and iterations < max_iterations:
            iterations += 1
            
            # Pop candidate with highest upper bound
            candidate = heapq.heappop(pq)
            
            # Skip if already selected
            if candidate.doc_id in self._selected_ids:
                continue
            
            # Compute actual score
            distance = self._compute_distance(candidate.embedding)
            actual_score = self._compute_selection_score(distance, candidate.quality_score)
            
            # Check if actual score beats next upper bound
            if pq and actual_score < -pq[0].priority:
                # Update upper bound and re-insert
                candidate.upper_bound = actual_score
                candidate.priority = -actual_score
                heapq.heappush(pq, candidate)
                self._stats['lazy_skips'] += 1
            else:
                # Select this candidate
                selected.append(candidate)
                self._selected_embeddings.append(candidate.embedding)
                self._selected_ids.append(candidate.doc_id)
                
                if len(selected) % 100 == 0:
                    logger.info(f"Selected {len(selected)}/{k} documents")
        
        # Log statistics
        logger.info(f"FPS selection complete: {len(selected)} documents selected")
        logger.info(f"Stats: {self._stats['distance_computations']} distance computations, "
                   f"{self._stats['lazy_skips']} lazy skips")
        
        # Return selected candidates with metadata
        result = []
        for s in selected:
            result.append({
                'id': s.doc_id,
                'embedding': s.embedding.tolist(),
                'quality_score': s.quality_score,
                'metadata': s.metadata
            })
        
        return result
    
    def get_selected_ids(self) -> List[str]:
        """Returns list of selected document IDs."""
        return self._selected_ids.copy()
    
    def get_stats(self) -> Dict[str, int]:
        """Returns selection statistics."""
        return self._stats.copy()
    
    def reset(self):
        """Resets the sampler state."""
        self._selected_embeddings = []
        self._selected_ids = []
        self._stats = {
            'total_considered': 0,
            'distance_computations': 0,
            'lazy_skips': 0,
        }


class StreamingFPSSampler:
    """
    Streaming variant of FPS for online processing.
    
    Maintains a reservoir of candidates and periodically runs
    batch FPS to select the best documents.
    """
    
    def __init__(
        self,
        reservoir_size: int,
        selection_ratio: float,
        quality_weight_alpha: float,
        novelty_threshold: float
    ):
        if reservoir_size is None:
            raise ValueError("reservoir_size is required")
        if selection_ratio is None:
            raise ValueError("selection_ratio is required")
        if quality_weight_alpha is None:
            raise ValueError("quality_weight_alpha is required")
        if novelty_threshold is None:
            raise ValueError("novelty_threshold is required")
        
        self.reservoir_size = reservoir_size
        self.selection_ratio = selection_ratio
        self.novelty_threshold = novelty_threshold
        
        self._reservoir: List[Dict[str, Any]] = []
        self._sampler = FarthestPointSampler(quality_weight_alpha=quality_weight_alpha)
        self._total_processed = 0
        self._total_selected = 0
    
    def add(self, doc_id: str, embedding: np.ndarray, quality_score: float, metadata: Dict = None) -> Optional[Dict]:
        """
        Adds a document to the reservoir.
        
        Returns selected document if reservoir is full and selection is made.
        """
        if doc_id is None:
            raise ValueError("doc_id is required")
        if embedding is None:
            raise ValueError("embedding is required")
        if quality_score is None:
            raise ValueError("quality_score is required")
        
        self._total_processed += 1
        
        # Add to reservoir
        self._reservoir.append({
            'id': doc_id,
            'embedding': embedding if isinstance(embedding, list) else embedding.tolist(),
            'quality_score': quality_score,
            'metadata': metadata or {}
        })
        
        # Check if reservoir is full
        if len(self._reservoir) >= self.reservoir_size:
            return self._flush_reservoir()
        
        return None
    
    def _flush_reservoir(self) -> List[Dict[str, Any]]:
        """Flushes reservoir by running FPS selection."""
        if not self._reservoir:
            return []
        
        k = int(len(self._reservoir) * self.selection_ratio)
        k = max(k, 1)
        
        selected = self._sampler.batch_select(self._reservoir, k)
        self._total_selected += len(selected)
        
        # Clear reservoir
        self._reservoir = []
        
        return selected
    
    def flush(self) -> List[Dict[str, Any]]:
        """Forces flush of remaining reservoir."""
        return self._flush_reservoir()
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns streaming stats."""
        return {
            'total_processed': self._total_processed,
            'total_selected': self._total_selected,
            'selection_rate': self._total_selected / max(self._total_processed, 1),
            'reservoir_size': len(self._reservoir),
            **self._sampler.get_stats()
        }
