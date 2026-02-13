"""
Stage-2 post-processing for Phase 1:
GPU-scoring, GPU-deduplication, and GPU mapping (project + cluster).
"""

import json
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from atlas_core import config, get_logger, tensor_ops
from atlas_core.errors import AtlasPipelineError
from common.database import db as _default_db
from common.qdrant_manager import QdrantManager
from common.repositories import DocumentRepository, DocumentTextRepository
from curation.scoring import ScoringPipeline
from curation.scoring.detection import RuleBasedContentTypeDetector

logger = get_logger("phase1.post_processor")


def _safe_qdrant_id(doc_id: str) -> Optional[str]:
    """Convert document id to UUID string when possible."""
    try:
        return str(uuid.UUID(doc_id))
    except Exception:
        return None


class PostProcessor:
    """Batch post-processor aligned with guide sections 4/5/6."""

    def __init__(
        self,
        database=None,
        qdrant_manager=None,
        scoring_pipeline=None,
        doc_repo=None,
        text_repo=None,
        content_type_detector=None,
    ):
        self._database = database or _default_db
        self._qdrant_mgr = qdrant_manager or QdrantManager(create_if_missing=False, timeout=5)
        self._pipeline = scoring_pipeline or ScoringPipeline()
        self._doc_repo = doc_repo or DocumentRepository(self._database)
        self._text_repo = text_repo or DocumentTextRepository(self._database)
        self._detector = content_type_detector or RuleBasedContentTypeDetector()

        self.quality_threshold = config.require("batch_processing.quality_threshold")
        self._dedup_threshold = self._resolve_dedup_threshold()
        self._mapping_sample = int(config.require("mapping.sample_for_fit"))

        self._stats = {
            "total_processed": 0,
            "scored": 0,
            "quality_rejected": 0,
            "dedup_rejected": 0,
            "accepted": 0,
            "errors": 0,
            "embedded_for_dedup": 0,
            "mapped": 0,
        }

    def _resolve_dedup_threshold(self) -> float:
        """
        Resolve cosine dedup threshold without silent defaults.
        Prefer the new key; fall back to legacy embedding key.
        """
        new_value = config.get("deduplication.cosine_threshold")
        if new_value is not None:
            return float(new_value)
        legacy_value = config.get("deduplication.embedding_similarity_threshold")
        if legacy_value is not None:
            logger.warning(
                "Using legacy deduplication.embedding_similarity_threshold; "
                "please migrate to deduplication.cosine_threshold"
            )
            return float(legacy_value)
        raise AtlasPipelineError(
            "post_processor",
            "Missing deduplication.cosine_threshold (or legacy embedding_similarity_threshold)",
        )

    def run(
        self,
        batch_size: int = 1000,
        quality_threshold: Optional[float] = None,
        skip_fps: bool = False,
        skip_dedup: bool = False,
    ) -> Dict[str, Any]:
        if quality_threshold is not None:
            self.quality_threshold = quality_threshold

        logger.info(
            "PostProcessor start batch_size=%d quality_th=%.3f skip_fps=%s skip_dedup=%s",
            batch_size,
            self.quality_threshold,
            skip_fps,
            skip_dedup,
        )
        start_time = time.time()
        batch_num = 0

        while True:
            batch = self._text_repo.get_unscored_batch(batch_size)
            if not batch:
                break
            batch_num += 1
            self._process_batch(batch=batch, skip_dedup=skip_dedup)
            elapsed = max(time.time() - start_time, 1.0)
            logger.info(
                "Batch %d done processed=%d scored=%d dedup=%d rate=%.2f docs/s",
                batch_num,
                self._stats["total_processed"],
                self._stats["scored"],
                self._stats["dedup_rejected"],
                self._stats["total_processed"] / elapsed,
            )

        total_elapsed = time.time() - start_time
        logger.info("PostProcessor finished in %.2fs stats=%s", total_elapsed, self._stats)
        return dict(self._stats)

    def _process_batch(self, batch: List[Tuple[str, str]], skip_dedup: bool) -> None:
        doc_rows: List[Dict[str, Any]] = []
        metric_names: List[str] = list(self._pipeline.registry.names)
        if not metric_names:
            raise AtlasPipelineError("post_processor", "scoring registry has no metrics")

        for doc_id, text in batch:
            try:
                doc = self._doc_repo.get_by_id(doc_id)
                metadata = {
                    "url": doc.url if doc else "",
                    "title": doc.title if doc else "",
                    "domain": doc.domain if doc else "",
                }
                content_type = self._detector.detect(text, metadata)
                raw_metrics = self._pipeline.compute_raw_metrics(text, metadata)
                metric_vector = [float(raw_metrics.get(name, 0.0)) for name in metric_names]

                doc_rows.append(
                    {
                        "doc_id": doc_id,
                        "text": text,
                        "metadata": metadata,
                        "content_type": content_type,
                        "raw_metrics": raw_metrics,
                        "metric_vector": metric_vector,
                    }
                )
            except Exception as exc:
                logger.error("Batch prep failed for %s: %s", doc_id, exc)
                self._stats["errors"] += 1

        if not doc_rows:
            return

        scores, wilson_scores = self._gpu_score_batch(doc_rows=doc_rows, metric_names=metric_names)
        duplicate_mask = self._gpu_duplicate_mask(doc_rows=doc_rows, skip_dedup=skip_dedup)

        self._apply_score_updates(doc_rows, scores, wilson_scores, duplicate_mask)
        self._run_gpu_mapping(doc_rows=doc_rows, scores=scores)
        self._stats["total_processed"] += len(doc_rows)

    def _gpu_score_batch(
        self,
        doc_rows: List[Dict[str, Any]],
        metric_names: List[str],
    ) -> Tuple[List[float], List[float]]:
        features = torch.tensor([row["metric_vector"] for row in doc_rows], dtype=torch.float32)
        positive_counts = (features > 0.5).sum(dim=1).float()
        total_counts = torch.full_like(positive_counts, fill_value=len(metric_names), dtype=torch.float32)

        scores = torch.zeros(features.shape[0], dtype=torch.float32)
        indices_by_type: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(doc_rows):
            indices_by_type[row["content_type"]].append(idx)

        for content_type, indices in indices_by_type.items():
            weights_dict = self._pipeline.get_weights(content_type)
            weights = torch.tensor(
                [float(weights_dict.get(name, 0.0)) for name in metric_names],
                dtype=torch.float32,
            )
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            part_features = features.index_select(0, idx_tensor)
            part_scores = tensor_ops.score(part_features, weights)
            scores.index_copy_(0, idx_tensor, part_scores.cpu())

        wilson = tensor_ops.wilson_score(positive_counts=positive_counts, total_counts=total_counts)
        return scores.tolist(), wilson.tolist()

    def _gpu_duplicate_mask(self, doc_rows: List[Dict[str, Any]], skip_dedup: bool) -> List[bool]:
        if skip_dedup:
            return [False] * len(doc_rows)

        vectors: List[List[float]] = []
        vector_indices: List[int] = []

        if not self._qdrant_mgr.available:
            logger.warning("Qdrant unavailable; skipping GPU deduplication")
            return [False] * len(doc_rows)

        for idx, row in enumerate(doc_rows):
            qdrant_id = _safe_qdrant_id(row["doc_id"])
            if qdrant_id is None:
                continue
            records = self._qdrant_mgr.retrieve(ids=[qdrant_id], with_vectors=True)
            if not records:
                continue
            vector = records[0].vector
            if vector:
                vectors.append(vector)
                vector_indices.append(idx)

        if not vectors:
            return [False] * len(doc_rows)

        embeddings = torch.tensor(vectors, dtype=torch.float32)
        duplicate_submask = tensor_ops.find_duplicates(
            embeddings=embeddings,
            threshold=float(self._dedup_threshold),
        ).tolist()
        duplicate_mask = [False] * len(doc_rows)
        for local_idx, global_idx in enumerate(vector_indices):
            duplicate_mask[global_idx] = bool(duplicate_submask[local_idx])

        self._stats["embedded_for_dedup"] += len(vectors)
        return duplicate_mask

    def _apply_score_updates(
        self,
        doc_rows: List[Dict[str, Any]],
        scores: List[float],
        wilson_scores: List[float],
        duplicate_mask: List[bool],
    ) -> None:
        updates: List[Tuple[str, float, str, str, float, str]] = []

        for idx, row in enumerate(doc_rows):
            score = float(scores[idx])
            wilson = float(wilson_scores[idx])
            is_duplicate = bool(duplicate_mask[idx])
            status = "duplicate" if is_duplicate else "active"

            if is_duplicate:
                self._stats["dedup_rejected"] += 1
            elif score < self.quality_threshold:
                self._stats["quality_rejected"] += 1
            else:
                self._stats["accepted"] += 1

            updates.append(
                (
                    row["doc_id"],
                    score,
                    json.dumps(row["raw_metrics"]),
                    row["content_type"],
                    wilson,
                    status,
                )
            )

        try:
            self._doc_repo.update_scores_batch(updates)
            self._stats["scored"] += len(updates)
        except Exception as exc:
            logger.error("Batch DB update failed: %s", exc)
            self._stats["errors"] += len(updates)

    def _run_gpu_mapping(self, doc_rows: List[Dict[str, Any]], scores: List[float]) -> None:
        if not self._qdrant_mgr.available:
            return

        vectors: List[List[float]] = []
        doc_ids: List[str] = []
        for idx, row in enumerate(doc_rows):
            qdrant_id = _safe_qdrant_id(row["doc_id"])
            if qdrant_id is None:
                continue
            records = self._qdrant_mgr.retrieve(ids=[qdrant_id], with_vectors=True)
            if not records or not records[0].vector:
                continue
            vectors.append(records[0].vector)
            doc_ids.append(row["doc_id"])

        if len(vectors) < 3:
            return

        if len(vectors) > self._mapping_sample:
            vectors = vectors[: self._mapping_sample]
            doc_ids = doc_ids[: self._mapping_sample]

        try:
            embeddings = torch.tensor(vectors, dtype=torch.float32)
            coords_2d = tensor_ops.project_2d(embeddings)
            labels = tensor_ops.cluster(coords_2d)
            updates = []
            for i, doc_id in enumerate(doc_ids):
                importance = float(scores[i]) if i < len(scores) else 0.0
                updates.append((int(labels[i].item()), importance, doc_id))
            self._doc_repo.update_mappings_batch(updates)
            self._stats["mapped"] += len(updates)
        except Exception as exc:
            logger.warning("GPU mapping skipped for current batch: %s", exc)

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
