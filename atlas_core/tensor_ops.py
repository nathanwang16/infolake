"""
GPU tensor operations for atlas_core.

All core linear algebra runs on GPU (MPS/CUDA) via PyTorch.  These are
the building blocks every pipeline module calls — no module contains its
own math.  CPU fallback is available but logged as a warning.

Micro-batching (10 K rows) is used throughout so VRAM acts as a cache
over Parquet and OOM does not wipe state.
"""

import math
from typing import Optional

import torch

from atlas_core.config import config
from atlas_core.errors import AtlasConfigError, AtlasTensorError
from atlas_core.logging import get_logger

logger = get_logger("atlas_core.tensor_ops")

# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

_device: Optional[torch.device] = None


def resolve_device() -> torch.device:
    """Return the configured torch device, cached after first call."""
    global _device
    if _device is not None:
        return _device

    requested = config.get("gpu.device", config.get("embedding.device", "cpu"))

    if requested == "mps" and torch.backends.mps.is_available():
        _device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        if requested not in ("cpu",):
            logger.warning(
                "Requested device '%s' unavailable, falling back to CPU", requested
            )
        _device = torch.device("cpu")

    logger.info("Resolved compute device: %s", _device)
    return _device


def _micro_batch_size() -> int:
    return int(config.get("gpu.micro_batch_size", 10_000))


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed(
    texts: list[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_tokens: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Batch-encode *texts* through *model* on *device* with OOM protection.

    On OOM the batch size is halved automatically and the batch retried.

    Args:
        texts: Raw document strings.
        model: PyTorch model with a forward pass returning last_hidden_state.
        tokenizer: HuggingFace-compatible tokenizer.
        device: Target torch device.
        max_tokens: Maximum token length per text.
        batch_size: Starting batch size (auto-halved on OOM).

    Returns:
        Float32 tensor of shape (len(texts), D).

    Raises:
        AtlasTensorError: If batch_size reaches 0 without success.
    """
    if not texts:
        raise AtlasTensorError("embed", "received empty text list")

    all_embeddings: list[torch.Tensor] = []
    current_batch = batch_size

    i = 0
    while i < len(texts):
        batch_texts = texts[i : i + current_batch]
        try:
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)
                # Mean-pool over token dimension
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts

            all_embeddings.append(pooled.cpu())
            i += current_batch

        except torch.cuda.OutOfMemoryError:
            current_batch = current_batch // 2
            if current_batch < 1:
                raise AtlasTensorError("embed", "OOM with batch_size=1")
            logger.warning("OOM during embed — halving batch to %d", current_batch)
            torch.cuda.empty_cache()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                current_batch = current_batch // 2
                if current_batch < 1:
                    raise AtlasTensorError("embed", "OOM with batch_size=1")
                logger.warning("OOM during embed — halving batch to %d", current_batch)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise AtlasTensorError("embed", str(exc))

    result = torch.cat(all_embeddings, dim=0)
    logger.debug("Embedded %d texts → %s", len(texts), tuple(result.shape))
    return result


# ---------------------------------------------------------------------------
# Scoring: weighted feature projection
# ---------------------------------------------------------------------------

def score(feature_matrix: torch.Tensor, weight_vector: torch.Tensor) -> torch.Tensor:
    """
    Compute quality scores as a dot product: ``F @ w``.

    Args:
        feature_matrix: (N, F) float32 on GPU.
        weight_vector:  (F,)   float32 on same device.

    Returns:
        (N,) float32 scores clamped to [0, 1].

    Raises:
        AtlasTensorError: On shape mismatch.
    """
    if feature_matrix.ndim != 2:
        raise AtlasTensorError("score", f"feature_matrix must be 2-D, got {feature_matrix.ndim}-D")
    if weight_vector.ndim != 1:
        raise AtlasTensorError("score", f"weight_vector must be 1-D, got {weight_vector.ndim}-D")
    if feature_matrix.shape[1] != weight_vector.shape[0]:
        raise AtlasTensorError(
            "score",
            f"dimension mismatch: features has {feature_matrix.shape[1]} cols "
            f"but weights has {weight_vector.shape[0]} elements",
        )

    scores = feature_matrix @ weight_vector
    return scores.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Wilson score: vectorised confidence interval
# ---------------------------------------------------------------------------

def wilson_score(
    positive_counts: torch.Tensor,
    total_counts: torch.Tensor,
    z: float = 1.96,
) -> torch.Tensor:
    """
    Vectorised Wilson score interval (lower bound) across entire corpus.

    Args:
        positive_counts: (N,) int or float tensor.
        total_counts:    (N,) int or float tensor (must be > 0).
        z: Z-value for confidence (default 1.96 = 95 %).

    Returns:
        (N,) float32 Wilson lower-bound scores.
    """
    n = total_counts.float().clamp(min=1.0)
    p = positive_counts.float() / n
    z2 = z * z

    denominator = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    spread = z * torch.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)

    return ((centre - spread) / denominator).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Deduplication: cosine similarity via gram matrix
# ---------------------------------------------------------------------------

def find_duplicates(
    embeddings: torch.Tensor,
    threshold: float = 0.95,
) -> torch.Tensor:
    """
    Identify duplicates by computing the cosine-similarity gram matrix in
    micro-batches so VRAM is bounded.

    Args:
        embeddings: (N, D) float32 tensor.
        threshold: Cosine similarity above which two documents are duplicates.

    Returns:
        Boolean mask of shape (N,) — True for rows to remove (keeps the
        first occurrence of each duplicate cluster).
    """
    if embeddings.ndim != 2:
        raise AtlasTensorError("find_duplicates", f"expected 2-D tensor, got {embeddings.ndim}-D")

    device = resolve_device()
    N = embeddings.shape[0]
    mb = _micro_batch_size()

    # Normalise once on device
    E = embeddings.to(device)
    norms = E.norm(dim=1, keepdim=True).clamp(min=1e-8)
    E_normed = E / norms

    duplicate_mask = torch.zeros(N, dtype=torch.bool, device=device)

    for start in range(0, N, mb):
        end = min(start + mb, N)
        batch = E_normed[start:end]                 # (B, D)
        sim = batch @ E_normed.T                     # (B, N)

        # Zero out self-similarity and already-marked duplicates
        sim[:, start:end] = sim[:, start:end].triu(diagonal=1)

        # Any row whose max sim exceeds threshold — mark the *later* index
        row_max, col_idx = sim.max(dim=1)
        high = row_max > threshold
        # Only mark the batch row (the later one) as duplicate
        batch_indices = torch.arange(start, end, device=device)
        duplicate_mask[batch_indices[high]] = True

    logger.info(
        "Deduplication: %d / %d marked as duplicates (threshold=%.2f)",
        duplicate_mask.sum().item(), N, threshold,
    )
    return duplicate_mask.cpu()


# ---------------------------------------------------------------------------
# Farthest Point Sampling
# ---------------------------------------------------------------------------

def farthest_point_sample(
    embeddings: torch.Tensor,
    k: int,
    quality_weights: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    GPU-parallel farthest-point sampling biased by quality.

    At each step the chosen point maximises
    ``distance^alpha * quality_weight``.

    Args:
        embeddings: (N, D) float32.
        k: Number of points to select.
        quality_weights: Optional (N,) weights in [0, 1].
        alpha: Distance exponent (higher = more diversity).

    Returns:
        (k,) int64 tensor of selected row indices.

    Raises:
        AtlasTensorError: If k > N or tensors are malformed.
    """
    N = embeddings.shape[0]
    if k > N:
        raise AtlasTensorError("farthest_point_sample", f"k={k} > N={N}")
    if k < 1:
        raise AtlasTensorError("farthest_point_sample", f"k must be >= 1, got {k}")

    device = resolve_device()
    E = embeddings.to(device)

    if quality_weights is not None:
        qw = quality_weights.to(device).float()
    else:
        qw = torch.ones(N, device=device)

    # Start from the highest-quality point
    selected: list[int] = [int(qw.argmax().item())]
    min_dists = torch.full((N,), float("inf"), device=device)

    for _ in range(k - 1):
        last = E[selected[-1]].unsqueeze(0)       # (1, D)
        dists = torch.cdist(E, last).squeeze(1)    # (N,)
        min_dists = torch.min(min_dists, dists)

        criterion = (min_dists ** alpha) * qw
        # Exclude already-selected points
        for idx in selected:
            criterion[idx] = -1.0
        selected.append(int(criterion.argmax().item()))

    result = torch.tensor(selected, dtype=torch.int64)
    logger.debug("FPS selected %d / %d points", k, N)
    return result


# ---------------------------------------------------------------------------
# Gap detection: inverted k-NN density
# ---------------------------------------------------------------------------

def find_gaps(
    embeddings: torch.Tensor,
    k_neighbors: int,
) -> torch.Tensor:
    """
    Estimate coverage gaps via mean k-NN distance.

    High mean k-NN distance indicates a sparse region — a gap worth
    exploring in Phase 2.

    Args:
        embeddings: (N, D) float32.
        k_neighbors: Number of neighbours for density estimation.

    Returns:
        (N,) float32 gap scores (higher = sparser neighbourhood).
    """
    if k_neighbors < 1:
        raise AtlasTensorError("find_gaps", f"k_neighbors must be >= 1, got {k_neighbors}")

    device = resolve_device()
    E = embeddings.to(device)
    N = E.shape[0]
    mb = _micro_batch_size()
    k = min(k_neighbors + 1, N)   # +1 because self is included

    gap_scores = torch.zeros(N, device=device)

    for start in range(0, N, mb):
        end = min(start + mb, N)
        dists = torch.cdist(E[start:end], E)            # (B, N)
        topk_dists, _ = dists.topk(k, dim=1, largest=False)  # (B, k)
        # Skip the self-distance column (index 0)
        gap_scores[start:end] = topk_dists[:, 1:].mean(dim=1)

    logger.debug("Gap detection complete for %d points, k=%d", N, k_neighbors)
    return gap_scores.cpu()


# ---------------------------------------------------------------------------
# 2-D projection (UMAP wrapper)
# ---------------------------------------------------------------------------

def project_2d(
    embeddings: torch.Tensor,
    n_neighbors: Optional[int] = None,
    min_dist: Optional[float] = None,
    metric: Optional[str] = None,
) -> torch.Tensor:
    """
    Project high-dimensional embeddings to 2-D coordinates via UMAP.

    Parameters default to ``config.json → mapping.umap.*``.

    Args:
        embeddings: (N, D) float32 tensor.
        n_neighbors: UMAP n_neighbors.
        min_dist: UMAP min_dist.
        metric: UMAP distance metric.

    Returns:
        (N, 2) float32 coordinate tensor.
    """
    try:
        import umap
    except ImportError as exc:
        raise AtlasTensorError("project_2d", "umap-learn not installed") from exc

    n_neighbors = n_neighbors or int(config.get("mapping.umap.neighbors", 50))
    min_dist = min_dist or float(config.get("mapping.umap.min_dist", 0.1))
    metric = metric or str(config.get("mapping.umap.metric", "cosine"))

    data = embeddings.cpu().numpy()

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )

    logger.info(
        "UMAP projection: N=%d, D=%d, n_neighbors=%d, min_dist=%.2f, metric=%s",
        data.shape[0], data.shape[1], n_neighbors, min_dist, metric,
    )
    coords = reducer.fit_transform(data)
    return torch.from_numpy(coords).float()


# ---------------------------------------------------------------------------
# Clustering (HDBSCAN wrapper)
# ---------------------------------------------------------------------------

def cluster(
    coords: torch.Tensor,
    min_cluster_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Cluster 2-D coordinates via HDBSCAN.

    Args:
        coords: (N, 2) float32 tensor.
        min_cluster_size: Minimum cluster size (default from config).

    Returns:
        (N,) int64 cluster labels (-1 = noise).
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise AtlasTensorError("cluster", "hdbscan not installed") from exc

    min_cluster_size = min_cluster_size or int(
        config.get("mapping.hdbscan.min_cluster_size", 5)
    )

    data = coords.cpu().numpy()

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info(
        "HDBSCAN: %d clusters, %d noise points (min_cluster_size=%d)",
        n_clusters, n_noise, min_cluster_size,
    )
    return torch.from_numpy(labels).long()
