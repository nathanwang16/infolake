"""
Domain types for atlas_core.

Lightweight NewType aliases and dataclasses that define the vocabulary of
the entire system.  Every pipeline module uses these — never raw strings
or untyped dicts for domain concepts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, NewType, Optional

import torch

# ---------------------------------------------------------------------------
# Scalar type aliases
# ---------------------------------------------------------------------------

DocumentID = NewType("DocumentID", str)
"""SHA256(canonical_url)[:16] — primary key across all stores."""

URL = NewType("URL", str)
"""Canonical (normalised) URL string."""

HyperedgeLabel = NewType("HyperedgeLabel", str)
"""Label for a hyperedge column (topic name, domain, date-bin key)."""

Embedding = NewType("Embedding", torch.Tensor)
"""1-D float32 tensor of shape (D,) — a single document embedding."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    """Detected document content type."""
    SCIENTIFIC = "scientific"
    TECHNICAL_CODE = "technical_code"
    PERSONAL_ESSAY = "personal_essay"
    NEWS = "news"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Lifecycle status of a document."""
    ACTIVE = "active"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class QualityProfile(str, Enum):
    """Whether the document has been scored."""
    PENDING = "pending"
    SCORED = "scored"


class SourcePhase(str, Enum):
    """Which pipeline phase produced this document."""
    BATCH_DUMP = "batch_dump"
    ACTIVE_EXPLORATION = "active_exploration"


# ---------------------------------------------------------------------------
# Composite data records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Record:
    """
    Uniform record emitted by every SourceFunctor.

    This is the lingua franca between dump adapters and the ingestion
    pipeline.  All fields except *url* may be absent at parse time and
    filled later by the pipeline.
    """
    url: URL
    html: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    domain: Optional[str] = None
    language: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredDocument:
    """
    A document with all scores computed.  Produced by the post-processing
    stage and consumed by the writer / exporter.
    """
    doc_id: DocumentID
    quality_score: float
    wilson_score: float
    importance_score: float
    content_type: ContentType
    cluster_id: int
    is_duplicate: bool
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass
class TensorCheckpoint:
    """
    Metadata for a saved tensor checkpoint file.
    """
    path: str
    shape: List[int]
    dtype: str
    device: str
    row_count: int
