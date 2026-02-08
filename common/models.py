"""
Domain model dataclasses for Truth Atlas.

Each model provides:
- from_row(): classmethod to construct from a database row tuple
- to_dict(): returns dict with backwards-compatible key names
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Document:
    """Full document from single-doc queries (includes summary)."""
    id: str
    url: str
    title: Optional[str]
    summary: Optional[str]
    domain: str
    content_type: Optional[str]
    quality_score: Optional[float]
    wilson_score: Optional[float]
    importance_score: Optional[float]
    cluster_id: Optional[int]
    created_at: Optional[str]

    @classmethod
    def from_row(cls, row: tuple) -> "Document":
        return cls(
            id=row[0],
            url=row[1],
            title=row[2],
            summary=row[3],
            domain=row[4],
            content_type=row[5],
            quality_score=row[6],
            wilson_score=row[7],
            importance_score=row[8],
            cluster_id=row[9],
            created_at=row[10],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'url': self.url,
            'title': self.title,
            'summary': self.summary,
            'domain': self.domain,
            'content_type': self.content_type,
            'quality_score': self.quality_score,
            'wilson_score': self.wilson_score,
            'importance_score': self.importance_score,
            'cluster_id': self.cluster_id,
            'created_at': self.created_at,
        }


@dataclass
class DocumentListItem:
    """Slim document for list queries (no summary)."""
    id: str
    url: str
    title: Optional[str]
    domain: str
    content_type: Optional[str]
    quality_score: Optional[float]
    wilson_score: Optional[float]
    importance_score: Optional[float]
    cluster_id: Optional[int]
    created_at: Optional[str]

    @classmethod
    def from_row(cls, row: tuple) -> "DocumentListItem":
        return cls(
            id=row[0],
            url=row[1],
            title=row[2],
            domain=row[3],
            content_type=row[4],
            quality_score=row[5],
            wilson_score=row[6],
            importance_score=row[7],
            cluster_id=row[8],
            created_at=row[9],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'url': self.url,
            'title': self.title,
            'domain': self.domain,
            'content_type': self.content_type,
            'quality_score': self.quality_score,
            'wilson_score': self.wilson_score,
            'importance_score': self.importance_score,
            'cluster_id': self.cluster_id,
            'created_at': self.created_at,
        }


@dataclass
class DocumentCreate:
    """Write-only model for INSERT operations."""
    id: str
    url: str
    title: str
    summary: str
    content_hash: str
    domain: str
    content_type: str = 'unscored'
    quality_score: float = 0.0
    quality_components: str = '{}'  # JSON string
    quality_profile_used: str = 'pending'
    raw_html_hash: str = ''
    novelty_distance: float = 0.0
    source_phase: str = 'batch'
    source_dump: Optional[str] = None
    content_length: int = 0


@dataclass
class ClusterInfo:
    """Cluster stats from GROUP BY."""
    cluster_id: Optional[int]
    doc_count: int
    avg_quality: Optional[float]
    min_quality: Optional[float]
    max_quality: Optional[float]

    @classmethod
    def from_row(cls, row: tuple) -> "ClusterInfo":
        return cls(
            cluster_id=row[0],
            doc_count=row[1],
            avg_quality=row[2],
            min_quality=row[3],
            max_quality=row[4],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'doc_count': self.doc_count,
            'avg_quality': self.avg_quality,
            'min_quality': self.min_quality,
            'max_quality': self.max_quality,
            'is_orphaned': self.cluster_id == -1,
        }


@dataclass
class CoverageMetrics:
    """Health coverage metrics."""
    topic_gini: float
    domain_gini: float
    orphan_rate: float
    high_quality_orphans: int
    cluster_count: int
    largest_cluster_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'topic_gini': self.topic_gini,
            'domain_gini': self.domain_gini,
            'orphan_rate': self.orphan_rate,
            'high_quality_orphans': self.high_quality_orphans,
            'cluster_count': self.cluster_count,
            'largest_cluster_pct': self.largest_cluster_pct,
        }


@dataclass
class SearchResult:
    """Document with similarity score from vector search."""
    document: Document
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        d = self.document.to_dict()
        d['similarity_score'] = self.similarity_score
        return d


@dataclass
class GoldenSetEntry:
    """Golden set entry for calibration."""
    url: str
    label: str
    content_type: str
    notes: str
    raw_metrics: str  # JSON string
    version: int
    domain: str


@dataclass
class DumpJob:
    """Processing job record."""
    id: str
    dump_name: str
    dump_path: str
