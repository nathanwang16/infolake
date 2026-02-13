"""
Custom exception hierarchy for atlas_core.

All parameters are required — no silent defaults. Missing values raise
AtlasConfigError. Pipeline failures raise domain-specific subclasses
of AtlasError so callers can catch at the granularity they need.
"""


class AtlasError(Exception):
    """Base exception for all atlas_core errors."""


class AtlasConfigError(AtlasError):
    """Raised when a required configuration key is missing or invalid."""

    def __init__(self, key: str, reason: str = "missing or None"):
        self.key = key
        super().__init__(f"Configuration error for '{key}': {reason}")


class AtlasTensorError(AtlasError):
    """Raised on GPU/tensor operation failures (OOM, shape mismatch, device errors)."""

    def __init__(self, operation: str, detail: str):
        self.operation = operation
        super().__init__(f"Tensor operation '{operation}' failed: {detail}")


class AtlasStorageError(AtlasError):
    """Raised when a storage backend (SQLite, Qdrant, Parquet) operation fails."""

    def __init__(self, backend: str, detail: str):
        self.backend = backend
        super().__init__(f"Storage error [{backend}]: {detail}")


class AtlasEmbeddingError(AtlasError):
    """Raised when embedding generation fails."""

    def __init__(self, detail: str):
        super().__init__(f"Embedding error: {detail}")


class AtlasPipelineError(AtlasError):
    """Raised when a pipeline stage encounters an unrecoverable error."""

    def __init__(self, stage: str, detail: str):
        self.stage = stage
        super().__init__(f"Pipeline stage '{stage}' failed: {detail}")


class AtlasHypergraphError(AtlasError):
    """Raised on hypergraph construction or query failures."""

    def __init__(self, detail: str):
        super().__init__(f"Hypergraph error: {detail}")


class AtlasFunctorError(AtlasError):
    """Raised when a source functor fails to map records."""

    def __init__(self, source: str, detail: str):
        self.source = source
        super().__init__(f"Functor error [{source}]: {detail}")
