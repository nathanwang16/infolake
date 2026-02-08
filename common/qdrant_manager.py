"""
Unified Qdrant connection manager for Truth Atlas.

Replaces the three duplicated _init_qdrant() methods in writer.py, atlas_store.py, and mapper.py.

Usage:
    # Reader (atlas_store, mapper) — collection must already exist
    mgr = QdrantManager()

    # Writer — create collection if missing
    mgr = QdrantManager(create_if_missing=True)

    # Injected with custom settings
    mgr = QdrantManager(url="http://custom:6333", collection="my_collection")
"""

from typing import Any, List, Optional

from common.config import config
from common.logging.logger import get_logger

logger = get_logger("qdrant_manager")


class QdrantManager:
    """Lazy-connecting Qdrant wrapper with optional collection creation."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection: Optional[str] = None,
        timeout: Optional[float] = None,
        create_if_missing: bool = False,
        vector_size: int = 384,
    ):
        self._url = url or config.get("qdrant.url")
        self._collection = collection or config.get("qdrant.collection")
        self._timeout = (
            timeout if timeout is not None
            else config.require("qdrant.timeout_seconds")
        )
        self._create_if_missing = create_if_missing
        self._vector_size = vector_size

        self._client = None
        self._available = False
        self._initialized = False
        self._error_logged = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True

        try:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self._url, timeout=self._timeout)

            # Verify / create collection
            try:
                self._client.get_collection(self._collection)
                self._available = True
                logger.info(f"Connected to Qdrant collection: {self._collection}")
            except Exception:
                if self._create_if_missing:
                    try:
                        from qdrant_client.models import Distance, VectorParams

                        logger.info(f"Creating Qdrant collection: {self._collection}")
                        self._client.create_collection(
                            collection_name=self._collection,
                            vectors_config=VectorParams(
                                size=self._vector_size,
                                distance=Distance.COSINE,
                            ),
                        )
                        self._available = True
                    except Exception as e:
                        logger.warning(f"Qdrant not available: {e}. Running without vector search.")
                        self._client = None
                else:
                    logger.warning(f"Qdrant collection '{self._collection}' not found")

        except ImportError:
            logger.warning("qdrant-client not installed, running without vector search")
            self._client = None
        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}. Running without vector search.")
            self._client = None

    # ---- properties ----

    @property
    def client(self):
        """Returns the QdrantClient or None if unavailable."""
        self._ensure_initialized()
        return self._client

    @property
    def available(self) -> bool:
        """True if connected and collection exists."""
        self._ensure_initialized()
        return self._available

    @property
    def collection_name(self) -> str:
        return self._collection

    # ---- convenience methods ----

    def search(self, query_vector, limit: int = 10, **kwargs):
        self._ensure_initialized()
        if not self._available or not self._client:
            return []
        return self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=limit,
            **kwargs,
        )

    def upsert(self, points):
        self._ensure_initialized()
        if not self._available or not self._client:
            return None
        return self._client.upsert(collection_name=self._collection, points=points)

    def scroll(self, **kwargs):
        self._ensure_initialized()
        if not self._available or not self._client:
            return [], None
        return self._client.scroll(collection_name=self._collection, **kwargs)

    def retrieve(self, ids, with_vectors: bool = False):
        self._ensure_initialized()
        if not self._available or not self._client:
            return []
        return self._client.retrieve(
            collection_name=self._collection, ids=ids, with_vectors=with_vectors
        )

    def get_collection_info(self):
        self._ensure_initialized()
        if not self._available or not self._client:
            return None
        try:
            return self._client.get_collection(self._collection)
        except Exception:
            return None
