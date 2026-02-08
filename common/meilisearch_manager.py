"""
Meilisearch connection manager for Truth Atlas.

Provides document indexing and full-text search via Meilisearch.
Follows the same lazy-init and graceful degradation pattern as QdrantManager.

Usage:
    # Reader (visualizer, server) — index must already exist
    mgr = MeilisearchManager()

    # Writer (pipeline) — creates index if missing
    mgr = MeilisearchManager(create_if_missing=True)
"""

from typing import Any, Dict, List, Optional

from common.config import config
from common.logging.logger import get_logger

logger = get_logger("meilisearch_manager")


class MeilisearchManager:
    """Lazy-connecting Meilisearch wrapper with optional index creation."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        create_if_missing: bool = False,
    ):
        self._url = url or config.get("meilisearch.url")
        self._api_key = api_key or config.get("meilisearch.api_key")
        self._index_name = index_name or config.get("meilisearch.index")
        self._create_if_missing = create_if_missing

        self._client = None
        self._index = None
        self._available = False
        self._initialized = False
        self._error_logged = False

    def _ensure_initialized(self):
        """Connects to Meilisearch on first access. Thread-safe via GIL."""
        if self._initialized:
            return
        self._initialized = True

        try:
            import meilisearch

            self._client = meilisearch.Client(self._url, self._api_key)

            # Health check
            try:
                self._client.health()
            except Exception as e:
                logger.warning(f"Meilisearch not reachable at {self._url}: {e}")
                self._client = None
                return

            # Get or create index
            try:
                self._index = self._client.get_index(self._index_name)
                self._available = True
                logger.info(f"Connected to Meilisearch index: {self._index_name}")
            except Exception:
                if self._create_if_missing:
                    try:
                        logger.info(f"Creating Meilisearch index: {self._index_name}")
                        task = self._client.create_index(
                            self._index_name, {'primaryKey': 'id'}
                        )
                        self._client.wait_for_task(task.task_uid, timeout_in_ms=30000)
                        self._index = self._client.get_index(self._index_name)
                        self._configure_index()
                        self._available = True
                    except Exception as e:
                        logger.warning(f"Meilisearch index creation failed: {e}")
                        self._client = None
                else:
                    logger.warning(
                        f"Meilisearch index '{self._index_name}' not found. "
                        "Use create_if_missing=True to create it."
                    )

        except ImportError:
            logger.warning("meilisearch package not installed. Run: pip install meilisearch")
            self._client = None
        except Exception as e:
            logger.warning(f"Meilisearch connection failed: {e}")
            self._client = None

    def _configure_index(self):
        """Configures searchable, filterable, and sortable attributes on the index."""
        if not self._index:
            return

        try:
            self._index.update_searchable_attributes([
                'title', 'summary', 'domain', 'url',
            ])
            self._index.update_filterable_attributes([
                'content_type', 'quality_score', 'domain', 'cluster_id',
            ])
            self._index.update_sortable_attributes([
                'quality_score',
            ])
            self._index.update_displayed_attributes([
                'id', 'title', 'url', 'domain', 'content_type',
                'quality_score', 'summary', 'cluster_id',
            ])
            logger.info("Meilisearch index attributes configured")
        except Exception as e:
            logger.warning(f"Meilisearch index configuration failed: {e}")

    # ---- properties ----

    @property
    def client(self):
        """Returns the meilisearch.Client or None if unavailable."""
        self._ensure_initialized()
        return self._client

    @property
    def index(self):
        """Returns the meilisearch Index object or None."""
        self._ensure_initialized()
        return self._index

    @property
    def available(self) -> bool:
        """True if connected and index exists."""
        self._ensure_initialized()
        return self._available

    @property
    def index_name(self) -> str:
        return self._index_name

    # ---- convenience methods ----

    def add_documents(self, documents: List[Dict[str, Any]]) -> Optional[Any]:
        """Adds or updates documents in the index. Meilisearch deduplicates by 'id'."""
        self._ensure_initialized()
        if not self._available or not self._index:
            return None

        try:
            return self._index.add_documents(documents)
        except Exception as e:
            if not self._error_logged:
                logger.warning(f"Meilisearch add_documents failed (further errors suppressed): {e}")
                self._error_logged = True
            return None

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        filter_str: Optional[str] = None,
        sort: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full-text search across indexed documents.

        Args:
            query: Search query string
            limit: Maximum results to return
            offset: Pagination offset
            filter_str: Meilisearch filter expression (e.g. "quality_score > 0.5")
            sort: Sort expressions (e.g. ["quality_score:desc"])

        Returns:
            Meilisearch response dict with 'hits', 'estimatedTotalHits', 'processingTimeMs'
        """
        self._ensure_initialized()
        if not self._available or not self._index:
            return {'hits': [], 'estimatedTotalHits': 0, 'processingTimeMs': 0}

        try:
            params = {'limit': limit, 'offset': offset}
            if filter_str:
                params['filter'] = filter_str
            if sort:
                params['sort'] = sort
            return self._index.search(query, params)
        except Exception as e:
            logger.error(f"Meilisearch search failed: {e}")
            return {'hits': [], 'estimatedTotalHits': 0, 'processingTimeMs': 0}

    def delete_documents(self, ids: List[str]) -> Optional[Any]:
        """Deletes documents by IDs."""
        self._ensure_initialized()
        if not self._available or not self._index:
            return None
        try:
            return self._index.delete_documents(ids)
        except Exception as e:
            logger.warning(f"Meilisearch delete failed: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Returns index statistics."""
        self._ensure_initialized()
        if not self._available or not self._index:
            return {'numberOfDocuments': 0, 'isIndexing': False}
        try:
            stats = self._index.get_stats()
            return {
                'numberOfDocuments': stats.number_of_documents,
                'isIndexing': stats.is_indexing,
            }
        except Exception:
            return {'numberOfDocuments': 0, 'isIndexing': False}
