"""
Base functor class for source adapters.

In category-theoretic terms, each dump adapter is a functor
``Source → Atlas`` that maps heterogeneous data sources into the atlas's
uniform Record type while preserving the source's structural properties.

Concrete adapters subclass :class:`BaseFunctor` and implement
:meth:`_read_source`.  The base class handles common concerns (path
validation, error wrapping, logging) so adapters stay focused on
parsing.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Iterator

from atlas_core.errors import AtlasFunctorError
from atlas_core.logging import get_logger
from atlas_core.types import Record, URL

logger = get_logger("atlas_core.functors")


class BaseFunctor:
    """
    Abstract base for all source-to-atlas functors.

    Subclasses must implement:
        * ``source_name`` — short identifier (e.g. ``"slop"``, ``"dmoz"``).
        * ``_read_source(path)`` — yields :class:`Record` instances from the
          raw source file.

    The public :meth:`read` method wraps ``_read_source`` with path
    validation, error handling, and progress logging.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier for this source (e.g. ``'slop'``, ``'dmoz'``)."""
        ...

    @abstractmethod
    def _read_source(self, path: Path) -> Iterator[Record]:
        """
        Yield :class:`Record` instances from the source at *path*.

        Implementations should **not** catch broad exceptions — let
        :meth:`read` wrap them in :class:`AtlasFunctorError`.
        """
        ...

    def read(self, path: str) -> Iterator[Record]:
        """
        Public entry point.  Validates *path*, delegates to
        :meth:`_read_source`, and wraps errors.

        Args:
            path: Filesystem path to the source dump.

        Yields:
            :class:`Record` instances in atlas-uniform format.

        Raises:
            AtlasFunctorError: On missing path or adapter-level failure.
        """
        p = Path(path)
        if not p.exists():
            raise AtlasFunctorError(self.source_name, f"path does not exist: {path}")

        logger.info("[%s] Reading source: %s", self.source_name, path)
        count = 0

        try:
            for record in self._read_source(p):
                count += 1
                if count % 10_000 == 0:
                    logger.info("[%s] Yielded %d records", self.source_name, count)
                yield record
        except AtlasFunctorError:
            raise
        except Exception as exc:
            raise AtlasFunctorError(self.source_name, str(exc)) from exc

        logger.info("[%s] Finished — %d records total", self.source_name, count)

    def map_url(self, raw_url: str) -> URL:
        """
        Normalise a raw URL string into the atlas URL type.

        Basic normalisation: strip whitespace, lowercase scheme+host,
        remove trailing slash, remove fragment.
        """
        url = raw_url.strip()
        if not url:
            raise AtlasFunctorError(self.source_name, "empty URL")

        # Lowercase scheme + host (but not path)
        if "://" in url:
            scheme_host, _, rest = url.partition("://")
            scheme = scheme_host.lower()
            if "/" in rest:
                host, _, path_and_query = rest.partition("/")
                host = host.lower()
                url = f"{scheme}://{host}/{path_and_query}"
            else:
                url = f"{scheme}://{rest.lower()}"

        # Strip fragment
        url = url.split("#")[0]

        # Strip trailing slash (unless it's the root)
        if url.endswith("/") and url.count("/") > 3:
            url = url.rstrip("/")

        return URL(url)
