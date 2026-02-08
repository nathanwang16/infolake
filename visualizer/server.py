"""
Atlas Visualizer Server - deck.gl web UI and REST API.

Backend serves mapping data (UMAP coordinates), documents, and clusters.
Frontend renders via deck.gl ScatterplotLayer + OrthographicView.

Start:
    python scripts/start_visualizer.py
    python scripts/start_visualizer.py --port 8000
"""

import argparse
import json
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs
import threading

from common.logging.logger import get_logger, setup_logger
from common.config import config
from common.meilisearch_manager import MeilisearchManager
from storage.atlas_store import AtlasStore

logger = get_logger("visualizer")


class AtlasAPIHandler(SimpleHTTPRequestHandler):
    """
    HTTP request handler for Atlas API and static files.

    API Endpoints:
        GET /api/stats          - Atlas statistics
        GET /api/documents      - List documents (with filters)
        GET /api/documents/:id  - Single document
        GET /api/clusters       - Cluster statistics
        GET /api/clusters/:id   - Documents in cluster
        GET /api/mappings       - UMAP coordinates for deck.gl
        GET /api/mapping-list   - Available mapping datasets
        GET /api/search         - Text search (SQLite LIKE)
        GET /api/search/meili   - Full-text search (Meilisearch)
    """

    store: Optional[AtlasStore] = None
    static_dir: Optional[Path] = None

    def __init__(self, *args, directory=None, **kwargs):
        if AtlasAPIHandler.static_dir:
            directory = str(AtlasAPIHandler.static_dir)
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith('/api/'):
            self._handle_api(parsed.path, parsed.query)
        else:
            super().do_GET()

    def _handle_api(self, path: str, query_string: str):
        params = parse_qs(query_string)
        params = {k: v[0] if len(v) == 1 else v for k, v in params.items()}

        try:
            if path == '/api/stats':
                self._api_stats()
            elif path == '/api/documents':
                self._api_documents(params)
            elif path.startswith('/api/documents/'):
                self._api_document_detail(path.split('/')[-1])
            elif path == '/api/clusters':
                self._api_clusters()
            elif path.startswith('/api/clusters/'):
                self._api_cluster_documents(int(path.split('/')[-1]), params)
            elif path == '/api/mappings':
                self._api_mappings(params)
            elif path == '/api/mapping-list':
                self._api_mapping_list()
            elif path == '/api/search/meili':
                self._api_search_meili(params)
            elif path == '/api/search':
                self._api_search(params)
            else:
                self._send_error(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"API error on {path}: {e}")
            self._send_error(500, str(e))

    # --- Response helpers ---

    def _send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode('utf-8'))

    def _send_error(self, status: int, message: str):
        self._send_json({'error': message, 'status': status}, status)

    # --- Endpoints ---

    def _api_stats(self):
        self._send_json(self.store.get_atlas_stats())

    def _api_documents(self, params: Dict):
        limit = int(params.get('limit', 100))
        offset = int(params.get('offset', 0))
        content_type = params.get('content_type')
        min_quality = float(params.get('min_quality', 0))

        docs = self.store.get_documents(
            limit=limit, offset=offset,
            content_type=content_type,
            min_quality=min_quality if min_quality > 0 else None,
        )
        total = self.store.get_document_count(
            content_type=content_type,
            min_quality=min_quality if min_quality > 0 else None,
        )
        self._send_json({'documents': docs, 'total': total, 'limit': limit, 'offset': offset})

    def _api_document_detail(self, doc_id: str):
        doc = self.store.get_document(doc_id)
        if doc:
            self._send_json(doc)
        else:
            self._send_error(404, "Document not found")

    def _api_clusters(self):
        self._send_json({'clusters': self.store.get_cluster_stats()})

    def _api_cluster_documents(self, cluster_id: int, params: Dict):
        limit = int(params.get('limit', 100))
        docs = self.store.get_cluster_documents(cluster_id, limit=limit)
        self._send_json({'cluster_id': cluster_id, 'documents': docs})

    def _api_mappings(self, params: Dict):
        """Returns UMAP coordinates enriched with document metadata for deck.gl."""
        mapping_id = params.get('mapping_id')
        mappings = self._load_mappings(mapping_id)

        def is_missing(value: Optional[str]) -> bool:
            if value is None:
                return True
            if isinstance(value, str) and not value.strip():
                return True
            return False

        # Enrich with document info when metadata is missing
        needs_enrich = any(
            is_missing(m.get('title')) or
            is_missing(m.get('url')) or
            is_missing(m.get('domain')) or
            is_missing(m.get('content_type')) or
            is_missing(m.get('excerpt'))
            for m in mappings
        )
        if mappings and needs_enrich:
            try:
                doc_ids = [m.get('doc_id') for m in mappings if m.get('doc_id')]
                docs = self.store.get_documents_by_ids(doc_ids)
                docs_map = {d['id']: d for d in docs}
                missing_docs = 0
                for m in mappings:
                    doc = docs_map.get(m.get('doc_id'))
                    if not doc:
                        missing_docs += 1
                        continue
                    if is_missing(m.get('title')):
                        m['title'] = doc.get('title')
                    if is_missing(m.get('url')):
                        m['url'] = doc.get('url')
                    if is_missing(m.get('domain')):
                        m['domain'] = doc.get('domain')
                    if is_missing(m.get('content_type')):
                        m['content_type'] = doc.get('content_type')
                    if is_missing(m.get('excerpt')):
                        summary = doc.get('summary')
                        if summary:
                            m['excerpt'] = summary
                if missing_docs:
                    logger.warning(f"Mappings missing documents in DB: {missing_docs}")
            except Exception as e:
                logger.warning(f"Failed to enrich mappings: {e}")

        min_quality = float(params.get('min_quality', 0))
        if min_quality > 0:
            mappings = [m for m in mappings if m.get('quality_score', 0) >= min_quality]

        self._send_json({'mappings': mappings, 'count': len(mappings)})

    def _load_mappings(self, mapping_id: Optional[str] = None) -> List[Dict]:
        """Loads mapping data from parquet or JSON. mapping_id=None loads latest."""
        # Specific mapping by ID
        if mapping_id:
            result = self._load_mapping_by_id(mapping_id)
            if result:
                return result

        # Try ParquetStore (latest)
        try:
            mappings = self.store.parquet_store.load_mappings()
            if mappings:
                return mappings
        except Exception:
            pass

        # Fall back to JSON
        try:
            mappings_path = Path(config.get("mapping.output_path")).resolve()
            if mappings_path.exists():
                with open(mappings_path) as f:
                    return json.load(f).get('mappings', [])
        except Exception as e:
            logger.warning(f"Failed to load JSON mappings: {e}")

        return []

    def _load_mapping_by_id(self, mapping_id: str) -> List[Dict]:
        """Loads a specific mapping file by name."""
        # Try JSON in mappings directory
        try:
            mappings_dir = Path(config.get("paths.mappings_dir")).resolve()
            json_path = mappings_dir / mapping_id
            if json_path.exists():
                with open(json_path) as f:
                    return json.load(f).get('mappings', [])
        except Exception as e:
            logger.debug(f"Load by ID failed for {mapping_id}: {e}")
        return []

    def _api_mapping_list(self):
        """Lists available mapping datasets for the frontend switcher."""
        available: List[Dict[str, Any]] = []

        # Parquet manifest entries
        try:
            for entry in self.store.parquet_store._manifest.get('mappings', []):
                available.append({
                    'id': entry.get('filename', ''),
                    'name': entry.get('filename', '').replace('.parquet', ''),
                    'created_at': entry.get('created_at', ''),
                })
        except Exception:
            pass

        # JSON files in mappings directory
        try:
            mappings_dir = Path(config.get("paths.mappings_dir")).resolve()
            if mappings_dir.exists():
                for f in sorted(mappings_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                    if not any(m['id'] == f.name for m in available):
                        available.append({
                            'id': f.name,
                            'name': f.stem,
                            'created_at': datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                        })
        except Exception:
            pass

        self._send_json({'mappings': available, 'count': len(available)})

    def _api_search(self, params: Dict):
        query = params.get('q', '')
        limit = int(params.get('limit', 20))
        if not query:
            self._send_error(400, "Query parameter 'q' is required")
            return
        results = self.store.search_text_documents(query, limit)
        self._send_json({'query': query, 'results': results, 'count': len(results)})

    def _api_search_meili(self, params: Dict):
        """Full-text search via Meilisearch. Returns hits with doc IDs for map highlighting."""
        query = params.get('q', '')
        limit = int(params.get('limit', 20))
        offset = int(params.get('offset', 0))
        filter_str = params.get('filter')

        if not query:
            self._send_error(400, "Query parameter 'q' is required")
            return

        result = self.store.search_meilisearch(
            query=query, limit=limit, offset=offset, filter_str=filter_str,
        )
        self._send_json({
            'query': query,
            'hits': result.get('hits', []),
            'estimatedTotalHits': result.get('estimatedTotalHits', 0),
            'processingTimeMs': result.get('processingTimeMs', 0),
            'limit': limit,
            'offset': offset,
        })


class AtlasServer:
    """Atlas visualization server. Serves REST API + deck.gl static UI."""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8080,
        static_dir: Optional[str] = None,
        store: Optional[AtlasStore] = None,
    ):
        self.host = host
        self.port = port

        if static_dir:
            self.static_dir = Path(static_dir).resolve()
        else:
            self.static_dir = (Path(__file__).parent / 'static').resolve()

        self.static_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_index()

        self.store = store or AtlasStore()
        AtlasAPIHandler.store = self.store
        AtlasAPIHandler.static_dir = self.static_dir
        self.server = None

    def _ensure_index(self):
        """Creates a minimal fallback if index.html is missing."""
        index_path = self.static_dir / 'index.html'
        if index_path.exists():
            return
        index_path.write_text(
            '<!DOCTYPE html><html><head><title>Truth Atlas</title>'
            '<style>body{background:#1a1a2e;color:#eee;font-family:sans-serif;'
            'display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}'
            'code{background:#0f0f23;padding:2px 6px;border-radius:3px;color:#00d4ff}</style></head>'
            '<body><div style="text-align:center"><h1 style="color:#00d4ff">Truth Atlas</h1>'
            '<p>UI missing. Restore <code>visualizer/static/index.html</code></p>'
            '<p style="font-size:12px;color:#888;margin-top:12px">API active at <code>/api/stats</code></p>'
            '</div></body></html>'
        )
        logger.info(f"Created fallback UI at {index_path}")

    def start(self):
        self.server = HTTPServer((self.host, self.port), AtlasAPIHandler)
        logger.info(f"Starting Atlas Visualizer at http://{self.host}:{self.port}")
        print(f"\n{'='*60}")
        print(f"  Truth Atlas Visualizer")
        print(f"  Running at: http://{self.host}:{self.port}")
        print(f"  Press Ctrl+C to stop")
        print(f"{'='*60}\n")
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
            self.server.shutdown()

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread


def main():
    parser = argparse.ArgumentParser(description="Atlas Visualizer Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--static-dir", help="Static files directory")
    args = parser.parse_args()

    setup_logger("visualizer", console_output=True)
    server = AtlasServer(host=args.host, port=args.port, static_dir=args.static_dir)
    server.start()


if __name__ == "__main__":
    main()
