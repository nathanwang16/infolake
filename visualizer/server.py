"""
Atlas Visualizer Server - Web UI and API for exploring the atlas.

Implements Section 6.4 and Appendix B of Technical Engineering Guide.

Provides:
- REST API for documents, clusters, and mappings
- Static file serving for web UI
- Export endpoints for visualization data

Usage:
    python -m visualizer.server --port 8080
    python -m visualizer.server --host 0.0.0.0 --port 8080
"""

import argparse
import json
import os
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs
import threading

from common.logging.logger import get_logger, setup_logger
from common.config import config
from storage.atlas_store import AtlasStore

logger = get_logger("visualizer")


class AtlasAPIHandler(SimpleHTTPRequestHandler):
    """
    HTTP request handler for Atlas API and static files.
    
    API Endpoints:
        GET /api/stats          - Atlas statistics
        GET /api/documents      - List documents (with filters)
        GET /api/documents/:id  - Get single document
        GET /api/clusters       - Cluster statistics
        GET /api/clusters/:id   - Documents in cluster
        GET /api/mappings       - UMAP coordinates for visualization
        GET /api/search         - Search documents by query
        
    Static Files:
        GET /                   - Main visualization page
        GET /static/*           - Static assets (CSS, JS)
    """
    
    # Class-level state (shared across requests).
    # NOTE: SimpleHTTPRequestHandler creates a new instance per request, so instance
    # attributes are lost between requests.  Class-level attributes are the standard
    # pattern for sharing state (like `store`) across requests with the stdlib
    # http.server module.  If the server needs to handle concurrent requests or
    # per-request isolation, migrate to a WSGI/ASGI framework (e.g. Flask, FastAPI).
    store: Optional[AtlasStore] = None
    static_dir: Optional[Path] = None
    
    def __init__(self, *args, directory=None, **kwargs):
        # Set directory for static files (use absolute path)
        if AtlasAPIHandler.static_dir:
            directory = str(AtlasAPIHandler.static_dir)
        super().__init__(*args, directory=directory, **kwargs)
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # API routes
        if path.startswith('/api/'):
            self._handle_api(path, parsed.query)
        else:
            # Static files
            super().do_GET()
    
    def _handle_api(self, path: str, query_string: str):
        """Routes API requests."""
        params = parse_qs(query_string)
        
        # Convert single-value lists to values
        params = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
        
        try:
            if path == '/api/stats':
                self._api_stats()
            elif path == '/api/documents':
                self._api_documents(params)
            elif path.startswith('/api/documents/'):
                doc_id = path.split('/')[-1]
                self._api_document_detail(doc_id)
            elif path == '/api/clusters':
                self._api_clusters()
            elif path.startswith('/api/clusters/'):
                cluster_id = int(path.split('/')[-1])
                self._api_cluster_documents(cluster_id, params)
            elif path == '/api/mappings':
                self._api_mappings(params)
            elif path == '/api/search':
                self._api_search(params)
            else:
                self._send_error(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"API error: {e}")
            self._send_error(500, str(e))
    
    def _send_json(self, data: Any, status: int = 200):
        """Sends JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = json.dumps(data, default=str)
        self.wfile.write(response.encode('utf-8'))
    
    def _send_error(self, status: int, message: str):
        """Sends error response."""
        self._send_json({'error': message, 'status': status}, status)
    
    def _api_stats(self):
        """GET /api/stats - Returns atlas statistics."""
        stats = self.store.get_atlas_stats()
        self._send_json(stats)
    
    def _api_documents(self, params: Dict):
        """GET /api/documents - Returns paginated documents."""
        limit = int(params.get('limit', 100))
        offset = int(params.get('offset', 0))
        content_type = params.get('content_type')
        min_quality = float(params.get('min_quality', 0))
        
        docs = self.store.get_documents(
            limit=limit,
            offset=offset,
            content_type=content_type,
            min_quality=min_quality if min_quality > 0 else None
        )
        
        total = self.store.get_document_count(
            content_type=content_type,
            min_quality=min_quality if min_quality > 0 else None
        )
        
        self._send_json({
            'documents': docs,
            'total': total,
            'limit': limit,
            'offset': offset,
        })
    
    def _api_document_detail(self, doc_id: str):
        """GET /api/documents/:id - Returns single document."""
        doc = self.store.get_document(doc_id)
        if doc:
            self._send_json(doc)
        else:
            self._send_error(404, "Document not found")
    
    def _api_clusters(self):
        """GET /api/clusters - Returns cluster statistics."""
        clusters = self.store.get_cluster_stats()
        self._send_json({'clusters': clusters})
    
    def _api_cluster_documents(self, cluster_id: int, params: Dict):
        """GET /api/clusters/:id - Returns documents in cluster."""
        limit = int(params.get('limit', 100))
        docs = self.store.get_cluster_documents(cluster_id, limit=limit)
        self._send_json({
            'cluster_id': cluster_id,
            'documents': docs,
        })
    
    def _api_mappings(self, params: Dict):
        """GET /api/mappings - Returns UMAP coordinates."""
        mappings = []
        
        # Try ParquetStore first
        try:
            mappings = self.store.parquet_store.load_mappings()
        except Exception:
            pass
        
        # Fall back to JSON file
        if not mappings:
            try:
                mappings_path = config.get("mapping.output_path")
                mappings_path = Path(mappings_path).resolve()
                if mappings_path.exists():
                    with open(mappings_path) as f:
                        data = json.load(f)
                        mappings = data.get('mappings', [])
            except Exception as e:
                logger.warning(f"Failed to load JSON mappings: {e}")
        
        # Optionally filter by quality
        min_quality = float(params.get('min_quality', 0))
        if min_quality > 0:
            mappings = [m for m in mappings if m.get('quality_score', 0) >= min_quality]
        
        self._send_json({
            'mappings': mappings,
            'count': len(mappings),
        })
    
    def _api_search(self, params: Dict):
        """GET /api/search - Search documents by text query."""
        query = params.get('q', '')
        limit = int(params.get('limit', 20))

        if not query:
            self._send_error(400, "Query parameter 'q' is required")
            return

        results = self.store.search_text_documents(query, limit)

        self._send_json({
            'query': query,
            'results': results,
            'count': len(results),
        })


class AtlasServer:
    """
    Atlas visualization server.
    
    Serves both API endpoints and static web UI.
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8080,
        static_dir: Optional[str] = None,
        store: Optional[AtlasStore] = None,
    ):
        self.host = host
        self.port = port

        # Static files directory
        if static_dir:
            self.static_dir = Path(static_dir).resolve()
        else:
            # Use default visualizer/static (absolute path)
            self.static_dir = (Path(__file__).parent / 'static').resolve()

        # Ensure static dir exists
        self.static_dir.mkdir(parents=True, exist_ok=True)

        # Generate default index.html if not exists
        self._ensure_static_files()

        # Initialize store
        self.store = store or AtlasStore()
        
        # Configure handler
        AtlasAPIHandler.store = self.store
        AtlasAPIHandler.static_dir = self.static_dir
        
        self.server = None
    
    def _ensure_static_files(self):
        """Creates default static files if they don't exist."""
        index_path = self.static_dir / 'index.html'
        
        if not index_path.exists():
            self._create_default_ui()
    
    def _create_default_ui(self):
        """Creates a default visualization UI."""
        index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Truth Atlas</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; 
            color: #eee; 
            min-height: 100vh;
        }
        .container { max-width: 1600px; margin: 0 auto; padding: 15px; }
        .main-content { display: grid; grid-template-columns: 280px 1fr; gap: 15px; }
        
        .sidebar { 
            background: #16213e; 
            border-radius: 8px; 
            padding: 15px;
            height: fit-content;
            max-height: calc(100vh - 30px);
            overflow-y: auto;
        }
        .sidebar h2 { font-size: 14px; margin-bottom: 12px; color: #00d4ff; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }
        .stat-box {
            background: #0f0f23;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-value { font-size: 20px; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 10px; color: #888; margin-top: 3px; }
        
        .filter-group { margin-bottom: 15px; }
        .filter-group label { display: block; font-size: 11px; color: #888; margin-bottom: 6px; }
        .filter-group select, .filter-group input[type="range"] {
            width: 100%;
            padding: 8px;
            background: #0f0f23;
            border: 1px solid #333;
            border-radius: 4px;
            color: #eee;
        }
        
        .cluster-list { list-style: none; max-height: 300px; overflow-y: auto; }
        .cluster-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 10px;
            margin: 4px 0;
            background: #0f0f23;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 13px;
        }
        .cluster-item:hover { background: #1a1a3e; }
        .cluster-item.active { background: #00d4ff22; border-left: 3px solid #00d4ff; }
        .cluster-count { color: #888; font-size: 11px; }
        
        .main-panel { 
            background: #16213e; 
            border-radius: 8px; 
            padding: 15px;
            min-height: calc(100vh - 30px);
            display: flex;
            flex-direction: column;
        }
        
        .tab-bar { display: flex; gap: 8px; margin-bottom: 15px; }
        .tab {
            padding: 8px 16px;
            background: #0f0f23;
            border: none;
            border-radius: 4px;
            color: #888;
            cursor: pointer;
            font-size: 13px;
        }
        .tab.active { background: #00d4ff; color: #000; }
        
        #map-container {
            flex: 1;
            background: #0f0f23;
            border-radius: 8px;
            position: relative;
            overflow: hidden;
            cursor: grab;
            min-height: 500px;
        }
        #map-container:active { cursor: grabbing; }
        
        #map-canvas {
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
        }
        
        .map-controls {
            position: absolute;
            bottom: 10px; right: 10px;
            display: flex;
            gap: 5px;
            z-index: 50;
        }
        .map-btn {
            width: 32px; height: 32px;
            background: #16213e;
            border: 1px solid #333;
            border-radius: 4px;
            color: #eee;
            cursor: pointer;
            font-size: 16px;
        }
        .map-btn:hover { background: #1a1a3e; }
        
        .zoom-info {
            position: absolute;
            top: 10px; left: 10px;
            background: #16213ecc;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 11px;
            color: #888;
        }
        
        .document-list { list-style: none; overflow-y: auto; flex: 1; }
        .document-item {
            padding: 12px;
            margin: 8px 0;
            background: #0f0f23;
            border-radius: 4px;
            border-left: 3px solid #00d4ff;
        }
        .document-title { font-size: 14px; margin-bottom: 4px; }
        .document-title a { color: #00d4ff; text-decoration: none; }
        .document-title a:hover { text-decoration: underline; }
        .document-meta { font-size: 11px; color: #888; }
        .quality-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 10px;
            margin-left: 8px;
        }
        .quality-high { background: #00ff88; color: #000; }
        .quality-medium { background: #ffcc00; color: #000; }
        .quality-low { background: #ff4444; color: #fff; }
        
        .tooltip {
            position: fixed;
            background: #000e;
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 12px;
            max-width: 320px;
            z-index: 1000;
            pointer-events: none;
            display: none;
            border: 1px solid #333;
            line-height: 1.4;
        }
        .tooltip-title { color: #00d4ff; font-weight: bold; margin-bottom: 5px; }
        .tooltip-excerpt { color: #aaa; font-size: 11px; margin: 8px 0; font-style: italic; }
        .tooltip-meta { color: #888; font-size: 10px; }
        
        #list-view { display: none; flex: 1; flex-direction: column; }
        
        .spinner {
            width: 40px; height: 40px;
            border: 3px solid #333;
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 50px auto;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-content">
            <aside class="sidebar">
                <h2>Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value" id="stat-points">-</div>
                        <div class="stat-label">Map Points</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="stat-clusters">-</div>
                        <div class="stat-label">Clusters</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="stat-quality">-</div>
                        <div class="stat-label">Avg Quality</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="stat-docs">-</div>
                        <div class="stat-label">In Database</div>
                    </div>
                </div>
                
                <h2>Filters</h2>
                <div class="filter-group">
                    <label>Content Type</label>
                    <select id="filter-type">
                        <option value="">All Types</option>
                        <option value="technical_code">Technical/Code</option>
                        <option value="scientific">Scientific</option>
                        <option value="news">News</option>
                        <option value="blog">Blog/Essay</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Min Quality Score: <span id="quality-value">0%</span></label>
                    <input type="range" id="filter-quality" min="0" max="100" value="0">
                </div>
                
                <h2 style="margin-top: 20px;">Clusters</h2>
                <ul class="cluster-list" id="cluster-list">
                    <li class="cluster-item">Loading...</li>
                </ul>
            </aside>
            
            <main class="main-panel">
                <div class="tab-bar">
                    <button class="tab active" data-view="map">Map View</button>
                    <button class="tab" data-view="list">List View</button>
                </div>
                
                <div id="map-container">
                    <canvas id="map-canvas"></canvas>
                    <div class="zoom-info">Scroll to zoom, drag to pan</div>
                    <div class="map-controls">
                        <button class="map-btn" id="toggle-lines" title="Toggle Connections" style="font-size:12px">⟟</button>
                        <button class="map-btn" id="zoom-in" title="Zoom In">+</button>
                        <button class="map-btn" id="zoom-out" title="Zoom Out">−</button>
                        <button class="map-btn" id="reset-view" title="Reset View">⟲</button>
                    </div>
                </div>
                
                <div id="list-view">
                    <ul class="document-list" id="document-list"></ul>
                </div>
            </main>
        </div>
    </div>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const API_BASE = '/api';
        let allMappings = [];
        let filteredMappings = [];
        let documentsData = [];
        let clusterStats = {};
        
        // Canvas state for pan/zoom
        let canvas, ctx;
        let scale = 1, offsetX = 0, offsetY = 0;
        let isDragging = false, lastX = 0, lastY = 0;
        let minQuality = 0, selectedType = '';
        
        // Colors for clusters
        const clusterColors = [
            '#00d4ff', '#00ff88', '#ff6b6b', '#ffd93d', '#6bcbff',
            '#c9b1ff', '#ff9f43', '#54a0ff', '#5f27cd', '#00d2d3',
            '#ff6b81', '#7bed9f', '#70a1ff', '#ffa502', '#2ed573'
        ];
        
        function getClusterColor(id) {
            if (id === -1) return '#666';
            return clusterColors[Math.abs(id) % clusterColors.length];
        }
        
        // Load stats
        async function loadStats() {
            try {
                const res = await fetch(`${API_BASE}/stats`);
                const data = await res.json();
                
                document.getElementById('stat-docs').textContent = 
                    data.total_documents?.toLocaleString() || '0';
                
                const qdist = data.quality_distribution || {};
                const total = (qdist.high||0) + (qdist.medium||0) + (qdist.low||0);
                const avgQ = total > 0 ? 
                    ((qdist.high*0.85 + qdist.medium*0.55 + qdist.low*0.2) / total * 100).toFixed(0) : 0;
                document.getElementById('stat-quality').textContent = avgQ + '%';
            } catch (e) { console.error('Stats error:', e); }
        }
        
        // Update stats from loaded mappings
        function updateStatsFromMappings() {
            document.getElementById('stat-points').textContent = allMappings.length.toLocaleString();
            
            // Compute avg quality from mappings
            if (allMappings.length) {
                const avgQ = allMappings.reduce((s,m) => s + (m.quality_score||0), 0) / allMappings.length;
                document.getElementById('stat-quality').textContent = (avgQ * 100).toFixed(0) + '%';
            }
        }
        
        // Load clusters from mappings data
        function updateClustersFromMappings() {
            clusterStats = {};
            allMappings.forEach(m => {
                const cid = m.cluster_id ?? -1;
                if (!clusterStats[cid]) clusterStats[cid] = { count: 0, quality_sum: 0 };
                clusterStats[cid].count++;
                clusterStats[cid].quality_sum += (m.quality_score || 0);
            });
            
            const list = document.getElementById('cluster-list');
            list.innerHTML = '';
            
            const sorted = Object.entries(clusterStats)
                .map(([id, s]) => ({ id: parseInt(id), count: s.count, avgQ: s.quality_sum/s.count }))
                .sort((a,b) => b.count - a.count);
            
            document.getElementById('stat-clusters').textContent = sorted.filter(c => c.id !== -1).length;
            
            sorted.forEach(c => {
                const li = document.createElement('li');
                li.className = 'cluster-item';
                li.dataset.clusterId = c.id;
                li.innerHTML = `
                    <span style="color: ${getClusterColor(c.id)}">
                        ${c.id === -1 ? 'Orphaned' : 'Cluster ' + c.id}
                    </span>
                    <span class="cluster-count">${c.count} docs</span>
                `;
                li.onclick = () => filterByCluster(c.id);
                list.appendChild(li);
            });
        }
        
        let activeClusterId = null;
        
        function filterByCluster(clusterId) {
            // Toggle - click same cluster to deselect
            if (activeClusterId === clusterId) {
                activeClusterId = null;
                document.querySelectorAll('.cluster-item').forEach(el => el.classList.remove('active'));
                applyFilters(null);
                loadDocuments({ content_type: selectedType, min_quality: minQuality, limit: 50 });
            } else {
                activeClusterId = clusterId;
                document.querySelectorAll('.cluster-item').forEach(el => {
                    el.classList.toggle('active', parseInt(el.dataset.clusterId) === clusterId);
                });
                applyFilters(clusterId);
                // Load documents for this cluster
                loadClusterDocuments(clusterId);
            }
        }
        
        async function loadClusterDocuments(clusterId) {
            try {
                const res = await fetch(`${API_BASE}/clusters/${clusterId}`);
                const data = await res.json();
                documentsData = data.documents || [];
                renderDocumentList();
            } catch(e) { console.error('Cluster docs error:', e); }
        }
        
        // Load mappings with document details
        async function loadMappings() {
            try {
                const res = await fetch(`${API_BASE}/mappings`);
                const data = await res.json();
                allMappings = data.mappings || [];
                
                // Enrich with document data
                const docsRes = await fetch(`${API_BASE}/documents?limit=1000`);
                const docsData = await docsRes.json();
                const docsMap = {};
                (docsData.documents || []).forEach(d => { docsMap[d.id] = d; });
                
                allMappings.forEach(m => {
                    const doc = docsMap[m.doc_id];
                    if (doc) {
                        m.title = doc.title;
                        m.url = doc.url;
                        m.domain = doc.domain;
                        m.content_type = doc.content_type;
                        m.excerpt = doc.summary || '';
                    }
                });
                
                filteredMappings = [...allMappings];
                updateClustersFromMappings();
                updateStatsFromMappings();
                initCanvas();
                renderCanvas();
            } catch (e) {
                console.error('Mappings error:', e);
            }
        }
        
        // Apply filters
        function applyFilters(clusterId = null) {
            filteredMappings = allMappings.filter(m => {
                if (minQuality > 0 && (m.quality_score || 0) < minQuality) return false;
                if (selectedType && m.content_type !== selectedType) return false;
                if (clusterId !== null && m.cluster_id !== clusterId) return false;
                return true;
            });
            renderCanvas();
        }
        
        // Canvas rendering
        function initCanvas() {
            canvas = document.getElementById('map-canvas');
            ctx = canvas.getContext('2d');
            resizeCanvas();
            window.addEventListener('resize', () => { resizeCanvas(); renderCanvas(); });
            
            // Mouse events for pan
            canvas.addEventListener('mousedown', e => {
                isDragging = true;
                lastX = e.clientX; lastY = e.clientY;
                canvas.style.cursor = 'grabbing';
            });
            canvas.addEventListener('mousemove', e => {
                if (isDragging) {
                    offsetX += e.clientX - lastX;
                    offsetY += e.clientY - lastY;
                    lastX = e.clientX; lastY = e.clientY;
                    renderCanvas();
                } else {
                    handleHover(e);
                }
            });
            canvas.addEventListener('mouseup', () => { isDragging = false; canvas.style.cursor = 'grab'; });
            canvas.addEventListener('mouseleave', () => { isDragging = false; hideTooltip(); });
            
            // Scroll zoom
            canvas.addEventListener('wheel', e => {
                e.preventDefault();
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;
                
                const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
                const newScale = Math.max(0.2, Math.min(10, scale * zoomFactor));
                
                // Zoom towards mouse position
                offsetX = mx - (mx - offsetX) * (newScale / scale);
                offsetY = my - (my - offsetY) * (newScale / scale);
                scale = newScale;
                renderCanvas();
            });
            
            // Click to open URL
            canvas.addEventListener('click', e => {
                const hit = getPointAtPos(e);
                if (hit && hit.url) window.open(hit.url, '_blank');
            });
            
            // Buttons
            document.getElementById('zoom-in').onclick = () => { scale *= 1.3; renderCanvas(); };
            document.getElementById('zoom-out').onclick = () => { scale *= 0.7; renderCanvas(); };
            document.getElementById('reset-view').onclick = () => { scale = 1; offsetX = 0; offsetY = 0; renderCanvas(); };
            document.getElementById('toggle-lines').onclick = () => { 
                showConnections = !showConnections; 
                document.getElementById('toggle-lines').style.background = showConnections ? '#00d4ff33' : '#16213e';
                renderCanvas(); 
            };
        }
        
        function resizeCanvas() {
            const container = document.getElementById('map-container');
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        }
        
        function getPointAtPos(e) {
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left, my = e.clientY - rect.top;
            
            const xs = filteredMappings.map(m => m.x);
            const ys = filteredMappings.map(m => m.y);
            if (!xs.length) return null;
            
            const minX = Math.min(...xs), maxX = Math.max(...xs);
            const minY = Math.min(...ys), maxY = Math.max(...ys);
            const padding = 50;
            
            for (const m of filteredMappings) {
                const x = padding + ((m.x - minX) / (maxX - minX || 1)) * (canvas.width - 2*padding);
                const y = padding + ((m.y - minY) / (maxY - minY || 1)) * (canvas.height - 2*padding);
                const sx = x * scale + offsetX;
                const sy = y * scale + offsetY;
                const r = (4 + (m.z || 0.5) * 6) * scale;
                
                const dist = Math.sqrt((mx - sx)**2 + (my - sy)**2);
                if (dist <= r + 5) return m;
            }
            return null;
        }
        
        function handleHover(e) {
            const m = getPointAtPos(e);
            if (m) {
                showTooltip(e, m);
                canvas.style.cursor = 'pointer';
            } else {
                hideTooltip();
                canvas.style.cursor = isDragging ? 'grabbing' : 'grab';
            }
        }
        
        let showConnections = true; // Toggle for connection lines
        
        function renderCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (!filteredMappings.length) {
                ctx.fillStyle = '#666';
                ctx.font = '14px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('No data matching filters', canvas.width/2, canvas.height/2);
                return;
            }
            
            const xs = filteredMappings.map(m => m.x);
            const ys = filteredMappings.map(m => m.y);
            const minX = Math.min(...xs), maxX = Math.max(...xs);
            const minY = Math.min(...ys), maxY = Math.max(...ys);
            const padding = 50;
            
            // Precompute screen positions
            const positions = filteredMappings.map(m => {
                const x = padding + ((m.x - minX) / (maxX - minX || 1)) * (canvas.width - 2*padding);
                const y = padding + ((m.y - minY) / (maxY - minY || 1)) * (canvas.height - 2*padding);
                return { 
                    sx: x * scale + offsetX, 
                    sy: y * scale + offsetY,
                    m 
                };
            });
            
            // Draw intra-cluster connections (thin lines between nearby points in same cluster)
            if (showConnections && scale > 0.5) {
                ctx.globalAlpha = 0.15;
                ctx.lineWidth = 0.5 * scale;
                
                // Group by cluster
                const clusterGroups = {};
                positions.forEach((p, i) => {
                    const cid = p.m.cluster_id ?? -1;
                    if (cid === -1) return; // Skip orphans
                    if (!clusterGroups[cid]) clusterGroups[cid] = [];
                    clusterGroups[cid].push({ ...p, idx: i });
                });
                
                // Draw connections within each cluster (k-nearest in screen space)
                Object.entries(clusterGroups).forEach(([cid, points]) => {
                    if (points.length < 2) return;
                    ctx.strokeStyle = getClusterColor(parseInt(cid));
                    
                    // For each point, connect to 2 nearest neighbors in same cluster
                    points.forEach(p1 => {
                        const neighbors = points
                            .filter(p2 => p2.idx !== p1.idx)
                            .map(p2 => ({ p: p2, dist: Math.hypot(p1.sx - p2.sx, p1.sy - p2.sy) }))
                            .sort((a, b) => a.dist - b.dist)
                            .slice(0, 2);
                        
                        neighbors.forEach(n => {
                            if (n.dist < 100 * scale) { // Only draw if close enough
                                ctx.beginPath();
                                ctx.moveTo(p1.sx, p1.sy);
                                ctx.lineTo(n.p.sx, n.p.sy);
                                ctx.stroke();
                            }
                        });
                    });
                });
                ctx.globalAlpha = 1;
            }
            
            // Draw points
            positions.forEach(({ sx, sy, m }) => {
                const r = (4 + (m.z || 0.5) * 6) * scale;
                
                ctx.beginPath();
                ctx.arc(sx, sy, r, 0, Math.PI * 2);
                ctx.fillStyle = getClusterColor(m.cluster_id ?? -1);
                ctx.globalAlpha = 0.8;
                ctx.fill();
                ctx.globalAlpha = 1;
            });
        }
        
        function showTooltip(e, m) {
            const tooltip = document.getElementById('tooltip');
            tooltip.style.display = 'block';
            
            // Position tooltip, keeping it on screen
            let left = e.clientX + 15;
            let top = e.clientY + 15;
            if (left + 320 > window.innerWidth) left = e.clientX - 330;
            if (top + 150 > window.innerHeight) top = e.clientY - 160;
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
            
            const title = m.title || m.url?.split('/').pop() || `Point ${m.doc_id?.substring(0,8)}`;
            const excerpt = m.excerpt ? m.excerpt.substring(0, 120) + '...' : '';
            const domain = m.domain || (m.url ? new URL(m.url).hostname : '');
            
            tooltip.innerHTML = `
                <div class="tooltip-title">${title}</div>
                ${excerpt ? `<div class="tooltip-excerpt">"${excerpt}"</div>` : ''}
                <div class="tooltip-meta">
                    ${domain ? `<span style="color:#00d4ff">${domain}</span><br>` : ''}
                    Quality: ${((m.quality_score||0) * 100).toFixed(0)}% · 
                    Cluster: ${m.cluster_id === -1 ? 'Orphaned' : m.cluster_id} ·
                    Importance: ${((m.z||0.5) * 100).toFixed(0)}%
                    ${m.url ? '<br><span style="color:#888;font-size:9px">Click to open website</span>' : ''}
                </div>
            `;
        }
        
        function hideTooltip() {
            document.getElementById('tooltip').style.display = 'none';
        }
        
        // Documents list
        async function loadDocuments(params = {}) {
            const query = new URLSearchParams(params).toString();
            const res = await fetch(`${API_BASE}/documents?${query}`);
            const data = await res.json();
            documentsData = data.documents || [];
            renderDocumentList();
        }
        
        function renderDocumentList() {
            const list = document.getElementById('document-list');
            list.innerHTML = '';
            
            documentsData.forEach(doc => {
                const li = document.createElement('li');
                li.className = 'document-item';
                
                const quality = doc.quality_score || 0;
                let badgeClass = quality >= 0.7 ? 'quality-high' : quality >= 0.4 ? 'quality-medium' : 'quality-low';
                
                li.innerHTML = `
                    <div class="document-title">
                        <a href="${doc.url}" target="_blank">${doc.title || 'Untitled'}</a>
                        <span class="quality-badge ${badgeClass}">${(quality * 100).toFixed(0)}%</span>
                    </div>
                    <div class="document-meta">
                        ${doc.domain || ''} · ${doc.content_type || 'unknown'} · Cluster ${doc.cluster_id ?? 'N/A'}
                    </div>
                `;
                list.appendChild(li);
            });
        }
        
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.onclick = () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const view = tab.dataset.view;
                document.getElementById('map-container').style.display = view === 'map' ? 'block' : 'none';
                document.getElementById('list-view').style.display = view === 'list' ? 'flex' : 'none';
                if (view === 'list' && !documentsData.length) loadDocuments({ limit: 50 });
            };
        });
        
        // Filters
        document.getElementById('filter-type').onchange = (e) => {
            selectedType = e.target.value;
            applyFilters();
            loadDocuments({ content_type: selectedType, min_quality: minQuality, limit: 50 });
        };
        
        document.getElementById('filter-quality').oninput = (e) => {
            document.getElementById('quality-value').textContent = e.target.value + '%';
        };
        
        document.getElementById('filter-quality').onchange = (e) => {
            minQuality = e.target.value / 100;
            applyFilters();
            loadDocuments({ content_type: selectedType, min_quality: minQuality, limit: 50 });
        };
        
        // Initialize
        loadStats();
        loadMappings();
    </script>
</body>
</html>
'''
        
        index_path = self.static_dir / 'index.html'
        with open(index_path, 'w') as f:
            f.write(index_html)
        
        logger.info(f"Created default UI at {index_path}")
    
    def start(self):
        """Starts the server."""
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
        """Starts server in background thread."""
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Atlas Visualizer Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server on default port
    python -m visualizer.server
    
    # Start on custom port
    python -m visualizer.server --port 8000
    
    # Bind to all interfaces
    python -m visualizer.server --host 0.0.0.0 --port 8080
"""
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)"
    )
    parser.add_argument(
        "--static-dir",
        help="Directory for static files"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger("visualizer", console_output=True)
    
    # Start server
    server = AtlasServer(
        host=args.host,
        port=args.port,
        static_dir=args.static_dir
    )
    server.start()


if __name__ == "__main__":
    main()
