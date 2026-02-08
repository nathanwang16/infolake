# Visualizer Module

Interactive deck.gl-based visualization for the Truth Atlas.

## Quick Start

```bash
# Start the server (reads config from config.json)
python scripts/start_visualizer.py

# Custom port
python scripts/start_visualizer.py --port 9000

# Open in browser
open http://localhost:8080
```

## Features

### Map View (deck.gl)
- **ScatterplotLayer** rendering UMAP coordinates via WebGL
- **OrthographicView** for non-geographic 2D data
- **Color coding** by cluster ID
- **Point size** proportional to importance (Z-axis)
- **LineLayer** for intra-cluster connections (togglable)

### Interactivity
| Control | Action |
|---------|--------|
| Scroll wheel | Zoom in/out |
| Click & drag | Pan the view |
| Click point | Open website in new tab |
| Hover point | Tooltip with title, cluster, quality, excerpt |
| Click cluster (sidebar) | Filter map to that cluster |
| Quality slider | Filter by minimum quality |
| Content type dropdown | Filter by document type |
| Mapping dropdown | Switch between mapping datasets |

### List View
- Document list with title, domain, quality, cluster
- Click to open website

### Sidebar
- **Statistics**: Map points, clusters, avg quality, DB count
- **Mapping selector**: Switch between available datasets
- **Filters**: Content type, quality threshold
- **Cluster list**: Click to filter, shows doc count per cluster

## Architecture

```
Browser (deck.gl)           Python Server (stdlib http.server)
┌──────────────┐            ┌────────────────────────────────┐
│ index.html   │  ──REST──▶ │  AtlasAPIHandler               │
│ deck.gl CDN  │            │    /api/stats                  │
│ ScatterplotL │            │    /api/mappings               │
│ OrthographicV│            │    /api/mapping-list           │
│              │            │    /api/documents              │
│              │            │    /api/clusters/:id           │
└──────────────┘            │    /api/search                 │
                            └───────────┬────────────────────┘
                                        │
                                   AtlasStore
                                (SQLite + Qdrant + Parquet)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Aggregate statistics |
| `/api/mappings` | GET | UMAP coordinates with metadata |
| `/api/mappings?mapping_id=X` | GET | Load specific mapping by filename |
| `/api/mapping-list` | GET | Available mapping datasets |
| `/api/documents` | GET | Paginated documents (limit, offset, content_type, min_quality) |
| `/api/documents/{id}` | GET | Single document |
| `/api/clusters` | GET | Cluster statistics |
| `/api/clusters/{id}` | GET | Documents in a cluster |
| `/api/search?q=X` | GET | Text search |

## Configuration (`config.json`)

```json
{
  "visualizer": {
    "host": "localhost",
    "port": 8080,
    "static_dir": "./visualizer/static"
  }
}
```

## File Structure

```
visualizer/
├── __init__.py          # Module exports
├── server.py            # HTTP server + REST API
├── README.md            # This file
└── static/
    └── index.html       # deck.gl frontend (single file)
```
