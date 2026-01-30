# Visualizer Module

Interactive web-based visualization for the Truth Atlas.

## Features

### Map View (Canvas-based)
- **2D scatter plot** of UMAP coordinates
- **Color coding** by cluster
- **Size** proportional to importance (Z-axis)
- **Intra-cluster connections** showing semantic relationships

### Interactivity
| Control | Action |
|---------|--------|
| Scroll wheel | Zoom in/out |
| Click & drag | Pan the view |
| Click point | Open website in new tab |
| Hover point | Show tooltip with metadata |
| Click cluster | Filter to that cluster |
| Quality slider | Filter by minimum quality |
| Content type | Filter by document type |

### List View
- Sortable document list
- Click to open website
- Shows title, domain, quality, cluster

### Statistics Panel
- Map Points: Total visualized points
- Clusters: Number of topic clusters
- Avg Quality: Mean quality score
- In Database: Documents in SQLite

## Architecture

```
┌─────────────────────────────────────────────────┐
│                AtlasServer                       │
│  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ Static Files│  │      REST API            │  │
│  │ (HTML/JS)   │  │ /api/documents           │  │
│  │             │  │ /api/mappings            │  │
│  │             │  │ /api/stats               │  │
│  │             │  │ /api/clusters/{id}       │  │
│  └─────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────┘
         ▲                    ▲
         │                    │
    index.html          AtlasStore
    (generated)        (SQLite + Qdrant)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/documents` | GET | List documents with pagination |
| `/api/documents/{id}` | GET | Single document details |
| `/api/mappings` | GET | All 2D/3D coordinates |
| `/api/stats` | GET | Aggregate statistics |
| `/api/clusters/{id}` | GET | Documents in a cluster |
| `/api/search` | POST | Semantic search |

## Configuration (`config.json`)

```json
{
  "visualizer": {
    "host": "0.0.0.0",
    "port": 8080,
    "static_dir": "visualizer/static"
  }
}
```

## Usage

```bash
python scripts/start_visualizer.py
# Open http://localhost:8080
```

## Implementation Notes

### Dynamic HTML Generation
If `index.html` doesn't exist, server generates it with embedded JavaScript:
- Canvas rendering for performance
- Client-side filtering and zoom/pan
- Tooltip positioning with screen bounds

### Thread-safe Database Access
Uses absolute paths for SQLite to avoid threading issues:
```python
self.db_path = os.path.abspath(config.get("database.sqlite_path"))
```

### Connection Lines
Draws thin lines between nearby points in the same cluster:
- Computed in screen space for efficiency
- k=2 nearest neighbors per point
- Toggle via ⟟ button
