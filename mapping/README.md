# Mapping Module

Semantic mapping of documents to 2D/3D coordinates with topic clustering. Refactored with protocol-based, modular architecture.

## Architecture

```
mapping/
├── __init__.py           # Public API + backward compatibility
├── pipeline.py           # MappingPipeline orchestrator
├── protocols.py          # Type protocols (Projector, Clusterer, AxisScorer)
├── registry.py           # ComponentRegistry for extensibility
├── mapper.py             # AtlasMapper (backward compat facade)
├── projectors.py         # UMAPProjector
├── clusterers.py         # HDBSCANClusterer
└── axis_scorers.py       # DomainAuthorityAxisScorer
```

## Overview

Transforms high-dimensional embeddings (384D) into visual coordinates:
- **X, Y**: UMAP 2D projection (semantic similarity)
- **Z**: Importance score (domain authority, quality)
- **Cluster**: HDBSCAN topic clustering

## Algorithm

```
Embeddings (384D) ─▶ UMAP ─▶ 2D Coordinates
                      │
                      ▼
                   HDBSCAN ─▶ Cluster Labels
                      │
                      ▼
              Importance Scoring ─▶ Z-axis
```

## Usage

### Backward Compatible API

```python
# Old API still works
from mapping.mapper import AtlasMapper

mapper = AtlasMapper()
result = mapper.compute_mapping(sample_size=50000)
result.export_to_json("data/mappings/latest.json")
```

### New Modular API

```python
from mapping import MappingPipeline
from common.qdrant_manager import QdrantManager

pipeline = MappingPipeline()

# Load embeddings from Qdrant
qm = QdrantManager()
embeddings = qm.get_all_vectors()

# Step 1: Project to 2D
coordinates = pipeline.project(embeddings, force_refit=False)

# Step 2: Cluster
cluster_labels = pipeline.cluster(coordinates)

# Step 3: Score importance (Z-axis)
for doc in documents:
    importance = pipeline.score_importance(
        domain=doc.domain,
        quality_score=doc.quality_score,
        content_type=doc.content_type,
    )
```

### Custom Components

Swap in custom implementations following the protocols:

```python
from mapping.protocols import Projector, Clusterer, AxisScorer
from mapping import MappingPipeline

class MyCustomProjector:
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        # Your projection logic
        return coordinates_2d

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        # Transform using fitted model
        return coordinates_2d

# Use custom component
pipeline = MappingPipeline(projector=MyCustomProjector())
```

## Configuration (`config.json`)

```json
{
  "mapping": {
    "umap": {
      "n_neighbors": 15,
      "min_dist": 0.1,
      "metric": "cosine",
      "n_components": 2
    },
    "hdbscan": {
      "min_cluster_size": 15,
      "min_samples": 5,
      "cluster_selection_method": "eom"
    },
    "sample_size": 50000,
    "output_path": "./data/mappings/latest.json"
  }
}
```

## Usage

```bash
python scripts/compute_mapping.py
```

## Output Format

```json
{
  "mappings": [
    {
      "doc_id": "abc123...",
      "x": -2.34,
      "y": 5.67,
      "z": 0.72,
      "cluster_id": 3,
      "quality_score": 0.65
    }
  ],
  "stats": {
    "total_documents": 475,
    "clusters": 10,
    "noise_points": 12
  }
}
```

## Implementation Notes

### NumPy JSON Serialization
Custom `NumpyEncoder` handles numpy types:
```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
```

### HDBSCAN Cluster Labels
- `-1` indicates noise/orphan points
- Cluster IDs are 0-indexed
- Small clusters may be merged by HDBSCAN
