# Mapping Module

Semantic mapping of documents to 2D/3D coordinates with topic clustering.

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

## Key Classes

### `SemanticMapper`
Main entry point for computing mappings:
```python
mapper = SemanticMapper(store)
result = mapper.compute_mapping(sample_size=50000)
result.export_to_json("data/mappings/latest.json")
```

### `MappingResult`
Container for mapping data with export methods:
- `to_dict()`: Convert to JSON-serializable dict
- `export_to_json()`: Write to file with NumpyEncoder
- Statistics: cluster distribution, quality metrics

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
