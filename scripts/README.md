# Scripts

Entry-point scripts for running pipeline stages independently.

## Pipeline Stages

### 1. Batch Processing
```bash
python scripts/batch_process.py --dump dataset/sample-test-50.tar --limit 100
```

Extracts and embeds documents from SLOP dump archives.

**Arguments:**
- `--dump`: Path to tar archive (default: from config)
- `--limit`: Max URLs to process (default: unlimited)

**Output:**
- SQLite: `data/atlas.db`
- Qdrant: `atlas_embeddings` collection
- Archive: `data/parsed_archive/YYYY/MM/DD/batch_*.jsonl.zst`

### 2. Compute Mapping
```bash
python scripts/compute_mapping.py
```

Generates 2D UMAP coordinates and HDBSCAN clusters.

**Input:** Embeddings from Qdrant
**Output:** `data/mappings/latest.json`

### 3. Start Visualizer
```bash
python scripts/start_visualizer.py
```

Launches web UI at http://localhost:8080

### 4. Health Check
```bash
python scripts/run_health_check.py
```

Validates system state and data quality.

**Checks:**
- Database connectivity
- Embedding count vs document count
- Quality score distribution
- Cluster balance (Gini coefficient)

## Configuration

All scripts read from `config.json`:

```json
{
  "batch_processing": { ... },
  "mapping": { ... },
  "visualizer": { ... },
  "monitor": { ... }
}
```

## Workflow

```
batch_process.py → compute_mapping.py → start_visualizer.py
      ↓                   ↓                    ↓
   SQLite +           mappings/            Web UI
   Qdrant             latest.json        :8080
```

## Logs

Each script writes structured JSON logs to `logs/`:
- `batch_process.jsonl`
- `compute_mapping.jsonl`
- `visualizer.jsonl`
- `health_check.jsonl`
