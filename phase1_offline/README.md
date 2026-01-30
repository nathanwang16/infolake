# Phase 1: Offline Batch Processing

Producer-consumer pipeline for extracting, embedding, and storing web documents.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Producer   │ ──▶ │  WorkerPool │ ──▶ │  Embedder   │ ──▶ │   Writer    │
│ (URL Queue) │     │ (Extract)   │     │ (Concurrent)│     │ (Batch DB)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Modules

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestrator, manages queues and threads |
| `producer.py` | Reads SLOP archives, yields URLs to queue |
| `worker.py` | Extracts content, validates quality, batches embedding |
| `writer.py` | Writes to SQLite and Qdrant in batches |
| `dump_adapters.py` | Adapters for different dump formats (SLOP, WET, etc.) |
| `deduplication.py` | SimHash/MinHash near-duplicate detection |
| `fps_sampler.py` | Quality-weighted farthest-point sampling |

## Key Features

### Concurrent Embedding
`ConcurrentEmbedder` runs embedding in a separate thread with batch accumulation:
- Collects items until `batch_size` or `timeout`
- MPS/CUDA acceleration when available
- Graceful drain on shutdown

### Content Extraction
Uses `trafilatura` (primary) with `readability` fallback:
- Extracts main content, title, author, date
- Language detection via `langdetect`
- Length validation (min/max configurable)

### Quality Scoring
Multi-dimensional scoring from `curation/scoring.py`:
- Content depth, authority signals, freshness
- Wilson confidence interval for ranking

## Configuration (`config.json`)

```json
{
  "batch_processing": {
    "default_dump": "dataset/sample-test-50.tar",
    "workers": 4,
    "worker_batch_size": 10,
    "url_queue_size": 1000,
    "embed_queue_size": 500,
    "limit": null
  },
  "embedding": {
    "model": "BAAI/bge-small-en-v1.5",
    "batch_size": 32
  }
}
```

## Usage

```bash
python scripts/batch_process.py --dump dataset/sample-test-50.tar --limit 100
```

## Output

- SQLite: `data/atlas.db` (documents table)
- Qdrant: `atlas_embeddings` collection
- Parsed archive: `data/parsed_archive/YYYY/MM/DD/batch_*.jsonl.zst`

## Throughput

Expected: 10-20 docs/sec (network/extraction bound)
- Embedding: ~100 docs/sec (batched on MPS)
- Writing: ~500 docs/sec (batched inserts)
