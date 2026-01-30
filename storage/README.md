# Storage Module

Persistent storage layer for documents, embeddings, and mappings.

## Components

### AtlasStore (`atlas_store.py`)
Unified interface combining SQLite and Qdrant:

```python
store = AtlasStore()
store.add_documents(documents)  # SQLite
store.add_embeddings(doc_ids, embeddings)  # Qdrant
results = store.search("query text", limit=10)  # Semantic search
```

### ParquetStore (`parquet_store.py`)
Columnar storage for mappings and exports:

```python
pstore = ParquetStore()
pstore.save_mappings(mapping_result)
mappings = pstore.load_mappings()
```

## SQLite Schema

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    domain TEXT,
    title TEXT,
    content TEXT,
    content_type TEXT,
    quality_score REAL,
    cluster_id INTEGER,
    importance_score REAL,
    word_count INTEGER,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX idx_documents_domain ON documents(domain);
CREATE INDEX idx_documents_quality ON documents(quality_score);
CREATE INDEX idx_documents_cluster ON documents(cluster_id);
```

## Qdrant Collection

```python
collection_config = {
    "name": "atlas_embeddings",
    "vectors": {
        "size": 384,  # bge-small-en-v1.5
        "distance": "Cosine"
    }
}
```

## Configuration (`config.json`)

```json
{
  "database": {
    "sqlite_path": "./data/atlas.db",
    "qdrant_url": "http://localhost:6333",
    "qdrant_collection": "atlas_embeddings"
  },
  "storage": {
    "exports_dir": "./data/exports",
    "mappings_dir": "./data/mappings",
    "parsed_archive_dir": "./data/parsed_archive"
  }
}
```

## Batch Operations

### Batch Insert (SQLite)
```python
# Uses executemany for efficiency
cursor.executemany(
    "INSERT OR REPLACE INTO documents ...",
    [(doc.id, doc.url, ...) for doc in batch]
)
```

### Batch Upsert (Qdrant)
```python
# Batched vector upsert
client.upsert(
    collection_name="atlas_embeddings",
    points=[PointStruct(id=id, vector=vec) for id, vec in batch]
)
```

## Thread Safety

- SQLite: Thread-local connections via `threading.local()`
- Qdrant: Thread-safe client with connection pooling
- Paths: Converted to absolute paths to avoid CWD issues

## Error Handling

- Graceful fallback if Qdrant unavailable
- Connection retry with exponential backoff
- Transaction rollback on failure
