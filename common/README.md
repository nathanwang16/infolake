# Common Module

Shared utilities, configuration management, and logging infrastructure.

## Components

### Logging (`logging/logger.py`)
Structured JSON logging with automatic file rotation:

```python
from common.logging.logger import get_logger, setup_logger

logger = get_logger("pipeline")
logger.info("Processing started", extra={"count": 100})
```

**Output format (JSONL):**
```json
{"timestamp": "2026-01-29T16:30:00", "level": "INFO", "name": "pipeline", "message": "Processing started", "count": 100}
```

**Features:**
- Per-module log files in `logs/`
- Console output (INFO+) and file output (DEBUG+)
- Automatic exception formatting
- Context fields via `extra` dict
- Rotating file handler (10MB max, 5 backups)

### Config (`config.py`)
Singleton configuration loader:

```python
from common.config import config

db_path = config.get("database.sqlite_path", "data/atlas.db")
workers = config.get("batch_processing.workers", 4)
```

**Features:**
- Dot-notation access to nested keys
- Default value fallback
- Single load from `config.json`

### Database (`database.py`)
SQLite connection manager with schema initialization:

```python
from common.database import Database, db

# Use module-level singleton
db = Database()
conn = db.get_connection()

# Or use context manager for automatic commit/rollback
with db.connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents LIMIT 10")
```

**Features:**
- Thread-safe connections (each thread gets its own)
- Absolute path resolution (avoids CWD issues)
- Schema auto-creation with `CREATE TABLE IF NOT EXISTS`
- Context manager for transaction handling

### Repositories (`repositories.py`)
Data access layer following the Repository pattern:

```python
from common.repositories import (
    DocumentRepository,
    DocumentTextRepository,
    MetricsRepository,
    GoldenSetRepository,
    JobRepository,
)

# Document operations
doc_repo = DocumentRepository()
doc = doc_repo.get_by_id("abc123")
docs = doc_repo.get_list(limit=100, min_quality=0.5)
count = doc_repo.get_count(content_type="scientific")

# Text storage for deferred scoring
text_repo = DocumentTextRepository()
text_repo.insert(doc_id="abc123", text="Full document text...")
text = text_repo.get_text("abc123")
unscored_batch = text_repo.get_unscored_batch(batch_size=1000)

# Metrics for monitoring
metrics_repo = MetricsRepository()
topic_dist = metrics_repo.get_topic_distribution()
orphan_count = metrics_repo.get_high_quality_orphan_count()
```

**Benefits:**
- Clean separation of data access logic
- Type-safe with domain models
- Thread-safe (each method opens its own connection)
- Testable (can inject mock database)
- Consistent error handling

### Qdrant Manager (`qdrant_manager.py`)
Unified interface for Qdrant vector database:

```python
from common.qdrant_manager import QdrantManager

# Initialize (creates collection if missing)
qm = QdrantManager(create_if_missing=True, timeout=5)

# Access client
client = qm.client
collection_name = qm.collection_name
is_available = qm.available

# Insert vectors
qm.upsert_points(points=[...])

# Query vectors
results = qm.search(query_vector=embedding, limit=10)
```

**Features:**
- Automatic collection creation with quantization
- Configuration from `config.json`
- Backward-compatible properties
- Health checking with timeout
- Scalar quantization + on-disk storage support

## Domain Models (`models.py`)
Dataclasses for type-safe data handling:

```python
from common.models import (
    Document,              # Full document with summary
    DocumentListItem,      # Slim document for lists
    DocumentCreate,        # Write-only model for inserts
    ClusterInfo,          # Cluster statistics
    CoverageMetrics,      # Atlas health metrics
    SearchResult,         # Document + similarity score
    GoldenSetEntry,       # Calibration entry
)

# Example: Load from database row
row = cursor.fetchone()
doc = Document.from_row(row)

# Convert to dict for JSON serialization
doc_dict = doc.to_dict()
```

**Benefits:**
- Type hints for IDE autocomplete
- Consistent field naming
- Easy serialization to JSON
- Backward-compatible with raw SQL

## Configuration Schema

See `config.json` for full schema. Key sections:

| Section | Purpose |
|---------|---------|
| `paths` | Data directories, archive paths |
| `database` | SQLite path |
| `qdrant` | Qdrant URL, collection, quantization |
| `batch_processing` | Concurrency, queue sizes, thresholds |
| `embedding` | Model name, device, batch size |
| `mapping` | UMAP/HDBSCAN parameters |
| `visualizer` | Host, port, static dir |
| `monitor` | Alert thresholds |
| `calibration` | Golden set configuration |

## Thread Safety

### Repository Pattern
Each repository method opens its own connection, ensuring thread safety:

```python
def get_by_id(self, doc_id: str) -> Optional[Document]:
    conn = self._db.get_connection()  # New connection per call
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT ... WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        return Document.from_row(row) if row else None
    finally:
        conn.close()  # Always close
```

### Transaction Handling
Use the `conn` parameter for multi-operation transactions:

```python
with db.connection() as conn:
    doc_repo.insert(doc, conn=conn)
    text_repo.insert(doc.id, text, conn=conn)
    # Both operations in same transaction
    # Auto-commit on exit, auto-rollback on exception
```

### Absolute Paths
```python
# Prevents issues when threads have different CWD
self.db_path = os.path.abspath(raw_path)
```

## Migration from Direct SQL

The repository pattern is backward compatible. Migrate gradually:

```python
# Old way (still works)
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
row = cursor.fetchone()

# New way (recommended)
doc_repo = DocumentRepository()
doc = doc_repo.get_by_id(doc_id)
```
