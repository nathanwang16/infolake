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
SQLite connection manager:

```python
from common.database import Database

db = Database()
docs = db.get_documents(limit=100)
db.insert_documents(documents)
```

**Features:**
- Thread-local connections
- Absolute path resolution (avoids CWD issues)
- Schema auto-creation
- Batch insert support

## Configuration Schema

See `config.json` for full schema. Key sections:

| Section | Purpose |
|---------|---------|
| `database` | SQLite and Qdrant paths |
| `batch_processing` | Worker count, queue sizes |
| `embedding` | Model name, batch size |
| `mapping` | UMAP/HDBSCAN parameters |
| `visualizer` | Host, port, static dir |
| `monitor` | Alert thresholds |

## Thread Safety

### SQLite Threading
```python
# Uses thread-local storage for connections
_thread_local = threading.local()

def get_connection(self):
    if not hasattr(_thread_local, 'connection'):
        _thread_local.connection = sqlite3.connect(self.db_path)
    return _thread_local.connection
```

### Absolute Paths
```python
# Prevents issues when threads have different CWD
self.db_path = os.path.abspath(raw_path)
```
