# Common Module

Shared utilities and configuration management.

## Components

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
