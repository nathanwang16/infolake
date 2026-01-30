# Source Utilities

Core utilities for logging and web crawling.

## Logging (`src/logging/logger.py`)

Structured JSON logging with automatic file rotation:

```python
from src.logging.logger import get_logger

logger = get_logger("pipeline")
logger.info("Processing started", extra={"count": 100})
```

**Output format (JSONL):**
```json
{"timestamp": "2026-01-29T16:30:00", "level": "INFO", "logger": "pipeline", "message": "Processing started", "count": 100}
```

**Features:**
- Per-module log files in `logs/`
- Console output (INFO+) and file output (DEBUG+)
- Automatic exception formatting
- Context fields via `extra` dict

## Crawling (`src/crawling/fetcher.py`)

HTTP fetcher with retry and rate limiting:

```python
from src.crawling.fetcher import Fetcher

fetcher = Fetcher()
response = fetcher.fetch("https://example.com")
```

**Features:**
- Exponential backoff on failures
- robots.txt compliance (configurable)
- Connection pooling
- Timeout handling

## Directory Structure

```
src/
├── __init__.py
├── logging/
│   ├── __init__.py
│   └── logger.py      # Structured JSON logger
└── crawling/
    ├── __init__.py
    └── fetcher.py     # HTTP fetcher
```

## Configuration

Logging settings are implicit (logs to `logs/{module}.jsonl`).

Crawling settings in `config.json`:
```json
{
  "crawling": {
    "timeout": 30,
    "max_retries": 3,
    "respect_robots": true
  }
}
```
