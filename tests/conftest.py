"""
Shared pytest fixtures for Truth Atlas tests.

Uses DI to inject temp-file SQLite databases and mock Qdrant managers,
proving the refactored components work without touching production state.

The conftest patches the Config singleton at import time so that the
module-level `db = Database()` in common/database.py doesn't fail
when config.json points to an unreachable path.
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path so imports work from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch Config BEFORE anything else imports common.database, so the
# module-level `db = Database()` singleton doesn't crash.
from common.config import Config

_test_config = Config.__new__(Config)
_test_config._config = {
    "database": {"sqlite_path": os.path.join(tempfile.gettempdir(), "atlas_test_singleton.db")},
    "paths": {},
}
Config._instance = _test_config

# Now it's safe to import database and repos
import pytest
from common.database import Database
from common.repositories import (
    DocumentRepository,
    MetricsRepository,
    GoldenSetRepository,
    JobRepository,
)


@pytest.fixture
def memory_db(tmp_path):
    """Provides a fresh file-backed SQLite database with full schema.

    Uses tmp_path so each test gets an isolated database (unlike :memory:
    which creates a new DB per connection and loses schema).
    """
    db_file = str(tmp_path / "test_atlas.db")
    return Database(db_path=db_file)


@pytest.fixture
def doc_repo(memory_db):
    return DocumentRepository(memory_db)


@pytest.fixture
def metrics_repo(memory_db):
    return MetricsRepository(memory_db)


@pytest.fixture
def golden_set_repo(memory_db):
    return GoldenSetRepository(memory_db)


@pytest.fixture
def job_repo(memory_db):
    return JobRepository(memory_db)
