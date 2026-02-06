import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from common.config import config
from common.logging.logger import get_logger

logger = get_logger("database")

class Database:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Convert to absolute path to avoid issues with relative paths in threads
            raw_path = config.get("database.sqlite_path")
            self.db_path = os.path.abspath(raw_path)
        elif db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = os.path.abspath(db_path)
        self._init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    @contextmanager
    def connection(self):
        """Context manager that provides a connection with automatic commit/rollback."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize the database schema."""
        schema = """
        -- Documents table
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            canonical_url TEXT UNIQUE NOT NULL,
            title TEXT,
            summary TEXT,
            content_hash TEXT NOT NULL,
            minhash_signature BLOB,
            content_length INTEGER,
            language TEXT,
            author TEXT,
            publication_date DATE,
            domain TEXT NOT NULL,
            source_phase TEXT,
            source_dump TEXT,
            source_gap_id TEXT,
            source_gap_type TEXT,
            source_query TEXT,
            detected_content_type TEXT,
            quality_score REAL,
            quality_score_version INTEGER DEFAULT 1,
            quality_components JSON,
            quality_profile_used TEXT,
            wilson_score REAL, -- New in v0.8.0
            importance_score REAL,
            cluster_id INTEGER, -- New in v0.8.0
            epistemic_claim_type TEXT,
            epistemic_stance TEXT,
            epistemic_quality REAL,
            epistemic_components JSON,
            farthest_point_rank INTEGER,
            novelty_distance REAL,
            status TEXT,
            merged_into TEXT,
            parsed_archive_path TEXT,
            raw_html_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_verified TIMESTAMP,
            FOREIGN KEY(merged_into) REFERENCES documents(id)
        );

        -- Golden Set table for calibration
        CREATE TABLE IF NOT EXISTS golden_set (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            label TEXT, -- 'exemplary' or 'garbage'
            content_type TEXT,
            domain TEXT, -- New in v0.8.0 for Topic-Cluster CV
            raw_metrics JSON,
            notes TEXT,
            added_by TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            version INTEGER
        );
        
        -- URL Queue for processing
        CREATE TABLE IF NOT EXISTS url_queue (
            canonical_url TEXT PRIMARY KEY,
            original_url TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            source_phase TEXT,
            source TEXT NOT NULL,
            source_metadata JSON,
            priority REAL DEFAULT 0.5,
            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            claimed_by TEXT,
            claimed_at TIMESTAMP,
            completed_at TIMESTAMP,
            retry_count INTEGER DEFAULT 0,
            error_message TEXT,
            parsed_archive_path TEXT
        );

        -- Processing Jobs
        CREATE TABLE IF NOT EXISTS dump_processing_jobs (
            id TEXT PRIMARY KEY,
            dump_name TEXT NOT NULL,
            dump_path TEXT,
            total_urls INTEGER,
            filtered_urls INTEGER,
            extracted_count INTEGER,
            embedded_count INTEGER,
            selected_count INTEGER,
            accepted_count INTEGER,
            rejected_count INTEGER,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT,
            error_message TEXT
        );

        -- Coverage Metrics (New in v0.8.0)
        CREATE TABLE IF NOT EXISTS coverage_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            topic_gini REAL,
            domain_gini REAL,
            orphan_rate REAL,
            high_quality_orphans INTEGER,
            cluster_count INTEGER,
            largest_cluster_pct REAL
        );

        -- Cluster Stats (New in v0.8.0)
        CREATE TABLE IF NOT EXISTS cluster_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER NOT NULL,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            doc_count INTEGER,
            avg_quality REAL,
            quality_std REAL,
            is_content_farm BOOLEAN,
            is_authority BOOLEAN,
            action_taken TEXT
        );

        -- Detected Gaps (New in v0.8.0)
        CREATE TABLE IF NOT EXISTS detected_gaps (
            id TEXT PRIMARY KEY,
            gap_type TEXT, -- topic, viewpoint
            detection_method TEXT,
            centroid BLOB,
            gap_size REAL,
            anchor_doc_id TEXT,
            neighbor_doc_ids JSON,
            dominant_stance TEXT,
            diversity_score REAL,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            queries_generated JSON,
            documents_added INTEGER DEFAULT 0,
            gap_reduction REAL,
            status TEXT
        );

        -- Exploration Provenance (New in v0.8.0)
        CREATE TABLE IF NOT EXISTS exploration_provenance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT,
            gap_id TEXT,
            query_text TEXT NOT NULL,
            search_api TEXT,
            search_rank INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES documents(id),
            FOREIGN KEY(gap_id) REFERENCES detected_gaps(id)
        );

        -- Document full text storage for deferred scoring
        CREATE TABLE IF NOT EXISTS document_texts (
            doc_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        );
        """
        
        try:
            with self.get_connection() as conn:
                conn.executescript(schema)
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

db = Database()
