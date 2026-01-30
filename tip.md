# Development Tips & Notes

## 2025-01-27: Extract-First Storage Design Decision

**Change:** Replaced raw HTML archival (Lake Pattern) with extract-first storage using trafilatura/readability.

**Rationale:**
- Raw HTML archival uses ~5x more storage (20KB/doc vs 2KB/doc for extracted content)
- Replay capability rarely needed—extractor bugs affect <1% of documents
- Visual layout (CSS, positioning) irrelevant for text-embedding use case

**Mitigations for lost capabilities:**
1. **Exact HTML replay:** Store `raw_html_hash` (SHA256) for provenance; re-crawl the small % where re-extraction matters
2. **Future extractor experiments:** Fetch fresh HTML from Wayback Machine or re-crawl a sample
3. **Visual layout:** N/A—not needed for text embeddings

**Storage savings:** 350 GB vs 750 GB for 50M documents (quality-gated HTML)

## 2025-01-28: Cartographic Techniques Implementation (v0.8.0)

**Change:** Integrated cartographic quality assurance and monitoring modules.

**Updates:**
1. **Database Schema:** Added `coverage_metrics`, `cluster_stats`, `detected_gaps`, and `wilson_score` field.
2. **Calibration:** Implemented **Topic-Cluster Cross-Validation** (split by domain) to prevent overfitting, and **QADI metrics** for nuanced validation.
3. **Scoring:** Added **Wilson Score** lower bound to rank "hidden gems" (high confidence quality) effectively.
4. **Monitoring:** Created `monitor` module to track **Coverage Gini coefficients** (topic/domain equity) and cluster health.
5. **Exploration:** Scaffolded `GapDetector` with Lonely Node heuristic and HDBSCAN clustering placeholders.

**Key Insight:** Standard random CV overfits by 28-40% due to domain autocorrelation. Splitting by domain during calibration is critical for realistic performance estimates.

## 2026-01-29: Phase 1 Offline Pipeline Implementation

**Change:** Completed Phase 1 batch processing pipeline for processing dataset dumps.

**New Modules:**
1. `dump_adapters.py` - Parsers for SLOP (Marginalia), JSONL, plain text formats
2. `fps_sampler.py` - Farthest Point Sampling with Lazy Greedy optimization
3. `deduplication.py` - SimHash, MinHash, Bloom filter for URL/content deduplication

**Enhanced Modules:**
- `producer.py` - Integrated dump adapters, URL filtering, Bloom filter deduplication
- `worker.py` - Added language detection, batch embedding, content filtering
- `writer.py` - FPS integration, calibrated scoring, JSONL archival
- `pipeline.py` - Progress monitoring, graceful shutdown, comprehensive statistics

**Bug Fixes:**
- SLOP adapter: Use streaming zstd decompression for frames without content size header
- Writer: Handle None values in metadata for content type detection

**Usage:**
```bash
python -m phase1_offline.pipeline --dump dataset/sample-m.tar --workers 8
python -m phase1_offline.pipeline --dump dataset/sample-m.tar --workers 8 --limit 10000
```

**Dependencies Required:**
- trafilatura (content extraction)
- langdetect (language filtering)
- zstandard (SLOP format decompression)
- sentence-transformers (embeddings)
- qdrant-client (vector storage, optional)

## 2026-01-29: Mapper, Storage, and Visualizer Implementation

**Change:** Completed the mapper, storage, and visualizer modules to enable the minimal feasible pipeline.

**New Modules:**

1. **`mapping/mapper.py`** - Atlas coordinate computation
   - UMAP projection (cosine metric) for 2D semantic mapping
   - HDBSCAN clustering for topic grouping
   - ImportanceScorer for Z-axis (domain authority)
   - Parquet and JSON export for visualization

2. **`storage/parquet_store.py`** - Columnar storage for exports
   - Efficient Parquet-based storage for mappings
   - Manifest tracking for versioned exports
   - Document export capabilities

3. **`storage/atlas_store.py`** - Unified storage interface
   - Abstraction over SQLite, Qdrant, and Parquet
   - Document queries with filtering/pagination
   - Cluster statistics and similarity search
   - Export utilities for visualization

4. **`visualizer/server.py`** - Web UI and API server
   - REST API for documents, clusters, mappings
   - Interactive 2D map visualization with UMAP coordinates
   - Document list view with quality filtering
   - Cluster exploration and search

5. **`run_pipeline.py`** - Orchestration script
   - Runs complete pipeline: batch → mapping → visualizer
   - Dependency checking and Qdrant availability
   - Multiple modes: full, map-only, visualize-only

**Usage (Separate Scripts - Each phase independently invokable):**
```bash
# Phase 1: Batch Processing
python scripts/batch_process.py                              # Uses config.json defaults
python scripts/batch_process.py --dump dataset/sample-test-50.tar
python scripts/batch_process.py --limit 100 --workers 8

# Phase 2: Compute Mapping
python scripts/compute_mapping.py                            # Uses config.json defaults
python scripts/compute_mapping.py --output data/mappings/custom.json
python scripts/compute_mapping.py --format parquet

# Phase 3: Start Visualizer
python scripts/start_visualizer.py                           # Uses config.json defaults
python scripts/start_visualizer.py --port 8000

# Health Check
python scripts/run_health_check.py
```

**Input/Output Contract:**
- Phase 1 Input: Dump file (config: batch_processing.default_dump)
- Phase 1 Output: SQLite docs + Qdrant embeddings + parsed archive
- Phase 2 Input: Qdrant embeddings (config: mapping.input_source)
- Phase 2 Output: Mapping file (config: mapping.output_path)
- Phase 3 Input: SQLite + Mapping files
- Phase 3 Output: Web server at config: visualizer.host:port

**Config Updates (`config.json`):**
- Added `mapping.hdbscan.min_cluster_size` for small test datasets
- Added `visualizer.host` and `visualizer.port`
- Added `monitor` thresholds for Gini alerts

**Dependencies Added:**
- umap-learn (UMAP projection)
- hdbscan (topic clustering)
- pyarrow (Parquet export)
- langdetect (language filtering - optional but recommended)

**Concurrency Architecture:**
The batch processing uses a producer-consumer pattern with concurrent embedding:

```
Producer (1 thread)     → url_queue (5000) →
Workers (8 threads)     → pre_embed_queue (1000) →
ConcurrentEmbedder (1)  → embed_queue (2000) →
Writer (1 thread)       → SQLite + Qdrant
```

- Workers do I/O-bound work (fetch, extract, filter) in parallel
- ConcurrentEmbedder batches items from all workers for GPU efficiency
- Expected throughput: 5-10 docs/sec (depends on content fetching)

**Throughput Analysis:**
- 2 docs/sec is typical when fetching live URLs (network-bound)
- With pre-fetched HTML (SLOP format), expect 10-20 docs/sec
- Bottlenecks: network I/O > embedding > extraction > filtering

**URL Count Clarification:**
The sample-test-50.tar contains 50 SLOP archives. Each archive is a crawl dump
with many URLs (not just 1). Total URLs ≈ 1000+ is expected behavior.

## 2026-01-29: Bug Fixes During Pipeline Testing

**Bugs Fixed:**

1. **ConcurrentEmbedder shutdown race condition** (`worker.py`)
   - Issue: Embedder stopped before processing queued items
   - Fix: Added `_draining` flag and `drain_and_stop()` method to ensure all items
     are processed before shutdown

2. **Pipeline shutdown sequence** (`pipeline.py`)
   - Issue: Writer stopped before embed_queue was drained
   - Fix: Added explicit drain wait after WorkerPool.stop() before Writer.stop()

3. **JSON serialization of numpy types** (`mapper.py`)
   - Issue: `int64` from HDBSCAN not JSON serializable
   - Fix: Added `NumpyEncoder` class for JSON export and explicit type conversion
     in `MappingResult.to_dict()`

4. **Writer thread daemon termination** (`pipeline.py`)
   - Issue: Writer thread set as daemon=True, causing buffer not to flush
   - Fix: Changed to daemon=False and added proper thread join on shutdown

5. **Database relative path in threaded context** (`database.py`)
   - Issue: Relative path `./data/atlas.db` failed in HTTP handler threads
   - Fix: Convert to absolute path in Database.__init__()

**Test Results:**
```
Phase 1 (Batch Processing): 50 URLs → 27 documents accepted (79.4%)
Phase 2 (Mapping): 473 documents mapped, 8 clusters, 7.1s
Phase 3 (Visualizer): Web UI at http://localhost:8080
Health Check: No alerts
```

## Known Data Consistency Issue

**Problem:** Map shows 475 points but "In Database" shows only 18.

**Root cause:** Qdrant embeddings persist across runs while SQLite was reset during testing.
- Qdrant: 475 embeddings from multiple test runs
- SQLite: 18 documents from the most recent run after DB reset

**Impact:**
- Most map points show "Untitled" (no title in mappings JSON)
- Tooltips lack rich metadata for older embeddings

**Fix for production:**
When resetting for a fresh run, clear BOTH:
```bash
rm -f data/atlas.db                          # SQLite
curl -X DELETE http://localhost:6333/collections/atlas_embeddings  # Qdrant
```

## Visualizer Features (v2)

- Statistics panel: Map Points, Clusters, Avg Quality, In Database
- Cluster sidebar: Click to filter map AND list view
- Filter controls: Content type and quality filter both map and list
- Connection lines: Toggle intra-cluster connections (⟟ button)
- Pan/zoom: Scroll wheel zoom, drag to pan, +/−/reset buttons
- Tooltips: Title, excerpt, domain, quality, cluster, importance
- Click-to-open: Clicking a dot opens the website
