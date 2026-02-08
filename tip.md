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

## 2026-01-30: Content Preview Feature

**Change:** Added content preview (excerpt) display in visualizer tooltips.

**Problem:** Tooltips only showed title, URL, and metrics (quality, cluster, importance) - no content preview.

**Root Cause:** The `summary` field in documents table wasn't being populated during batch processing.

**Solution (Runtime + Backfill):**

1. **Extract-time caching:** Modified `writer.py` to extract and store first 100 words / first paragraph as `summary` during batch processing using new `common/text_utils.py` utility.

2. **On-demand fetching:** Added `/api/preview/:id` endpoint in `visualizer/server.py` that:
   - Returns cached summary if available
   - Otherwise fetches URL, extracts content, saves to DB (backfill), returns excerpt
   - Uses UPDATE (not INSERT) to avoid conflicts with running pipeline

3. **Frontend async loading:** Updated `index.html` to:
   - Check cache → mapping data → trigger async fetch
   - Show "Loading preview..." while fetching
   - Cache results in `previewCache` Map to avoid re-fetching

**Key Design Decisions:**
- **No duplicate scraping:** Any runtime fetch is backfilled to `summary` column
- **Database safety:** Uses UPDATE with WHERE clause to avoid overwriting existing data
- **Non-blocking:** Async fetch doesn't block tooltip display
- **Graceful degradation:** Shows "Preview not available" if fetch fails

**Files Modified:**
- `common/text_utils.py` (new) - Excerpt extraction utility
- `phase1_offline/writer.py` - Save excerpts during batch processing
- `visualizer/server.py` - `/api/preview/:id` endpoint
- `storage/atlas_store.py` - Include summary in document queries
- `visualizer/static/index.html` - Async preview loading with caching

## 2026-01-30: LLM-Based Summarization

**Change:** Replaced algorithmic excerpt extraction with LLM/ML-based summarization for content previews.

**Problem:** First paragraph extraction wasn't always concise or informative.

**Solution:** Created `common/summarizer.py` with multiple backends:

1. **Local Model (DistilBART)** - `sshleifer/distilbart-cnn-6-6`
   - 2x faster than full BART
   - Good summarization quality
   - No API costs

2. **OpenAI API** - `gpt-4o-mini`
   - Very fast (~500ms)
   - High quality summaries
   - Requires API key (set `OPENAI_API_KEY` env var)

3. **Ollama** - `llama3.2:1b` (or any local model)
   - If Ollama is running locally
   - No external dependencies

4. **Fallback** - Algorithmic excerpt extraction
   - If no ML backend available

**Backend Selection (auto mode):**
1. Try local transformers model
2. Try Ollama if available
3. Try OpenAI if API key set
4. Fall back to excerpt extraction

**Config Options (`config.json`):**
```json
"summarizer": {
    "backend": "auto",  // auto, local, openai, ollama, excerpt
    "local_model": "sshleifer/distilbart-cnn-6-6",
    "ollama_model": "llama3.2:1b",
    "max_length": 80
}
```

**Performance:**
- Local DistilBART: ~200-500ms per summary (CPU), ~50ms (MPS/CUDA)
- OpenAI gpt-4o-mini: ~300-500ms (network dependent)
- Ollama: ~200-1000ms (depends on model)
- Excerpt fallback: ~1ms

**Dependencies:**
- `transformers` (for local summarization)
- `openai` (for OpenAI API)
- `torch` (for model inference)

## 2026-01-30: Performance Optimization for Previews

**Issue:** Web fetching was the bottleneck, and some outputs contained junk (cookie policies, table formatting).

**Fixes:**
1. **Faster fetching:** Reduced timeout to (3s connect, 5s read)
2. **Better extraction:** Using trafilatura with `favor_precision=True`
3. **Junk filtering:** Pre-filter removes:
   - Table formatting (|, pipes, separators)
   - Cookie/privacy notices
   - Bot detection messages
   - Navigation elements
   - Wikipedia infobox junk
4. **Post-processing:** Converts full sentences to concise phrase-style (semicolon-separated)
5. **Model switch:** Changed from DistilBART to T5-small (faster, better compatibility)

**Performance after warmup:**
- Fetch: ~100-200ms
- Extract: ~10-30ms (simple pages), ~200ms (complex like Wikipedia)
- Summarize: ~300-600ms (after warmup)
- **Total: ~500-1000ms** for most pages

**Phrase-style output example:**
Before: "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms."
After: "machine learning is a field of study in artificial intelligence; concerned with development of statistical algorithms"

## 2026-01-30: Visualizer Performance & Cluster Selection

**Issue 1:** Clicking a cluster would re-center/re-zoom to show only that cluster's points.
**Fix:** Pre-compute global coordinate bounds once when data loads. Use these bounds for all rendering, so filtered clusters stay in their original positions. Other clusters simply become hidden.

**Issue 2:** Zooming was laggy, especially with many points.
**Fixes:**
1. **requestAnimationFrame throttling** - prevents excessive redraws (~60fps max)
2. **Viewport culling** - only draw points visible on screen
3. **Batched rendering** - group points by color, single draw call per color
4. **Optimized connections** - sample large clusters, single stroke() call for all lines
5. **Smaller zoom steps** (0.95/1.05 vs 0.9/1.1) for smoother feel

**Result:** Smooth panning/zooming even with thousands of points.

## 2026-01-30: Zoom Smoothness & View State Preservation (v2)

**Issue 1:** Zooming was still laggy due to full rendering on every frame.
**Fix:** Two-tier rendering:
- During interaction (pan/zoom): simplified rendering with fixed-size dots, no connections
- After interaction ends (150ms delay): full quality render with connections and variable dot sizes

**Issue 2:** After zooming/panning, clicking a cluster reset the view position.
**Fix:** The code was already correct (using global scale/offsetX/offsetY), but the connection line algorithm was causing visual artifacts. Simplified the connection algorithm to connect sequential points instead of k-nearest.

**Changes:**
1. `startInteraction()` / `endInteraction()` pattern for simplified rendering during motion
2. Fixed-radius dots during interaction (no per-point size calculation)
3. Connections only drawn when idle (not during pan/zoom)
4. Sequential connection algorithm (sort by position, connect adjacent) instead of k-nearest
5. Zoom info display shows current zoom percentage

## 2026-01-30: Semantic Cluster Mapping

**Issue:** UMAP+HDBSCAN produces numeric cluster IDs (0, 1, 2, -1) with no semantic meaning. Users can't understand what each cluster represents without inspecting individual documents.

**Solution:** Created `mapping/semantic_mapper.py` - a parallel algorithm that generates semantically meaningful, hierarchical clusters.

**New Features:**

1. **Semantic Labels:** Each cluster gets a human-readable label derived from TF-IDF keyword extraction (e.g., "Machine Learning / Neural Networks" instead of "Cluster 3")

2. **Hierarchical Taxonomy:** Clusters are organized into parent-child relationships via agglomerative clustering on cluster centroids. Creates 3-7 top-level groups containing related sub-clusters.

3. **Inter-Cluster Relationships:** Computes similarity/distance between all cluster pairs, categorizes as:
   - `similar` (distance < 0.3)
   - `related` (distance < 0.6)
   - `sibling` (same parent cluster)

4. **Document Hierarchy Path:** Each document gets a breadcrumb trail like ["Technology", "Machine Learning", "NLP"]

5. **Optional LLM Enhancement:** Top clusters can get LLM-generated descriptions using the summarizer module

**Architecture:**

```
mapping/semantic_mapper.py
├── SemanticCluster       # Dataclass with label, keywords, hierarchy
├── ClusterRelationship   # Source, target, type, similarity
├── SemanticMappingResult # Complete result with all data
└── SemanticMapper        # Main algorithm class
    ├── _compute_base_clusters()     # HDBSCAN (reuses existing labels)
    ├── _generate_cluster_labels()   # TF-IDF keyword extraction
    ├── _build_hierarchy()           # Agglomerative clustering on centroids
    ├── _compute_relationships()     # Pairwise cosine distances
    └── _enhance_with_llm()          # Optional LLM descriptions
```

**Integration Points:**

1. **mapper.py:** Added `enable_semantic_mapping=True` param to `AtlasMapper.__init__()`. Semantic data is computed after HDBSCAN and merged into `MappingResult`.

2. **MappingResult:** Extended with `cluster_label`, `cluster_keywords`, `hierarchy_path`, `parent_cluster_id` fields.

3. **atlas_store.py:** Added `get_semantic_clusters()`, `get_cluster_hierarchy()`, `get_cluster_relationships()` methods.

4. **visualizer/server.py:** Added API endpoints:
   - `GET /api/semantic-clusters` - Full semantic cluster data
   - `GET /api/cluster-hierarchy` - Tree structure for sidebar
   - `GET /api/cluster-relationships` - Similarity relationships
   - Enhanced `GET /api/clusters` to include labels

5. **Frontend (index.html):** Completely replaced to:
   - Display hierarchical tree in sidebar with parent/child structure
   - Show cluster keywords and labels instead of "Cluster 0"
   - Display "Related Topics" panel when selecting a cluster
   - Show hierarchy path in tooltips
   - Color-code clusters consistently

**Data Flow:**

```
HDBSCAN labels → TF-IDF on cluster docs → Keyword extraction → Label generation
         ↓
Cluster centroids → Agglomerative clustering → Parent clusters → Hierarchy
         ↓
Centroid pairs → Cosine distance → Relationships (similar/related/sibling)
         ↓
Export to JSON → API endpoints → Frontend hierarchy tree
```

**Usage:**

```python
# Standalone usage
from mapping.semantic_mapper import SemanticMapper
mapper = SemanticMapper(min_cluster_size=5, use_llm=False)
result = mapper.run_semantic_mapping(doc_ids, embeddings, texts)
print(result.clusters[0].label)  # "Machine Learning / Neural Networks"

# Integrated with AtlasMapper (default behavior)
from mapping.mapper import AtlasMapper
mapper = AtlasMapper(enable_semantic_mapping=True)
mappings = mapper.run_full_mapping()
print(mappings[0].cluster_label)  # "Machine Learning / Neural Networks"
```

**Dependencies:**
- sklearn (TF-IDF, feature extraction)
- scipy (agglomerative clustering, linkage)
- hdbscan (base clustering - already required)

---

## deck.gl Frontend Rewrite (Feb 2026)

**Change:** Replaced Canvas-based rendering in `visualizer/static/index.html` with deck.gl
(ScatterplotLayer + OrthographicView + LineLayer). Added pydeck (0.9.1) server-side for
deck configuration and standalone HTML export.

**New API endpoints:**
- `GET /api/mapping-list` - Lists available mapping datasets for switching
- `GET /api/export-html` - Exports standalone deck.gl visualization via pydeck

**Key decisions:**
- deck.gl JS loaded from CDN (`unpkg.com/deck.gl@^9.0.0/dist.min.js`) handles all rendering
- OrthographicView (not MapView) since UMAP coordinates are non-geographic
- pydeck used server-side for HTML export and data enrichment, not for live rendering
- Server enriches parquet mapping data with document metadata before sending to frontend
- Mapping switcher dropdown supports multiple datasets via `/api/mapping-list`

**deck.gl gotcha:** `OrthographicView` zoom 0 = 1 unit per pixel. Compute initial zoom
from `Math.log2(viewportSize / dataRange)` to fit all points.

## 2026-02-06: Resume Safety + Placeholder Removal

**Bugs Fixed:**
1. **Mock fetch/embedding paths** (`phase1_offline/worker.py`)
   - Issue: Missing dependencies silently returned mock HTML/embeddings
   - Fix: Raise explicit errors, stop workers/embedder when dependencies are missing

2. **Resume dedup missing** (`phase1_offline/producer.py`, `common/repositories.py`)
   - Issue: Phase 1 restarts could reprocess already-ingested URLs
   - Fix: Load existing canonical URLs into Bloom filter for resume-safe runs

3. **Source dump provenance** (`phase1_offline/writer.py`, `common/models.py`, `common/repositories.py`)
   - Issue: Documents were missing `source_dump` metadata for traceability
   - Fix: Persist `source_dump` from producer metadata into SQLite

## 2026-02-07: Mapping Metadata Gaps After Resume

**Problem:** Visualizer showed hex doc IDs, missing URLs, and 50% default scores.

**Root causes:**
1. Parquet mapping exports omitted URL/title/domain/content type metadata.
2. Mapping enrichment in `visualizer/server.py` fetched documents by limit, not by doc_id.
3. Mapping script defaulted missing scores to `0.5`, masking data issues.

**Fix:**
1. Export metadata (url/title/domain/content_type/excerpt) into Parquet mappings.
2. Enrich mappings by doc_id list and log missing DB documents.
3. Raise explicit errors when quality/metadata is missing instead of defaulting.

## 2026-02-07: Empty document_texts Table (Critical)

**Problem:** `document_texts` had 0 rows despite 38,984 documents in `documents` table.
Post-processor scoring (deferred scoring pipeline) depends on full text in `document_texts`.

**Root cause:** `BatchWriter._flush_db_buffer()` and base `Writer._process_item()` passed
`conn=self.conn` to both `_doc_repo.insert_batch()` and `_text_repo.insert_batch()`.
Neither repository committed (since `owns_conn=False`), and the writer never called
`self.conn.commit()`. Python's sqlite3 with `isolation_level=""` requires explicit commit.
Without it, inserts were silently lost when the connection was garbage-collected.

**Fix:**
1. Added `self.conn.commit()` in `_flush_db_buffer()` after batch inserts.
2. Added `self.conn.commit()` after each item in base Writer.
3. Added `self.conn.commit()` + `self.conn.close()` in `_on_shutdown()`.
4. Stopped computing `summary` during ingestion (leave empty; full text in `document_texts`).
5. Created `scripts/backfill_texts.py` to populate `document_texts` from parsed archive.

**Backfill:** `python scripts/backfill_texts.py` reads 165 archive files (~165K records) and
inserts texts for existing documents missing from `document_texts`.

## 2026-02-07: Extraction Quality Improvements

**Problem:** Extracted texts were short, incomplete, contained ads, cookie banners,
navigation lists, CSS/JS code fragments, and other non-human-language artifacts.

**Root cause:** trafilatura was called with `include_tables=False` and default precision
mode, producing truncated output. No post-processing cleaned boilerplate.

**Fix (both `_extract_worker_fn` and `ContentExtractor.extract`):**
1. `favor_recall=True` — maximizes content completeness.
2. `include_tables=True` — includes table data (often contains real content).
3. `deduplicate=True` — strips repeated boilerplate paragraphs.
4. Added `_clean_extracted_text()` post-processor that removes:
   - Fenced code blocks and inline code
   - Surviving HTML/XML fragments and HTML entities
   - CSS selectors, property blocks, and media queries
   - Cookie/consent/GDPR banner text
   - Navigation-style pipe-separated lists
   - Ad/CTA lines (subscribe, buy now, download, etc.)
   - Lines that are <40% alphabetic (code, data hashes, etc.)
   - Excessive whitespace

**Tested:** Real SLOP extraction on sample-test-50.tar shows clean human-language output.

## 2026-02-08: Qdrant Writer Stall + Memory Cap

**Problem:** Long-running Phase 1 runs stalled when Qdrant became unstable, backing up
the embedding queue and steadily degrading throughput. macOS also warned about
high RAM usage.

**Fix:**
1. **Memory governor (18GB cap):** Added a process-tree RSS monitor that exits the
   pipeline gracefully when the cap is exceeded.
2. **Async Qdrant batching:** Writer now enqueues embeddings to a dedicated Qdrant
   batch worker, preventing the writer thread from blocking.
3. **Queue backpressure:** Qdrant queue is bounded; when full, embeddings are dropped
   with warnings to prevent unbounded memory growth.

**Config added:**
- `resource_limits.*` (max_rss_gb, check_interval_seconds)
- `qdrant.*` (timeout_seconds, batch_size, batch_timeout_seconds, write_queue_size)
