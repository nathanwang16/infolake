# Technical Engineering Guide: Truth Atlas

## Document Information

| Field           | Value              |
| --------------- | ------------------ |
| Version         | 2.0.0              |
| Last Updated    | February 2026      |
| Status          | Phase 1 Complete, GPU Migration Planned |
| Target Audience | Software Engineers |

---

## 1. Executive Summary

Truth Atlas builds a curated, navigable map of high-quality web content. The system ingests web dumps, embeds and scores documents, and projects them into a navigable 2D atlas with topic clustering.

**Core Principles:**

1. **Core-library-first**: Build a minimal, high-quality `atlas_core` library first. All pipeline modules import from it — never the reverse.
2. **GPU-first computation**: All linear algebra (embedding, scoring, dedup, clustering, projection) runs on GPU via PyTorch tensors. CPU is used only for I/O-bound work (HTTP fetching, HTML extraction).
3. **Hypergraph data model**: Documents and their attributes form a sparse incidence matrix, not a flat relational table. A URL can belong to multiple topics/domains simultaneously without duplication.
4. **Category-theoretic design**: Functors map heterogeneous data sources into a uniform atlas structure. Sheaf-like contexts allow locally conflicting "truths" to coexist without global contradiction.
5. **Two-universe storage**: Operational data (fixed-dimension tensors) lives in GPU/VRAM for math. Variable-length content (text, HTML hashes) lives in Parquet/Arrow on disk, accessed only after GPU identifies relevant IDs.
6. **Separation of concerns**: Each pipeline phase is an independent module communicating through shared stores.
7. **Fault tolerance**: Idempotent operations, micro-batching (10K rows), VRAM treated as cache over Parquet to survive OOM without state loss.
8. **Extract-first storage**: Store extracted text immediately via trafilatura. No raw HTML archival.
9. **Debuggability**: Complete audit trail; any intermediate state inspectable.

**Scale Parameters:**

- Target corpus: 3–50M documents
- Storage: 2TB Samsung T9 SSD
- Compute: Mac Mini M4 (24GB RAM, MPS GPU, 10core CPU) 

---

## 2. System Architecture

### 2.1 Hypergraph Data Model

Standard graphs connect pairs of nodes. Hypergraphs connect *sets* of nodes through hyperedges, solving two problems:

- **Folding**: An entire domain (e.g., `example.com`) is one hyperedge containing all its URLs — no hairball of pairwise links.
- **Multi-identity**: A single URL belongs to multiple topics ("AI", "Ethics", "Code") simultaneously by sitting at the intersection of multiple hyperedges.

**Physical representation:** Sparse incidence matrix \( H \) where rows = URLs, columns = attributes (topics, domains, authors, date bins).

```
H = [ H_topics (Sparse) | H_domains (Sparse) | H_dates (Dense/Binned) | H_embeddings (Dense) ]
```

**Operations via linear algebra:**

| SQL Equivalent             | Tensor Operation                                    |
| -------------------------- | --------------------------------------------------- |
| `SELECT WHERE topic='AI'`  | Dot product: \( H \cdot q \) with query vector \( q \) |
| Find related documents     | Gram matrix: \( H \times H^T \)                    |
| Clustering                 | GPU-accelerated HDBSCAN (cuML via DLPack)           |
| Deduplication              | Cosine similarity: \( E \times E^T \) or random projection hashing |
| FPS sampling               | `torch.cdist` parallel distance argmax              |
| Gap detection              | Mean k-NN distance (high mean = sparse region)      |

### 2.2 Category Theory as Design Pattern

Category theory is used as a **code architecture pattern**, not a database engine.

**Functors (data integration):** Generic adapter classes that map diverse sources (Marginalia SLOP, DMOZ, HN) into the atlas's uniform tensor structure while preserving each source's structural properties. Each dump adapter is a functor: `Source → Atlas`.

**Sheaves (epistemic context):** A logic layer where "truth" is local. Conflicting claims ("Eggs are bad" vs. "Eggs are good") coexist by assigning them to different contexts (open sets) in the hypergraph. No global contradiction — just locally consistent neighborhoods.

### 2.3 Two-Universe Storage Model

| Universe       | Location       | Contents                                                    | Access Pattern        |
| -------------- | -------------- | ----------------------------------------------------------- | --------------------- |
| **Operational** | GPU VRAM / RAM | Composite tensor: embeddings (Float32) + metadata IDs (Int64) + scores (Float32). Pre-allocate spare columns for schema evolution. | Every operation       |
| **Storage**     | T9 SSD         | Parquet/Arrow: full text, titles, summaries, HTML hashes. SQLite: relational metadata. Qdrant: HNSW vector index. | Post-GPU ID retrieval |

The operational tensor is the primary working surface. The storage universe is accessed only after GPU identifies specific document IDs to retrieve.

### 2.4 Data Flow

```
PHASE 1 (Batch):
  Dump → Producer(filter)
    → AsyncFetch(aiohttp)                                [CPU: I/O-bound]
    → ExtractPool(trafilatura, ProcessPool)               [CPU: DOM parsing]
    → Arrow shared memory buffer                          [CPU→GPU handoff]
    → GPU Embed(nomic-embed-text-v1.5, PyTorch)           [GPU]
    → Writer(Parquet + Qdrant + SQLite)                   [CPU: I/O]
    → GPU PostProcess(score, dedup, cluster, project)     [GPU]
    → Atlas

PHASE 2 (Active, future):
  Atlas → GPU Gap Detection(k-NN density) → Query Synthesis → Search API
    → Fetch → Extract → Embed → Score → Atlas (continuous loop)
```

### 2.5 Storage Layout

```
parsed_archive/
├── YYYY/MM/DD/
│   ├── batch_*.jsonl.zst       # Parsed content, zstd compressed
│   └── manifest.json           # URL → archive path
tensors/
├── embeddings.pt               # PyTorch tensor checkpoint
├── metadata.pt                 # Metadata ID tensor
└── scores.pt                   # Score tensor
```

---

## 3. Core Library (`atlas_core/`)

### 3.1 Design Philosophy

`atlas_core` is a minimal, dependency-light library that every other module imports. It defines the type system, protocols, tensor operations, and configuration interface. It contains **zero** pipeline logic — only primitives and contracts.

Development order: **build `atlas_core` first**, then all pipeline modules reference it.

### 3.2 Architecture

```
atlas_core/
├── __init__.py           # Public API
├── types.py              # Domain types: DocumentID, URL, Embedding, HyperedgeLabel
├── protocols.py          # Protocol definitions (see below)
├── tensor_ops.py         # GPU tensor operations (embed, score, dedup, project)
├── hypergraph.py         # Sparse incidence matrix construction and queries
├── functors.py           # Base functor class for source adapters
├── config.py             # Singleton config loader (dot-notation, config.json)
├── errors.py             # Custom exceptions (never silent defaults)
└── logging.py            # Structured JSON logger (rotating, per-module)
```

### 3.3 Protocols

All pipeline modules program against these protocols. Implementations are swappable.

| Protocol         | Methods                                      | Used By               |
| ---------------- | -------------------------------------------- | --------------------- |
| `Embedder`       | `encode(texts) → Tensor`                    | Phase 1 ingestion     |
| `Scorer`         | `score(features, weights) → Tensor`          | Curator               |
| `Deduplicator`   | `find_duplicates(embeddings) → mask`         | Post-processor        |
| `Projector`      | `fit_transform(embeddings) → coords_2d`      | Mapper                |
| `Clusterer`      | `fit_predict(coords) → labels`               | Mapper                |
| `AxisScorer`     | `score(domain, quality, type) → float`       | Mapper Z-axis         |
| `SourceFunctor`  | `read(path) → Iterator[Record]`              | Dump adapters         |
| `ScoringMetric`  | `compute(text, words, sentences, meta) → float` | Curator metrics    |

### 3.4 Tensor Operations (`tensor_ops.py`)

All core math runs on GPU via PyTorch. These are the building blocks every module calls.

```
# Embedding: batch encode via model on GPU
embed(texts, model, device) → Tensor[N, D]

# Scoring: weighted feature projection (dot product, no iteration)
score(feature_matrix, weight_vector) → Tensor[N]
// Replaces: per-document scoring loop

# Wilson score: vectorized element-wise across entire corpus
wilson_score(positive_counts, total_counts, z=1.96) → Tensor[N]

# Deduplication: cosine similarity via gram matrix
find_duplicates(E, threshold=0.95) → duplicate_mask
// E @ E.T computed on GPU, threshold applied

# FPS: parallel distance argmax
farthest_point_sample(embeddings, k, quality_weights, α) → selected_ids
// torch.cdist for parallel distance computation

# Gap detection: inverted density estimation
find_gaps(embeddings, k_neighbors) → gap_scores
// High mean k-NN distance = sparse region = gap

# Projection: UMAP on GPU (or cuML via DLPack)
project_2d(embeddings, n_neighbors, min_dist) → coords_2d
```

### 3.5 Hypergraph (`hypergraph.py`)

Constructs and queries the sparse incidence matrix.

```
BUILD_HYPERGRAPH(documents, topic_labels, domain_labels):
    H_topics ← sparse_matrix(docs × topics)     # from clustering
    H_domains ← sparse_matrix(docs × domains)    # from URL parsing
    H_dates ← dense_binned(docs × date_bins)     # publication dates
    H_embeddings ← dense(docs × embedding_dim)   # from embedder

    RETURN composite_tensor = concat(H_topics, H_domains, H_dates, H_embeddings)

QUERY(H, query_vector):
    scores ← H @ query_vector        # dot product filtering
    RETURN top_k(scores)
```

### 3.6 Configuration & Errors

- `config.py`: Singleton loader. `config.get("section.key", default)`. Sources from `config.json`.
- `errors.py`: All parameters required — no silent defaults. Missing values raise `AtlasConfigError`.
- `logging.py`: Per-module JSONL to `logs/`. Console INFO+, file DEBUG+. Rotating 10MB/5 backups.

---

## 4. Phase 1: Batch Dump Processing

### 4.1 Overview

Two-stage pipeline. Stage 1: high-throughput ingestion (CPU extraction → GPU embedding → store). Stage 2: GPU-accelerated post-processing (score, dedup, cluster, project). All parameters from `config.json`.

### 4.2 Data Sources

| Source              | Scale     | Quality Signal           | Adapter Status |
| ------------------- | --------- | ------------------------ | -------------- |
| Marginalia SLOP     | 100K–1M   | Pre-filtered for quality | Implemented    |
| DMOZ/ODP archive    | ~4M URLs  | Human curation           | Planned        |
| Pinboard/Delicious  | 1–10M     | Crowd tagging            | Planned        |
| Academic citations  | 10–100M   | Academic provenance      | Planned        |
| Hacker News         | ~5M       | Community voting         | Planned        |

Each source adapter is a `SourceFunctor` (from `atlas_core`) mapping source-specific records into the atlas's uniform `Record` type.

### 4.3 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 1: INGESTION (CPU I/O → GPU embed → store)      │
│                                                                     │
│  Producer ──► url_queue ──► AsyncFetcher ──► html_queue            │
│                                                  │                  │
│  ExtractPool(trafilatura) ◄──────────────────────┘                 │
│       │                                                             │
│  Arrow shared memory ──► GPU Embedder(PyTorch) ──► embed_queue     │
│                                                        │            │
│  Writer(Parquet + Qdrant + SQLite + Meilisearch) ◄─────┘           │
├─────────────────────────────────────────────────────────────────────┤
│              STAGE 2: GPU POST-PROCESSING (all on GPU)             │
│                                                                     │
│  Load embeddings → tensor_ops.score() → tensor_ops.find_duplicates │
│  → tensor_ops.project_2d() → tensor_ops.cluster()                  │
│  → Export Parquet/Arrow for visualizer                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.4 Stage 1: Ingestion

**CPU-bound components** (I/O and DOM parsing — cannot move to GPU):

**Producer:** Reads dump via `SourceFunctor`. Filters URLs, deduplicates against existing DB. Pushes to `url_queue`.

**AsyncFetcher:** aiohttp event loop. Concurrency from `config.json → batch_processing.fetch_concurrency`. Optional playwright for JS sites.

**ExtractPool:** ProcessPoolExecutor with trafilatura (`favor_recall=True`, `include_tables=True`). Post-extraction cleanup strips artifacts. This is the only multiprocess CPU component — DOM parsing is inherently recursive and cannot be GPU-accelerated. Hand-off to GPU via Arrow shared memory buffers (not pickle).

**GPU-bound components:**

**Embedder:** PyTorch on MPS/CUDA. Model: `nomic-embed-text-v1.5` (8192 token context, Matryoshka representation for dynamic dimension resizing). Batch accumulation with OOM protection (auto-halve batch size).

**Writer:** Single-threaded store-only. Writes to: SQLite (documents + document_texts), Qdrant (vectors via async batch), Parquet archive (compressed JSONL), Meilisearch (metadata index). All with `quality_profile_used='pending'`.

**Memory Governor:** Background thread monitors RSS against `config.json → resource_limits.max_rss_gb`. Graceful shutdown on exceed.

### 4.5 Stage 2: GPU Post-Processing

Runs entirely on GPU. Operates on stored embeddings from Stage 1.

```
GPU_POST_PROCESS():
    E ← load_embeddings_as_tensor()          # from Qdrant or .pt checkpoint
    texts ← document_texts.get_unscored()    # CPU: text retrieval

    // Scoring: weighted feature projection (single dot product, not per-doc loop)
    features ← extract_features(texts)       # CPU: text analysis
    F ← to_tensor(features, device='gpu')
    scores ← tensor_ops.score(F, weight_vectors[content_type])

    // Wilson confidence: vectorized across entire corpus
    wilson ← tensor_ops.wilson_score(positive_counts, total_counts)

    // Deduplication: GPU gram matrix
    dup_mask ← tensor_ops.find_duplicates(E, threshold=0.95)

    // Clustering: cuML HDBSCAN via DLPack (zero-copy from PyTorch)
    coords_2d ← tensor_ops.project_2d(E)
    labels ← tensor_ops.cluster(coords_2d)

    // Update DB + export Arrow for visualizer
    documents.batch_update(scores, wilson, labels, dup_mask)
    export_arrow(coords_2d, labels, scores)   # zero-copy to deck.gl
```

### 4.6 Speed Characteristics

| Component        | Throughput         | Bound     | GPU Migration Impact      |
| ---------------- | ------------------ | --------- | ------------------------- |
| Producer         | ~500K URLs/sec     | Disk I/O  | No change (CPU)           |
| AsyncFetcher     | ~100 URLs/sec      | Network   | No change (CPU)           |
| ExtractPool      | ~50 docs/sec       | CPU       | Stays CPU (DOM parsing)   |
| Embedder         | ~1000+ docs/sec    | GPU       | Faster with larger model  |
| Writer           | ~1200 docs/sec     | I/O       | Arrow buffers reduce copy |
| **Post-process** | **Entire corpus**  | **GPU**   | **Batch, not per-doc**    |

---

## 5. Database Schema

### 5.1 Operational Tensor Store (GPU)

Composite tensor loaded into VRAM for all math operations. Pre-allocate spare columns for schema evolution.

| Sub-tensor       | Shape           | Type    | Content                              |
| ---------------- | --------------- | ------- | ------------------------------------ |
| Embeddings       | (N, D)          | Float32 | Document embeddings (D=768 nomic)    |
| Metadata IDs     | (N, M)          | Int64   | Encoded domain/topic/date IDs        |
| Scores           | (N, S)          | Float32 | Quality, wilson, importance scores   |
| Spare            | (N, 16-M-S)     | Float32 | Reserved for schema evolution        |

Checkpointed to `tensors/*.pt`. Micro-batched (10K rows) to survive OOM.

### 5.2 SQLite Tables

Auto-created by `atlas_core` on startup. Path: `config.json → database.sqlite_path`.

**documents** — Primary metadata table.

| Column                | Type      | Key         | Description                                |
| --------------------- | --------- | ----------- | ------------------------------------------ |
| id                    | TEXT      | PK          | SHA256(canonical_url)[:16]                 |
| canonical_url         | TEXT      | UNIQUE NN   | Normalized URL                             |
| title, author         | TEXT      |             | Extracted metadata                         |
| domain                | TEXT      | NN          | From URL                                   |
| content_hash          | TEXT      | NN          | SimHash (64-bit hex)                       |
| content_length        | INTEGER   |             | Character count                            |
| language              | TEXT      |             | ISO 639-1                                  |
| detected_content_type | TEXT      |             | scientific, technical_code, essay, news, docs, other |
| quality_score         | REAL      |             | 0–1                                        |
| wilson_score          | REAL      |             | Sample-size-aware confidence               |
| importance_score      | REAL      |             | Z-axis (domain authority)                  |
| cluster_id            | INTEGER   |             | HDBSCAN label (-1 = noise)                 |
| quality_profile_used  | TEXT      |             | 'pending' or 'scored'                      |
| source_phase          | TEXT      |             | batch_dump / active_exploration            |
| source_dump           | TEXT      |             | Which dump file                            |
| status                | TEXT      |             | active, duplicate, rejected                |
| raw_html_hash         | TEXT      |             | SHA256 of original HTML                    |
| created_at            | TIMESTAMP | DEFAULT NOW |                                            |
| updated_at            | TIMESTAMP | DEFAULT NOW |                                            |

**document_texts** — Full text for deferred scoring. `(doc_id TEXT PK FK, text TEXT NN)`

**url_queue** — Processing queue. `(canonical_url TEXT PK, original_url TEXT NN, status TEXT DEFAULT 'pending', source TEXT NN, priority REAL DEFAULT 0.5, retry_count INTEGER DEFAULT 0, ...timestamps)`

**dump_processing_jobs** — Job audit trail. `(id TEXT PK, dump_name TEXT NN, total_urls, filtered_urls, extracted/embedded/accepted/rejected counts, status, ...timestamps)`

**golden_set** — Calibration exemplars. `(id INTEGER PK, url TEXT UNIQUE, label TEXT, content_type, domain, raw_metrics JSON, version INTEGER)`

**coverage_metrics** — Atlas health snapshots. `(topic_gini, domain_gini, orphan_rate, cluster_count, largest_cluster_pct)`

**cluster_stats** — HDBSCAN cluster characteristics. `(cluster_id INTEGER NN, doc_count, avg_quality, quality_std, is_content_farm, is_authority)`

**detected_gaps** / **exploration_provenance** — Phase 2, not yet active.

### 5.3 Qdrant Collection

| Property         | Value                              |
| ---------------- | ---------------------------------- |
| Collection       | `atlas_embeddings`                 |
| Vector dimension | 768 (nomic-embed-text-v1.5)        |
| Distance metric  | Cosine                             |
| Quantization     | Scalar (int8)                      |
| Storage          | On-disk (memory-mapped)            |

### 5.4 Meilisearch Index

`atlas_documents` — searchable fields: title, domain, url. Results limit: 20.

---

## 6. Module Specifications

All modules import from `atlas_core`. No module contains its own math, config loading, or logging — those live in the core.

### 6.1 Phase 1 Offline (`phase1_offline/`)

Two-stage batch pipeline. See Section 4 for full workflow.

| File                | Purpose                                      |
| ------------------- | -------------------------------------------- |
| `pipeline.py`       | Orchestrator, queues, threads                |
| `producer.py`       | Dump reading via `SourceFunctor`             |
| `worker.py`         | AsyncFetcher, ExtractPool, GPU Embedder      |
| `writer.py`         | Writer + BatchWriter (store-only)            |
| `dump_adapters.py`  | `SourceFunctor` implementations (SLOP)       |
| `deduplication.py`  | Calls `tensor_ops.find_duplicates`           |
| `fps_sampler.py`    | Calls `tensor_ops.farthest_point_sample`     |
| `post_processor.py` | Stage 2 GPU post-processing orchestrator     |

### 6.2 Storage (`storage/`)

| File               | Purpose                                      |
| ------------------ | -------------------------------------------- |
| `atlas_store.py`   | Unified SQLite + Qdrant query interface      |
| `parquet_store.py` | Parquet/Arrow for mappings and bulk exports   |

Arrow format enables zero-copy transfer: GPU tensor → Arrow buffer → deck.gl WebGL. Bypasses JSON serialization entirely.

### 6.3 Curator (`curation/scoring/`)

Scores documents using protocol-based metrics. **GPU migration:** scoring becomes a single dot product of feature matrix × weight vector, replacing per-document iteration.

```
curation/scoring/
├── pipeline.py       # ScoringPipeline orchestrator
├── protocols.py      # ScoringMetric protocol (from atlas_core)
├── registry.py       # MetricRegistry for runtime discovery
├── detection.py      # Content type detection
├── aggregation.py    # GPU: tensor_ops.score() + tensor_ops.wilson_score()
└── metrics/          # citation, depth, methodology, reputation,
                      # specificity, structure, writing
```

**Scoring flow:**

```
SCORE_BATCH(texts, metadata):
    raw_metrics ← registry.compute_all(texts)         # per-metric CPU
    F ← to_tensor(raw_metrics)                         # CPU → GPU
    content_types ← detect_types(texts, metadata)
    scores ← tensor_ops.score(F, weights[content_types])   # GPU dot product
    wilson ← tensor_ops.wilson_score(F)                     # GPU vectorized
    RETURN scores, wilson, content_types
```

**Content-type weights:**

| Content Type   | Threshold | Top Weights                                          |
| -------------- | --------- | ---------------------------------------------------- |
| scientific     | 0.50      | citation (0.28), methodology (0.24), peer_review (0.22) |
| technical_code | 0.45      | code_quality (0.27), recency (0.23), completeness (0.21) |
| personal_essay | 0.40      | writing (0.32), specificity (0.26), authenticity (0.18) |
| news           | 0.50      | attribution (0.31), perspectives (0.24), factual (0.21) |
| documentation  | 0.45      | completeness (0.32), accuracy (0.26), recency (0.18) |

**Calibration (future):** Replace scikit-learn logistic regression with single-layer PyTorch neural network — keeps entire pipeline in GPU memory.

**Deduplication:** `tensor_ops.find_duplicates(E, threshold)` on GPU. Replaces CPU SimHash/MinHash.

### 6.4 Mapper (`mapping/`)

Transforms embeddings into visual coordinates. **GPU migration:** UMAP via cuML (DLPack zero-copy from PyTorch), HDBSCAN via cuML.

```
mapping/
├── pipeline.py       # MappingPipeline orchestrator
├── protocols.py      # Projector, Clusterer, AxisScorer (from atlas_core)
├── projectors.py     # UMAPProjector (future: cuML GPU)
├── clusterers.py     # HDBSCANClusterer (future: cuML GPU)
└── axis_scorers.py   # DomainAuthorityAxisScorer
```

**Flow:**

```
COMPUTE_MAPPING():
    E ← load_embeddings()
    coords_2d ← projector.fit_transform(E)          # UMAP
    labels ← clusterer.fit_predict(coords_2d)        # HDBSCAN
    z_scores ← axis_scorer.score_batch(documents)    # importance
    export_arrow(coords_2d, labels, z_scores)         # zero-copy to viz
```

**Output:** Arrow binary blob consumed directly by deck.gl WebGL buffer. JSON fallback for compatibility. Parquet for analysis.

Config: `config.json → mapping` (umap.neighbors, umap.min_dist, umap.metric, hdbscan.min_cluster_size, sample_for_fit, output_path).

### 6.5 Visualizer (`visualizer/`)

Interactive deck.gl WebGL visualization. **GPU migration:** Arrow binary blobs feed directly into ScatterplotLayer — no JSON serialization bottleneck.

```
Browser (deck.gl + Arrow)      Python Server (http.server)
┌──────────────────┐           ┌──────────────────────────┐
│ ScatterplotLayer │ ──REST──► │ AtlasAPIHandler          │
│ OrthographicView │           │   /api/stats             │
│ Arrow WebGL buf  │           │   /api/mappings (Arrow)  │
│ LineLayer        │           │   /api/documents         │
└──────────────────┘           │   /api/clusters/:id      │
                               │   /api/search            │
                               └──────────────────────────┘
```

**Rendering:** ScatterplotLayer renders UMAP coordinates. Color = cluster ID. Size = importance (Z-axis). LineLayer for intra-cluster connections (togglable).

**Interactivity:** Scroll zoom, drag pan, click-to-open URL, hover tooltip, sidebar cluster filter, quality slider, content type dropdown, mapping selector, Meilisearch full-text search with GPU-based dimming of non-matches.

**API:** `/api/stats`, `/api/mappings`, `/api/mapping-list`, `/api/documents`, `/api/documents/{id}`, `/api/clusters`, `/api/clusters/{id}`, `/api/search?q=X`.

Config: `config.json → visualizer` (host, port, static_dir, dot_radius, connection_*).

---

## 7. Common Module (`common/`)

Shared infrastructure. Being gradually absorbed into `atlas_core` — new code should import from core where possible.

| File                    | Purpose                                          |
| ----------------------- | ------------------------------------------------ |
| `database.py`           | SQLite connection manager, schema init           |
| `repositories.py`       | DocumentRepository, DocumentTextRepository, etc. |
| `models.py`             | Type-safe dataclasses                            |
| `qdrant_manager.py`     | Qdrant client wrapper, health checks             |
| `meilisearch_manager.py`| Meilisearch client, graceful degradation         |

Thread safety: repositories open per-method connections. Absolute path resolution prevents CWD issues.

---

## 8. GPU-Native Capabilities (Future)

Enabled by the linear algebra engine, not possible in the CPU version:

**Truth Tensor:** Train a linear probe on embeddings to project onto a "Consensus vs. Dissent" axis — quantifiable epistemic stance metric.

**Dynamic Focus:** Attention masks \( \text{Softmax}(E \cdot q) \) dynamically re-weight quality scores based on user queries. The atlas "morphs" ranking criteria in real-time.

**Micro-batching:** Process 10K rows at a time. VRAM is a cache over Parquet storage — OOM crashes don't wipe state.

---

## Appendix A: Configuration Reference

All configuration in `config.json`. Access: `atlas_core.config.get("section.key")`.

| Section              | Key parameters                                                        |
| -------------------- | --------------------------------------------------------------------- |
| `paths`              | `data_dir`, `logs_dir`, `dumps_dir`, `archive_dir`, `parsed_archive_dir`, `exports_dir`, `mappings_dir` |
| `database`           | `sqlite_path`                                                         |
| `qdrant`             | `url`, `collection`, `quantization`, `on_disk`, `batch_size`, `write_queue_size` |
| `meilisearch`        | `url`, `api_key`, `index`, `search_results_limit`                     |
| `embedding`          | `model` (nomic-embed-text-v1.5), `device`, `batch_size`, `max_tokens` |
| `batch_processing`   | `default_dump`, `workers`, `worker_batch_size`, `url_queue_size`, `embed_queue_size`, `fetch_concurrency`, `fetch_timeout`, `extract_processes`, `quality_weight_alpha` |
| `resource_limits`    | `max_rss_gb`, `check_interval_seconds`                                |
| `content_extraction` | `extractor`, `fallback`, `min_length`, `max_length`                   |
| `deduplication`      | `cosine_threshold` (replaces simhash/minhash thresholds)              |
| `mapping`            | `umap.*`, `hdbscan.*`, `sample_for_fit`, `output_path`, `output_format` (arrow/json) |
| `visualizer`         | `host`, `port`, `static_dir`, `dot_radius`, `connection_*`           |
| `gpu`                | `device` (mps/cuda/cpu), `micro_batch_size`, `spare_columns`         |

---

## Appendix B: Revision History

| Version | Date     | Changes |
| ------- | -------- | ------- |
| 0.3–0.8 | Jan 2025 | Initial drafts through cartographic techniques integration |
| 1.0.0   | Jan 2026 | MVP: batch pipeline, UMAP+HDBSCAN mapping, web visualizer |
| 1.1.0   | Feb 2026 | Two-stage pipeline, repository pattern, protocol-based scoring/mapping, async fetcher, multiprocess extraction, GPU embedder with OOM protection |
| 1.2.0–4 | Feb 2026 | Meilisearch search, mapping metadata enrichment, writer commit fix, extraction quality improvements, memory governor, async Qdrant batch writer |
| 1.3.0   | Feb 2026 | Guide overhaul: consolidated Phase 1 workflow, full DB schema, detailed module specs, config.json references throughout |
| 2.0.0   | Feb 2026 | **GPU-first architecture:** Hypergraph data model (sparse incidence matrix), category theory design patterns (functors, sheaves), two-universe storage (GPU tensors + Parquet/Arrow), core-library-first development (`atlas_core`), PyTorch tensor operations for scoring/dedup/clustering/projection, nomic-embed-text-v1.5 (8192 tokens), cuML GPU clustering via DLPack, zero-copy Arrow→deck.gl visualization, micro-batching fault tolerance. CPU retained only for HTTP fetching and HTML extraction. Document compacted ~35%. |

---

## Appendix C: Implementation Status

### Phase 1 Components (Complete — pre-GPU migration)

| Component                    | Location                             | Status   |
| ---------------------------- | ------------------------------------ | -------- |
| Batch Ingestion (Stage 1)    | `phase1_offline/pipeline.py`         | Complete |
| AsyncFetcher + ExtractPool   | `phase1_offline/worker.py`           | Complete |
| ConcurrentEmbedder (MPS)     | `phase1_offline/worker.py`           | Complete |
| Writer + BatchWriter         | `phase1_offline/writer.py`           | Complete |
| Post-Processing (Stage 2)    | `phase1_offline/post_processor.py`   | Complete |
| SLOP Dump Adapter            | `phase1_offline/dump_adapters.py`    | Complete |
| Quality Scoring Package      | `curation/scoring/`                  | Complete |
| SimHash + MinHash Dedup      | `phase1_offline/deduplication.py`    | Complete |
| Repository Layer             | `common/repositories.py`             | Complete |
| SQLite + Qdrant + Meilisearch| `common/`                            | Complete |
| UMAP Mapping Pipeline        | `mapping/pipeline.py`                | Complete |
| deck.gl Visualizer           | `visualizer/server.py`               | Complete |
| Health Monitor               | `monitor/health.py`                  | Complete |
| Parquet/Arrow Store          | `storage/`                           | Complete |

### Planned (GPU Migration)

| Component                    | Priority | Notes                                          |
| ---------------------------- | -------- | ---------------------------------------------- |
| `atlas_core` library         | P0       | Build first — all modules depend on it         |
| `tensor_ops.py`              | P0       | GPU scoring, dedup, FPS, projection primitives |
| `hypergraph.py`              | P1       | Sparse incidence matrix construction           |
| `functors.py`                | P1       | Base functor class for source adapters         |
| nomic-embed-text-v1.5        | P1       | Replace bge-small, 8192 token context          |
| GPU post-processor           | P1       | Replace CPU scoring loop with dot product      |
| cuML HDBSCAN/UMAP            | P2       | DLPack zero-copy from PyTorch                  |
| Arrow→deck.gl pipeline       | P2       | Zero-copy visualization, bypass JSON           |
| Truth Tensor (epistemic)     | P3       | Linear probe on consensus/dissent axis         |
| Dynamic Focus (attention)    | P3       | Query-dependent quality re-weighting           |

### Architecture Deviations from Original Guide

| Original Specification  | Actual Implementation                   | Rationale                         |
| ----------------------- | --------------------------------------- | --------------------------------- |
| Single-stage pipeline   | Two-stage (ingest + post-process)       | 10x throughput, re-scorable       |
| CPU scoring loops       | GPU tensor dot product (planned)        | Orders of magnitude faster        |
| SimHash/MinHash (CPU)   | GPU gram matrix cosine (planned)        | Batch dedup on entire corpus      |
| bge-small (384D, 512tok)| nomic-embed-text-v1.5 (768D, 8192tok)  | Full essays without truncation    |
| JSON viz export         | Arrow binary blob (planned)             | Zero-copy to WebGL                |
| Flat relational model   | Hypergraph incidence matrix (planned)   | Multi-identity, efficient queries |

---

## Appendix D: Cartographic Techniques

| Technique          | Application               | Benefit                                    |
| ------------------ | ------------------------- | ------------------------------------------ |
| Topic-cluster CV   | Golden Set calibration    | Prevents 28-40% accuracy overestimate      |
| QADI metrics       | Verification diagnostics  | Separates quantity from allocation errors   |
| Wilson score       | Hidden gem ranking        | Sample-size-aware quality confidence        |
| HDBSCAN clustering | Quality topology analysis | Content farms, authority clusters, orphans  |
| Gini coefficient   | Coverage monitoring       | Tracks topic/domain balance                |
