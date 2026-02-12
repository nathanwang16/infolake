# Technical Engineering Guide: Truth Atlas

## Document Information

| Field           | Value              |
| --------------- | ------------------ |
| Version         | 1.0.0              |
| Last Updated    | January 2026       |
| Status          | MVP Semi Complete  |
| Target Audience | Software Engineers |

---

## 1. Executive Summary

Truth Atlas is a continuous pipeline that builds a curated, navigable map of high-quality web content. The system operates in two phases: **batch processing of existing dumps** (using farthest-point sampling for maximum coverage) and **active exploration** (using gap detection and expert query synthesis to discover what dumps missed).

**Core Design Principles:**

1. **Separation of concerns**: Each pipeline phase is an independent module communicating through shared data stores
2. **Data source agnosticism**: The atlas accepts documents from any source; source-specific logic is isolated in adapters
3. Essential core package agnostic: The project should maintain a high quality, reduced implementation core code library. All the actual implementations should use that library.
4. ~~**Maximum coverage sampling**: Use theoretically principled farthest-point sampling to ensure uniform coverage across embedding space~~
5. **Simple parallelism**: Producer-consumer pattern with queues; no distributed coordination needed for single-machine deployment
6. ~~**Epistemic preservation**: Classify stance (consensus/contested/dissent) separately from quality; preserve well-reasoned minority views~~
7. ~~**Domain adaptivity**: Different content types are scored with calibrated weights derived from Golden Set evaluation~~
8. **Fault tolerance**: Any module can crash and restart without corrupting state; all operations are idempotent or transactional
9. **Debuggability**: Complete audit trail for every document; any intermediate state can be inspected or visualized
10. **Extract-first storage**: Store extracted text and metadata immediately; no raw HTML archival (use trafilatura/readability for extraction)
11. **Embedding stability**: Use stable embedding model (bge-small) for navigation; add rerankers for quality later if needed
12. ~~**Cartographic validation**: Apply spatial statistics techniques (topic-cluster CV, QADI metrics, Gini coefficients) to ensure calibration generalizes and coverage is equitable~~

**Scale Parameters:**

- Target corpus: 3-50 million documents
- Storage budget: 2TB (Samsung T9 SSD attached to server)
- Compute: Single server (Mac Mini M4, 24GB RAM)

---

## 2. System Architecture

### 2.1 Two-Phase Processing Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRUTH ATLAS: TWO-PHASE MODEL                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: BATCH DUMP PROCESSING (Offline, Parallel)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │Marginalia│ │  DMOZ    │ │ Academic │ │ Pinboard │  ... more     │   │
│  │  │  Dump    │ │ Archive  │ │Citations │ │ Archive  │   dumps       │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘               │   │
│  │       │            │            │            │                      │   │
│  │       └────────────┴─────┬──────┴────────────┘                      │   │
│  │                          ▼                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │              PRODUCER-CONSUMER PIPELINE                       │  │   │
│  │  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │  │   │
│  │  │  │ PRODUCER │──►│  QUEUES  │──►│ WORKERS  │──►│  WRITER  │  │  │   │
│  │  │  │ (filter) │   │          │   │ (N thrd) │   │ (single) │  │  │   │
│  │  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │                 ┌─────────────────┐                                 │   │
│  │                 │  GLOBAL INDEX   │  Single-writer, no locks        │   │
│  │                 │  (Qdrant HNSW)  │                                 │   │
│  │                 └────────┬────────┘                                 │   │
│  │                          │                                          │   │
│  │  Algorithm: Quality-Weighted Farthest Point Sampling (Lazy Greedy) │   │
│  │  Guarantee: 2-approximation to optimal coverage                     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│                        ┌─────────────────┐                                 │
│                        │  ATLAS CORPUS   │                                 │
│                        └────────┬────────┘                                 │
│                                 │                                          │
│  PHASE 2: ACTIVE EXPLORATION (Online, Continuous)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐         │   │
│  │  │ Gap Detection│ ──► │Query Inversion│ ──► │Search APIs  │         │   │
│  │  │(Lonely Nodes)│      │(Expert Synth)│      │(Marginalia, │         │   │
│  │  └─────────────┘      └─────────────┘      │ Kagi, etc.) │         │   │
│  │        ▲                                    └──────┬──────┘         │   │
│  │        │                                           │                │   │
│  │        │              ┌────────────────────────────┘                │   │
│  │        │              ▼                                             │   │
│  │        │     ┌─────────────────┐                                    │   │
│  │        └─────│ Fetch + Score   │──► Add to Atlas (if novel+quality)│   │
│  │              └─────────────────┘                                    │   │
│  │                                                                      │   │
│  │  Algorithm: Lonely node detection (sparse region discovery)         │   │
│  │  Rate: ~100 novel documents per 6-hour cycle                        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRUTH ATLAS SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DATA SOURCES                                                                │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐          │
│  │     DUMPS (Phase 1)         │  │   LIVE SOURCES (Phase 2)    │          │
│  │  ┌───────┐ ┌───────┐       │  │  ┌───────┐ ┌───────┐        │          │
│  │  │Margin-│ │ DMOZ  │       │  │  │Margin-│ │ Kagi  │        │          │
│  │  │alia   │ │Archive│ ...   │  │  │alia   │ │ API   │ ...    │          │
│  │  │ Dump  │ │       │       │  │  │ API   │ │       │        │          │
│  │  └───────┘ └───────┘       │  │  └───────┘ └───────┘        │          │
│  └─────────────────────────────┘  └─────────────────────────────┘          │
│                │                              │                             │
│  ══════════════╪══════════════════════════════╪═════════════════════════   │
│                │                              │                             │
│  PROCESSING    ▼                              ▼                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        BATCH SAMPLER (Phase 1)                        │  │
│  │  Producer → Queues → Workers → Writer (no locks, no agents)          │  │
│  └──────────────────────────────────────┬───────────────────────────────┘  │
│                                         │                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          EXPLORER (Phase 2)                           │  │
│  │  Lonely Node Detection │ HDBSCAN Clustering │ Query Inversion │ Fetch │  │
│  └──────────────────────────────────────┬───────────────────────────────┘  │
│                                         │                                   │
│                                         ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                            CURATOR                                    │  │
│  │  Content type detection │ Golden-set calibrated scoring │ Wilson     │  │
│  │  score ranking │ Epistemic classification │ Deduplication │ Accept   │  │
│  └──────────────────────────────────────┬───────────────────────────────┘  │
│                                         │                                   │
│                                         ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                            MAPPER                                     │  │
│  │  Semantic projection (UMAP) │ Importance scoring (PageRank)          │  │
│  │  Topic clustering │ Diversity metrics                                 │  │
│  └──────────────────────────────────────┬───────────────────────────────┘  │
│                                         │                                   │
│                                         ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                            STORAGE                                    │  │
│  │  Qdrant (vectors) │ SQLite (metadata) │ Parquet (mappings)           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ══════════════════════════════════════════════════════════════════════    │
│                                                                              │
│  TOOLS                                                                       │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌──────────────┐ │
│  │  VISUALIZER    │ │  ORCHESTRATOR  │ │   VERIFIER     │ │  CALIBRATOR  │ │
│  │  Web UI        │ │  Scheduling    │ │  QADI metrics  │ │  Golden Set  │ │
│  │  API server    │ │  Coordination  │ │  Topic-CV      │ │  Weight tuning│ │
│  └────────────────┘ └────────────────┘ └────────────────┘ └──────────────┘ │
│                                                                              │
│  ┌────────────────┐                                                         │
│  │    MONITOR     │ Gini coefficient │ Cluster health │ Drift detection    │
│  └────────────────┘                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Data Flow

```
PHASE 1 (Batch) - Producer-Consumer with Extract-First:
Dump → Producer(filter) → url_queue → Workers(fetch,extract,embed) → embed_queue → Writer(FPS,dedupe,score) → Atlas
                                              │
                                              ▼
                                     STORE PARSED CONTENT + METADATA
                                     (extracted text, title, author, date, raw_html_hash)

PHASE 2 (Active):
Atlas → Lonely Node Detection → HDBSCAN Clustering → Query Inversion → Search API → Fetch → Extract → Embed → Dedupe → Score → Atlas
                ▲                                                                                                              │
                └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              (continuous loop)
```

### 2.4 Storage Architecture

With 24GB RAM and 2TB Samsung T9 SSD:

| Tier | Location           | Contents                                                                                             | Access Pattern              |
| ---- | ------------------ | ---------------------------------------------------------------------------------------------------- | --------------------------- |
| HOT  | RAM / Internal SSD | Qdrant HNSW index (quantized), SQLite WAL files, active working set                                  | Frequent reads/writes       |
| WARM | T9 SSD             | Full vector store (memory-mapped), document content (compressed), SQLite databases, Parquet mappings | Regular reads, batch writes |
| COLD | T9 SSD (archived)  | **Parsed content archive**, historical embeddings (if model changes), full audit logs          | Rare access                 |

**Extract-First Storage (No Raw HTML):**

Instead of storing raw HTML, extract text and metadata immediately using trafilatura/readability. Store only the parsed content, saving significant storage space and simplifying the pipeline.

```
parsed_archive/
├── 2025/
│   ├── 01/
│   │   ├── 27/
│   │   │   ├── batch_001.jsonl.zst  # Parsed content + metadata, zstd compressed
│   │   │   ├── batch_002.jsonl.zst
│   │   │   └── manifest.json        # URL -> archive path mapping
```

**Each archived record contains:** `url`, `raw_html_hash` (provenance), `extracted_text`, `title`, `author`, `publication_date`, `fetch_date`, `extractor_version`, `http_status`, `content_type`

---

## 3. Phase 1: Batch Dump Processing

### 3.1 Data Sources for Dumps

| Source                      | Description                                             | Scale             | Quality Signal                   |
| --------------------------- | ------------------------------------------------------- | ----------------- | -------------------------------- |
| Marginalia blog corpus      | ~4,500 quality sites, larger dumps available on request | 100K-1M           | Pre-filtered for quality         |
| DMOZ/ODP archive            | Human-curated directory (1990s-2000s)                   | ~4M URLs          | Human curation (many dead links) |
| Pinboard/Delicious archives | Social bookmarking exports                              | 1-10M             | Crowd curation via tagging       |
| Academic citation URLs      | Extracted from Semantic Scholar, arXiv                  | 10-100M           | Academic provenance              |
| Common Crawl (filtered)     | Use CC index to sample, don't process raw WARCs         | Petabyte (sample) | Needs heavy filtering            |
| Hacker News archive         | All submitted URLs with scores                          | ~5M               | Community voting                 |
| Lobste.rs archive           | Invite-only tech community                              | ~100K             | High signal-to-noise             |

### 3.2 Pre-Filtering Pipeline (Extract-First)

**Key principle:** Extract text and metadata immediately after fetch using trafilatura. Store parsed content, not raw HTML. This trades replay flexibility for 5x storage savings—acceptable since extractor bugs affect <1% of documents.

```
RAW DUMP
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. URL PATTERN FILTER                   │
│    - Block known spam domains           │
│    - Block social media, e-commerce     │
│    - Block URL patterns (login, cart)   │
│    Speed: ~500K URLs/sec                │
└─────────────────────────────────────────┘
    │ (~50% pass)
    ▼
┌─────────────────────────────────────────┐
│ 2. FETCH CONTENT                        │
│    - Parallel HTTP (100 concurrent)     │
│    - 5s timeout, 3 retries             │
│    - Skip if already in atlas           │
│    Speed: ~100 URLs/sec (network-bound) │
└─────────────────────────────────────────┘
    │ (~70% pass - dead links filtered)
    ▼
┌─────────────────────────────────────────┐
│ 3. CONTENT EXTRACTION                   │
│    - trafilatura / readability          │
│    - Extract main content + metadata    │
│    - C4-style cleanup (lorem ipsum,     │
│      excessive JS, nav menu removal)    │
│    - Compute raw_html_hash (provenance) │
│    Speed: ~50 docs/sec                  │
└─────────────────────────────────────────┘
    │ (~80% pass - boilerplate-only pages filtered)
    ▼
┌─────────────────────────────────────────┐
│ 4. LANGUAGE FILTER                      │
│    - fasttext language detection        │
│    - Keep English only (for MVP)        │
│    Speed: ~100K docs/sec                │
└─────────────────────────────────────────┘
    │ (~80% pass)
    ▼
┌─────────────────────────────────────────┐
│ 5. LENGTH FILTER                        │
│    - Skip < 100 characters              │
│    - Skip > 100K characters (truncate)  │
│    Speed: ~1M docs/sec                  │
└─────────────────────────────────────────┘
    │ (~90% pass)
    ▼
FILTERED CANDIDATES (ready for embedding)
```

**Net effect:** From 1M raw URLs → ~250K candidates for embedding.

**Re-extraction fallback:** If trafilatura proves inadequate, re-crawl affected URLs (~1% of corpus) or sample from Wayback Machine for extractor experiments. See `--recrawl` and `--wayback-sample` options.

### 3.3 Farthest-Point Sampling with Lazy Greedy Optimization

**Goal:** Select k documents from n candidates such that selected documents are maximally spread across embedding space.

**Theoretical guarantee:** 2-approximation to optimal k-center coverage.

**Basic Algorithm (Gonzalez):**

```
FARTHEST_POINT_SAMPLING(candidates, k):
  
    INPUT:
        candidates: list of (id, embedding) pairs, n total
        k: number of documents to select
  
    OUTPUT:
        selected: list of k document IDs with maximum coverage
  
    ALGORITHM:
        1. selected ← {arbitrary candidate}  // or highest quality
        2. distances ← [∞] × n  // distance to nearest selected, for each candidate
  
        3. REPEAT k-1 times:
            a. FOR each unselected candidate i:
                   d ← distance(embedding[i], nearest_in_selected)
                   distances[i] ← min(distances[i], d)
  
            b. next ← argmax(distances)  // candidate farthest from any selected
            c. selected ← selected ∪ {next}
  
        4. RETURN selected
  
    COMPLEXITY: O(n × k) naive
```

**Lazy Greedy Optimization (Minoux's Algorithm):**

The naive algorithm recomputes all distances every iteration. Lazy Greedy exploits the fact that distances only decrease monotonically:

```
LAZY_GREEDY_FPS(candidates, k, quality_scores, α=1.0):
  
    INPUT:
        candidates: list of (id, embedding) pairs, n total
        k: number to select
        quality_scores: quality score for each candidate
        α: quality weight (0=pure coverage, 1=balanced)
  
    DATA STRUCTURES:
        upper_bounds[i]: upper bound on candidate i's selection score
        selected_index: HNSW index of selected embeddings (initially empty)
        priority_queue: max-heap ordered by upper_bounds
  
    ALGORITHM:
        1. // Initialize with quality-weighted bounds
           FOR each candidate i:
               upper_bounds[i] ← ∞ × quality_scores[i]^α
           priority_queue ← build_max_heap(upper_bounds)
  
        2. // Select first (highest quality)
           first ← argmax(quality_scores)
           selected_index.add(first)
           YIELD first
  
        3. WHILE |selected| < k:
  
            a. // Pop candidate with highest upper bound
               candidate ← priority_queue.pop_max()
  
            b. // Compute actual score (lazy evaluation)
               nearest, dist ← selected_index.search(candidate.embedding, k=1)
               actual_score ← dist × quality_scores[candidate]^α
  
            c. // If actual score matches upper bound, select it
               IF actual_score >= priority_queue.peek_max():
                   selected_index.add(candidate)
                   YIELD candidate
   
               ELSE:
                   // Update upper bound and re-insert
                   upper_bounds[candidate] ← actual_score
                   priority_queue.push(candidate, actual_score)
  
        4. RETURN selected
  
    COMPLEXITY: O(n log n) build + O(k × log n × log |selected|) amortized
                Typically 100-1000x faster than naive for large k
  
    KEY INSIGHT: Most candidates are evaluated once or twice, not k times.
                 Only candidates near the selection frontier need re-evaluation.
```

**Quality-Weighted Selection Criterion:**

```
score(c) = min_distance(c) × quality_score(c)^α

TRADEOFFS:
    α = 0.0: Pure coverage, ignores quality entirely
    α = 0.5: Mild quality preference
    α = 1.0: Balanced (recommended default)
    α = 2.0: Strong quality preference, may sacrifice coverage
```

**Quality score source for dumps:**

- If dump includes quality signals (Marginalia, HN score): append a boost in quality score
- Otherwise: compute lightweight quality proxy:
  - Content length (log-normalized)
  - Domain reputation (lookup table)
  - Structural signals (has headings, paragraphs)

### 3.4 HNSW-Accelerated Implementation

```
FARTHEST_POINT_SAMPLING_HNSW(candidates, k, batch_size=1000):
  
    INPUT:
        candidates: list of (id, embedding, quality_score) pairs
        k: number to select
        batch_size: candidates to evaluate per iteration
  
    DATA STRUCTURES:
        candidate_index: HNSW index of all candidate embeddings
        selected_index: HNSW index of selected embeddings (initially empty)
        candidate_set: set of unselected candidate IDs
        upper_bounds: lazy greedy upper bounds
  
    ALGORITHM:
        1. Build candidate_index from all embeddings  // O(n log n)
  
        2. first ← argmax(quality_score)  // start with highest quality
        3. selected_index.add(first)
        4. candidate_set.remove(first)
        5. Initialize upper_bounds with quality-weighted ∞
  
        6. WHILE |selected_index| < k AND candidate_set not empty:
  
            a. // Get top candidates by upper bound
               batch ← top_k_by_upper_bound(candidate_set, batch_size)
  
            b. // Batch query: find distance to nearest selected for each
               FOR each candidate in batch:
                   nearest, dist ← selected_index.search(candidate.embedding, k=1)
                   candidate.actual_score ← dist × quality_score^α
  
            c. // Find best in batch
               best ← argmax(batch, key=actual_score)
  
            d. // Lazy check: is best actually best globally?
               IF best.actual_score >= next_upper_bound_outside_batch:
                   selected_index.add(best)
                   candidate_set.remove(best)
               ELSE:
                   // Update bounds and continue
                   update_upper_bounds(batch)
  
        7. RETURN selected_index.get_all_ids()
  
    COMPLEXITY: O(n log n) build + O(k × batch_size × log |selected|) selection
                ≈ O(n log n) for typical parameters
```

### 3.5 Two-Stage Pipeline Architecture

**Breaking change from v1.0:** The pipeline has been refactored into two stages for improved throughput and flexibility.

**Stage 1 (Batch Ingestion):** High-throughput storage-only pipeline
**Stage 2 (Post-Processing):** Deferred scoring, filtering, and selection

```
STAGE 1: BATCH INGESTION (async + multiprocess)

    ARCHITECTURE:
        ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
        │   PRODUCER   │────►│  url_queue      │────►│ AsyncFetcher │
        │  (1 process) │     │                 │     │ (event loop) │
        └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                             │
                             ┌─────────────────┐            ▼
                             │  html_queue     │◄───────────┘
                             └────────┬────────┘
                                      │
                             ┌────────▼────────┐     ┌──────────────┐
                             │  ExtractPool    │────►│  text_queue  │
                             │ (ProcessPool)   │     │              │
                             └─────────────────┘     └──────┬───────┘
                                                             │
                             ┌─────────────────┐            ▼
                             │ ConcurrentEmbed │◄───────────┘
                             │ (GPU batching)  │
                             └────────┬────────┘
                                      │
                             ┌────────▼────────┐     ┌──────────────┐
                             │  embed_queue    │────►│    WRITER    │
                             │                 │     │ (store-only) │
                             └─────────────────┘     └──────┬───────┘
                                                             │
                                                             ▼
                                                  ┌──────────────────┐
                                                  │ SQLite + Qdrant  │
                                                  │ + document_texts │
                                                  └──────────────────┘

STAGE 2: POST-PROCESSING (deferred scoring)

    ┌──────────────────┐     ┌──────────────┐     ┌──────────────┐
    │ document_texts   │────►│PostProcessor │────►│   Update DB  │
    │ (unscored batch) │     │ Score+Dedup  │     │ quality_score│
    └──────────────────┘     │  +FPS (opt)  │     │ wilson_score │
                             └──────────────┘     └──────────────┘

STAGE 1 COMPONENTS:

    PRODUCER (single process):
        1. Read dump file sequentially
        2. Apply URL pattern filters (moved from worker)
        3. Check Bloom filter for already-processed URLs
        4. Push valid URLs to url_queue
        5. No decisions beyond filtering

    ASYNC FETCHER (event loop, aiohttp):
        1. Pull URL from url_queue
        2. Async HTTP fetch (concurrency: 100)
        3. Optional playwright for JavaScript-heavy sites (concurrency: 4)
        4. 15s timeout, exponential backoff
        5. Push (url, html) to html_queue

    EXTRACT POOL (ProcessPoolExecutor, 8 workers):
        1. Pull (url, html) from html_queue
        2. Extract text with trafilatura (CPU-bound)
        3. Compute raw_html_hash for provenance
        4. Language + length filters
        5. Push (url, text, metadata, hash) to text_queue
        6. Module-level _extract_worker_fn for pickling

    CONCURRENT EMBEDDER (GPU batching, batch_size=256):
        1. Pull batch of texts from text_queue
        2. Compute embeddings with bge-small on MPS/CUDA
        3. OOM protection: auto-reduce batch_size on errors
        4. Push (url, embedding, text, metadata, hash) to embed_queue

    WRITER (single process, store-only):
        1. Only process that writes to databases
        2. Pull from embed_queue
        3. Generate document ID: SHA256(canonical_url)[:16]
        4. Write to documents (quality_profile_used='pending')
        5. Write to document_texts (for deferred scoring)
        6. Write vector to Qdrant
        7. Archive parsed content to compressed JSONL
        8. No scoring, no FPS, no dedup at this stage
        9. No locks needed: single-threaded entry point

STAGE 2 COMPONENTS:

    POSTPROCESSOR (batch processing):
        1. Query unscored documents (quality_profile_used='pending')
        2. Load full text from document_texts table
        3. Detect content type (scientific, code, essay, etc.)
        4. Compute quality scores with calibrated weights
        5. Compute Wilson scores (sample-size-aware confidence)
        6. Run SimHash deduplication
        7. Optional: Run FPS selection
        8. Update documents table with scores and status
        9. Process in batches (default: 1000 docs/batch)
  
TWO-STAGE BENEFITS:

    Stage 1 (Ingestion):
        - **10x writer throughput:** No scoring bottleneck (~1200 docs/sec vs ~500)
        - **Async I/O:** aiohttp saturates network, playwright handles JS sites
        - **Multiprocess extraction:** Saturates all CPU cores
        - **Fault tolerance:** Text preserved even if scoring crashes
        - **No locks:** Single-writer architecture, no coordination overhead

    Stage 2 (Post-Processing):
        - **Re-scorable:** Update weights without re-crawling
        - **A/B testable:** Compare scoring algorithms on same corpus
        - **Debuggable:** Clear separation, easier to isolate scoring bugs
        - **Flexible:** Run FPS optionally, adjust quality thresholds

    Overall:
        - No mutexes, no race conditions, no distributed coordination
        - Natural backpressure via queue sizes
        - Crash recovery: resume from queue state
        - Simpler debugging: stages isolated

QUEUE SIZING:
    url_queue: 10,000 items (memory ~1MB)
    html_queue: 5,000 items (memory ~500MB raw HTML)
    text_queue: 5,000 items (memory ~50MB extracted text)
    embed_queue: 5,000 items (memory ~50MB vectors + metadata)

CONFIGURATION (config.json):
    "batch_processing": {
        "fetch_concurrency": 100,         // Async HTTP concurrency
        "fetch_timeout": 15,              // Seconds
        "use_playwright": false,          // Enable for JS sites
        "playwright_concurrency": 4,      // Browser pool size
        "extract_processes": 8,           // CPU cores for extraction
        "skip_scoring": true,             // Must be true for new pipeline
        "skip_url_filter": true,          // Filter in producer instead
        "url_queue_size": 10000,
        "embed_queue_size": 5000
    }
```

**Implementation (Stage 1 - Ingestion):**

```
RUN_BATCH_INGESTION(dump_path):
    # Initialize queues
    url_queue ← bounded_queue(max=10000)
    html_queue ← bounded_queue(max=5000)
    text_queue ← bounded_queue(max=5000)
    embed_queue ← bounded_queue(max=5000)

    # Start producer
    PRODUCER:
        FOR url IN read_dump(dump_path):
            IF passes_filters(url) AND NOT in_bloom_filter(url):
                url_queue.put(url)

    # Start async fetcher (event loop)
    ASYNC FETCHER:
        WHILE NOT shutdown:
            url ← url_queue.get()
            html ← await aiohttp.get(url, timeout=15)
            IF html IS NULL: CONTINUE
            html_queue.put((url, html))

    # Start extract pool (ProcessPoolExecutor)
    EXTRACT POOL (8 processes):
        WHILE NOT shutdown:
            (url, html) ← html_queue.get()
            raw_html_hash ← sha256(html)
            text, metadata ← trafilatura.extract(html)
            IF len(text) < 100: CONTINUE
            text_queue.put((url, text, metadata, raw_html_hash))

    # Start embedder (GPU batching)
    CONCURRENT EMBEDDER:
        batch ← []
        WHILE NOT shutdown OR text_queue.not_empty:
            WHILE len(batch) < 256 AND text_queue.not_empty:
                batch.append(text_queue.get())
            embeddings ← model.encode([item.text for item in batch])
            FOR item, embedding IN zip(batch, embeddings):
                embed_queue.put((item.url, embedding, item.text,
                                item.metadata, item.hash))
            batch ← []

    # Start writer (single-threaded)
    WRITER:
        WHILE NOT shutdown OR embed_queue.not_empty:
            (url, embedding, text, metadata, hash) ← embed_queue.get()
            doc_id ← sha256(url)[:16]

            # Store document (quality_profile_used='pending')
            documents.insert(doc_id, url, metadata, quality_score=0.0)
            document_texts.insert(doc_id, text)
            qdrant.upsert(doc_id, embedding)
            archive.write(url, text, metadata, hash)
```

**CLI Usage:**

```bash
# Stage 1: Batch ingestion
python -m phase1_offline.pipeline --dump dataset/sample.tar
# Output: Documents with quality_profile_used='pending'

# Stage 2: Post-processing
python scripts/post_process.py
python scripts/post_process.py --batch-size 500 --quality-threshold 0.2
# Output: Documents scored, deduplicated, optionally FPS-selected
```

### 3.6 Speed Optimizations for Batch Processing

**Embedding computation (primary bottleneck):**

| Optimization                 | Impact                                      |
| ---------------------------- | ------------------------------------------- |
| Batch size 256-512           | Saturates GPU/Neural Engine                 |
| bge-small-en-v1.5 (384-dim)  | Fast + good quality                         |
| MPS backend on M4            | ~1000 embeddings/sec                        |
| Quantize immediately to int8 | 4x memory reduction, enables larger batches |

**Embedding Model Strategy:**

Stick with `bge-small-en-v1.5` for the navigation/sampling layer. This model is:

- Well-tested and stable
- Fast enough for batch processing
- What the HNSW index is built around

If better semantic understanding is needed later, add a **reranker** (e.g., `bge-reranker-base`) that runs only on small result sets (top-100), rather than changing the base embedding model. Rerankers don't affect the index structure.

**Parallelization strategy (producer-consumer):**

```
Producer (read dump)     ████
url_queue fills          ════════════════════════════════════════
Workers (fetch+embed)    ████████████████████████████████████████
embed_queue fills        ════════════════════════════════════════
Writer (FPS+score+write) ████████████████████████████████████████

- Producer is fast, fills queue early
- Workers are I/O bound, benefit from high concurrency (12-16 threads)
- Writer is single-threaded but fast (FPS is O(log n) per document)
- Queues provide natural backpressure
- No coordination overhead between workers
```

## 5. Data Models

### 5.1 Core Entities

#### 5.1.1 QueuedURL

Represents a URL discovered but not yet processed.

| Field               | Type      | Constraints                                      | Description                           |
| ------------------- | --------- | ------------------------------------------------ | ------------------------------------- |
| canonical_url       | TEXT      | PRIMARY KEY                                      | Normalized URL                        |
| original_url        | TEXT      | NOT NULL                                         | URL as discovered                     |
| status              | ENUM      | pending, processing, completed, failed, rejected | Processing state                      |
| source_phase        | ENUM      | batch_dump, active_exploration                   | Which phase discovered this           |
| source              | TEXT      | NOT NULL                                         | Source identifier (dump name, gap ID) |
| source_metadata     | JSON      |                                                  | Source-specific data                  |
| priority            | REAL      | DEFAULT 0.5                                      | Processing priority (0-1)             |
| discovered_at       | TIMESTAMP | NOT NULL                                         | When URL entered queue                |
| claimed_by          | TEXT      |                                                  | Worker ID if processing               |
| claimed_at          | TIMESTAMP |                                                  | When processing started               |
| completed_at        | TIMESTAMP |                                                  | When processing finished              |
| retry_count         | INTEGER   | DEFAULT 0                                        | Number of failed attempts             |
| error_message       | TEXT      |                                                  | Last error if failed                  |
| parsed_archive_path | TEXT      |                                                  | Path to stored parsed content         |

#### 5.1.2 Document

Represents a processed, accepted document in the atlas.

| Field                 | Type      | Constraints      | Description                                                            |
| --------------------- | --------- | ---------------- | ---------------------------------------------------------------------- |
| id                    | TEXT      | PRIMARY KEY      | SHA256(canonical_url)[:16]                                             |
| canonical_url         | TEXT      | UNIQUE, NOT NULL | Normalized URL                                                         |
| original_urls         | JSON      |                  | Array of all URLs that resolved to this                                |
| title                 | TEXT      |                  | Extracted title                                                        |
| summary               | TEXT      |                  | Generated/extracted summary                                            |
| content_hash          | TEXT      | NOT NULL         | SimHash of content (64-bit hex)                                        |
| minhash_signature     | BLOB      |                  | MinHash signature for near-duplicate LSH (128 hashes)                  |
| content_length        | INTEGER   |                  | Character count                                                        |
| language              | TEXT      |                  | ISO 639-1 code                                                         |
| author                | TEXT      |                  | Extracted author                                                       |
| publication_date      | DATE      |                  | Extracted or inferred date                                             |
| domain                | TEXT      | NOT NULL         | Extracted from URL                                                     |
| source_phase          | ENUM      |                  | batch_dump, active_exploration                                         |
| source_dump           | TEXT      |                  | If batch: which dump                                                   |
| source_gap_id         | TEXT      |                  | If exploration: which gap triggered discovery                          |
| source_gap_type       | TEXT      |                  | If exploration: "topic" or "viewpoint"                                 |
| source_query          | TEXT      |                  | If exploration: the query that found this document                     |
| detected_content_type | ENUM      |                  | scientific, technical_code, personal_essay, news, documentation, other |
| quality_score         | REAL      | 0-1              | Current active quality score                                           |
| quality_score_version | INTEGER   | DEFAULT 1        | Version of scoring algorithm used                                      |
| quality_components    | JSON      |                  | Breakdown of quality signals (enables recomputation)                   |
| quality_profile_used  | TEXT      |                  | Which scoring profile was applied                                      |
| wilson_score          | REAL      | 0-1              | Sample-size-aware quality confidence (see Section 6.3.2)               |
| cluster_id            | INTEGER   |                  | HDBSCAN cluster assignment (-1 = orphaned)                             |
| importance_score      | REAL      | 0-1              | Domain authority score for Z-axis                                      |
| epistemic_profile     | JSON      |                  | claim_type, stance, quality breakdown                                  |
| created_at            | TIMESTAMP | NOT NULL         | When added to atlas                                                    |
| updated_at            | TIMESTAMP |                  | Last modification                                                      |

#### 5.1.3 DetectedGap (ignore, for the later active phase)

Tracks gaps identified during exploration.

#### 5.1.4 ExplorationProvenance (ignore, for the later active phase)

#### ~~5.1.5 GoldenSetEntry~~

~~Manually curated examples for weight calibration.~~

#### 5.1.6 ClusterStats

Tracks HDBSCAN cluster characteristics over time.

| Field           | Type      | Constraints | Description                           |
| --------------- | --------- | ----------- | ------------------------------------- |
| id              | INTEGER   | PRIMARY KEY | Auto-increment                        |
| cluster_id      | INTEGER   | NOT NULL    | HDBSCAN cluster label (-1 = orphaned) |
| computed_at     | TIMESTAMP | NOT NULL    | When stats were computed              |
| doc_count       | INTEGER   |             | Number of documents in cluster        |
| avg_quality     | REAL      |             | Mean quality score                    |
| quality_std     | REAL      |             | Quality standard deviation            |
| is_content_farm | BOOLEAN   |             | Flagged as potential content farm     |
| is_authority    | BOOLEAN   |             | Flagged as authoritative cluster      |
| action_taken    | TEXT      |             | What action was taken (if any)        |

## 6. Module Specifications

### 6.1 Batch Sampler Module

**Responsibility:** Process dumps using producer-consumer pipeline with farthest-point sampling for maximum coverage.

**Interface:**

```
CLI: ./batch-sample [OPTIONS]

Options:
  --dump PATH             Path to dump file (JSON/JSONL/CSV)
  --dump-type TYPE        Type of dump (marginalia, dmoz, pinboard, hn, custom)
  --budget N              Maximum documents to select from this dump
  --quality-weight FLOAT  α parameter for quality weighting (default: 1.0)
  --workers N             Number of worker threads (default: 12)
  --dry-run               Analyze dump without processing
  --resume                Resume interrupted job
  --recrawl PATH          Re-crawl specific URLs from file (for extraction fixes)
  --wayback-sample N      Sample N URLs from Wayback Machine (for extractor experiments)
  --extractor NAME        Content extractor to use (default: trafilatura)

Environment:
  ATLAS_DB_PATH           Path to atlas databases
  QDRANT_URL              Qdrant server URL
  PARSED_ARCHIVE_PATH     Path for parsed content storage
```

**Processing Stages:**

```
1. PRODUCER: LOAD AND PRE-FILTER
   - Parse dump file
   - Apply URL pattern filters
   - Check Bloom filter for processed URLs
   - Push to url_queue

2. WORKERS: FETCH, EXTRACT, EMBED (parallel, N threads)
   - Fetch content (I/O bound)
   - Extract main content immediately (trafilatura + fallbacks)
   - Compute raw_html_hash for provenance
   - Language filter (English only)
   - Length filter (100-100K chars)
   - Compute embedding
   - Push to embed_queue

3. WRITER: SAMPLE, SCORE, ACCEPT (single-threaded, no locks)
   - Lazy greedy FPS selection
   - Novelty check against global index
   - Compute quality score using calibrated weights
   - Compute Wilson score for confidence-aware ranking
   - If accepted: add to atlas, archive parsed content
```

### 6.3 Curator Module

**Responsibility:** Score documents using Golden Set calibrated weights, detect duplicates, accept or reject.

**Interface:**

```
CLI: ./curate [OPTIONS]

Options:
  --pending               Process documents with status='pending_curation'
  --recompute-scores      Recompute scores for all documents (new version)
  --recompute-wilson      Recompute Wilson scores after signal updates
  --score-version N       Target score version for recomputation
  --document-id ID        Process specific document
  --verbose               Print detailed scoring breakdown

Environment:
  ATLAS_DB_PATH           Path to atlas databases
  QDRANT_URL              Qdrant server URL
```

**Content Type Detection:**

| Content Type   | Domain Patterns                      | Content Signals                              |
| -------------- | ------------------------------------ | -------------------------------------------- |
| scientific     | arxiv.org, *.edu, pubmed, nature.com | "abstract", "methodology", "references", DOI |
| technical_code | github.com, *.dev, stackoverflow.com | code blocks, "function", "class", "import"   |
| personal_essay | *.substack.com, medium.com/@*      | first-person, "my experience", "personally"  |
| news           | known news domains                   | "reported", "according to", "sources say"    |
| documentation  | docs.*, *.readthedocs.io             | "## Installation", "API", "Parameters"       |
| other          | fallback                             | —                                           |

#### ~~6.3.1 Golden Set Weight Calibration with Topic-Cluster Cross-Validation~~

~~**Do not guess scoring weights.** Calibrate them empirically using a Golden Set.~~

~~**Critical insight from cartographic validation:** Standard random cross-validation overfits by 28-40% due to topical autocorrelation—pages from the same domain or topic cluster aren't independent. Use topic-cluster CV instead.~~

#### ~~6.3.2 Wilson Score for Hidden Gem Ranking~~

~~**Problem:** Raw quality scores don't account for signal reliability. A page with 3 quality signals all positive (3/3 = 100%) should rank lower than a page with 50/55 positive signals (91%), because we have more confidence in the second estimate.~~

~~**Solution:** Wilson score confidence interval gives a sample-size-aware lower bound on quality.~~

~~**Storage:** Wilson score stored in `documents.wilson_score` field, recomputed when quality components change.~~

#### 6.3.3 Importance Scoring (Z-axis for visualization)

Compute importance score from: domain_authority (0.5 weight), inbound_links (0.3), academic_citations (0.2). Provides Z-axis in Atlas visualization.

#### 6.3.4 Deduplication (SimHash + MinHash)

| Method          | Threshold      | Purpose                                |
| --------------- | -------------- | -------------------------------------- |
| SimHash         | Hamming ≤ 3   | Exact duplicates                       |
| MinHash LSH     | Jaccard ≥ 0.5 | Near-duplicates (128 hashes, 32 bands) |
| Embedding       | Cosine ≥ 0.95 | Semantic duplicates                    |
| Content overlap | ≥ 0.80        | Fallback                               |

#### 6.3.5 Domain-Adaptive Scoring Profiles

Weights calibrated via Golden Set with topic-cluster CV (200 exemplary + 200 garbage per type).

| Content Type   | Quality Thresh | Top Weights (calibrated)                                                        | Validation QADI |
| -------------- | -------------- | ------------------------------------------------------------------------------- | --------------- |
| scientific     | 0.50           | citation_quality (0.28), methodology (0.24), peer_review (0.22)                 | 0.18            |
| technical_code | 0.45           | code_quality (0.27), recency (0.23), completeness (0.21)                        | 0.21            |
| personal_essay | 0.40           | writing_quality (0.32), specificity (0.26), authenticity (0.18)                 | 0.24            |
| news           | 0.50           | source_attribution (0.31), multiple_perspectives (0.24), factual_density (0.21) | 0.19            |
| documentation  | 0.45           | completeness (0.32), accuracy (0.26), recency (0.18)                            | 0.22            |
| default        | 0.45           | content_depth (0.22), source_signals (0.20), citation_quality (0.18)            | 0.25            |

**~~Score Versioning:~~**

~~When scoring weights change (after recalibration), increment `quality_score_version` and recompute:~~

#### ~~6.3.6 Epistemic Classification~~

### 6.4 Mapper Module

**Responsibility:** Compute visualization coordinates, track diversity metrics. Use cosine similarity for all computation; reserve hyperbolic projection for display only.

**Mapping Configuration:**

| Component             | Method               | Key Parameters                                                 |
| --------------------- | -------------------- | -------------------------------------------------------------- |
| Semantic (XY)         | UMAP                 | neighbors=15, min_dist=0.1, metric=cosine, sample_for_fit=100K |
| Importance (Z)        | domain authority     | source=importance_score from curator                           |
| Topic clusters        | HDBSCAN              | min_cluster_size=50, cluster_selection_method=eom              |
| Hyperbolic (viz only) | Poincaré projection | For display only, not indexing                                 |

---

## 9. Development Phases

### Phase 1: Foundation (Week 1)

- Directory structure and configuration
- URL canonicalization with tests
- SimHash + MinHash computation with tests
- SQLite schema (including Golden Set, Wilson score, cluster fields)
- Qdrant connection with quantization
- T9 SSD mount and tiered storage
- Parsed content archive infrastructure

### Phase 2: Batch Pipeline (Week 2)

- **Producer-consumer pipeline** (no multi-agent)
- Pre-filtering in producer
- Worker threads for fetch/extract/embed
- Single-threaded writer for FPS/score/accept
- **Content extraction with trafilatura** (extract-first, no HTML storage)
- bge-small embedding computation
- **Lazy greedy farthest-point sampling**
- Process first dump (Marginalia blogs)

### Phase 3: Scoring and Curation (Week 3)

- Content type detection
- **Build Golden Set (200 exemplary + 200 garbage per type)**
- **Topic-cluster CV for calibration** (prevents 28-40% overfit)
- **Calibrate weights via logistic regression**
- **QADI metrics for validation** (quantity vs allocation diagnosis)
- **Wilson score computation** (sample-size-aware ranking)
- Domain-adaptive quality scoring with calibrated weights
- **Importance scoring (domain authority)**
- Epistemic classification
- **MinHash LSH deduplication**
- Full curation pipeline
- Basic visualizer (list view with filters)

### Phase 4: Active Exploration (Week 4)

- **Lonely node gap detection**
- **HDBSCAN clustering** (quality topology, content farm detection)
- Query inversion (keyword + LLM methods)
- Search API integration (Marginalia, Kagi)
- **Exploration provenance tracking**
- Exploration cycle implementation
- Exploration effectiveness tracking

### Phase 5: Mapping and Visualization (Week 5)

- Semantic mapping (UMAP, **cosine metric**)
- **Hyperbolic mapping (visualization only)**
- **Importance Z-axis**
- Full visualizer (map view, filters, search)
- Diversity overlays

### Phase 6: Monitoring and Hardening (Week 6+)

- **Coverage Gini tracking** (topic and domain equity)
- **Cluster health monitoring**
- **Drift detection** (QADI comparison over time)
- Error handling and recovery
- Content extraction fallback testing
- Monitoring and metrics dashboard
- Backup and restore
- Documentation
- First verification cycle
- **Golden Set calibration validation**
- Scale testing (10M documents)

---

## Appendix A: Configuration Reference

**Paths:** `data_dir`, `logs_dir`, `dumps_dir`, `archive_dir`, `parsed_archive_dir` → /Volumes/T9/atlas/...

**Databases:** url_queue, documents, graph, events, exploration, parsed_content, golden_set, coverage_metrics, cluster_stats (all SQLite in data_dir)

**Qdrant:** url=localhost:6333, collection=atlas_embeddings, quantization=scalar, on_disk=true

**Embedding:** model=bge-small-en-v1.5, device=mps, batch_size=256, max_tokens=512

**LLM:** provider=openai, model=gpt-4o-mini, temperature=0.7

**Batch Processing:** workers=12, quality_weight_alpha=1.0, novelty_threshold=0.08, queues=(url:10K, embed:5K), lazy_greedy=true

**Content Extraction:** extractor=trafilatura, fallback=readability, store_raw_html_hash=true

**Exploration:** topic_gaps(lonely_nodes, samples=1000, threshold=0.20, gaps=10), viewpoint_gaps(5, diversity=0.3), hdbscan(min_cluster_size=50), query_synthesis(llm, 5/gap), cycle=6h

**Calibration:** golden_set_path, current_version=1, holdout=0.2, cv_method=topic_cluster, validation_metric=qadi

**Scoring:** wilson_z=1.96, min_signals_for_ranking=3

**Deduplication:** simhash≤3, minhash(128 hashes, 32 bands, Jaccard≥0.5), embedding≥0.95

**Mapping:** UMAP(neighbors=15, min_dist=0.1, cosine), hyperbolic(display-only, depth=6)

**Monitoring:** gini_alert_threshold=0.6, orphan_review_threshold=100, drift_alert_delta=0.05

---

## Appendix B: Command Reference

| Command            | Key Options                                                                                                                     | Purpose                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `./batch-sample` | `--dump`, `--dump-type`, `--budget`, `--workers`, `--resume`, `--recrawl`, `--wayback-sample`                     | Process dumps, re-extract   |
| `./explore`      | `--detect-topic-gaps`, `--detect-viewpoint-gaps`, `--detect-clusters`, `--generate-queries`, `--execute`, `--cycle` | Gap detection & exploration |
| `./calibrate`    | `--add-exemplary`, `--add-garbage`, `--compute-metrics`, `--train-weights`, `--validate`, `--qadi-report`           | Golden set management       |
| `./curate`       | `--pending`, `--apply-weights`, `--recompute-scores`, `--recompute-wilson`, `--document-id`                           | Document scoring            |
| `./map`          | `--type`, `--recompute`, `--incremental`                                                                                  | Generate mappings           |
| `./visualize`    | `--port`, `--host`, `--static-export`                                                                                     | Serve web UI                |
| `./orchestrate`  | `--batch-only`, `--explore-only`, `--continuous`                                                                          | Coordinate pipeline         |
| `./stats`        | `--summary`, `--exploration-report`, `--calibration-drift`, `--qadi-history`                                            | Analytics                   |
| `./monitor`      | `--compute-coverage`, `--compute-clusters`, `--drift-report`, `--full-report`                                           | Health monitoring           |
| `./archive`      | `--compact`, `--verify`, `--stats`                                                                                        | Archive management          |
| `./snapshot`     | `--output`                                                                                                                    | Backup atlas                |
| `./verify`       | `--sample`, `--output`, `--compare-previous`, `--add-to-golden`                                                         | Quality verification        |

---

## Appendix C: Revision History

| Version | Date     | Changes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0.3.0   | Jan 2025 | Initial draft                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 0.4.0   | Jan 2025 | Lake Pattern (HTML archival), HNSW strain gap detection, Lazy greedy FPS, MinHash deduplication, Importance scoring, Score versioning, Hyperbolic clarification, Provenance tracking                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 0.5.0   | Jan 2025 | Producer-consumer pipeline (replaced multi-agent), Lonely node gap detection (replaced HNSW strain), Archive retention policy (quality-gated), Golden Set calibration (replaced manual weights)                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 0.6.0   | Jan 2025 | Extract-first storage (replaced raw HTML archival with parsed content archive via trafilatura), 5x storage reduction, raw_html_hash for provenance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 0.7.0   | Jan 2025 | Replaced code snippets with concise pseudocode, condensed YAML configs to tables                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 0.8.0   | Jan 2025 | **Cartographic techniques integration:** Topic-cluster cross-validation (prevents 28-40% overfit), QADI metrics (quantity vs allocation diagnosis), Wilson score ranking (sample-size-aware confidence), HDBSCAN clustering (quality topology, content farm detection, orphaned gem discovery), Coverage Gini coefficient (equity tracking), Monitor module, updated data models and CLI commands                                                                                                                                                                                                                                                      |
| 1.0.0   | Jan 2026 | **MVP Implementation:** Complete batch pipeline (producer-consumer), UMAP+HDBSCAN mapping, interactive web visualizer, structured logging, modular scripts. See Appendix D for details.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 1.1.0   | Feb 2026 | **Architecture Refactoring:** Two-stage pipeline (ingestion + post-processing), Repository pattern for data access, Protocol-based scoring package (modular metrics), Protocol-based mapping package (swappable components), AsyncFetcher (aiohttp + playwright), ExtractPool (multiprocess extraction), ConcurrentEmbedder (GPU batching with OOM protection), document_texts table (deferred scoring), QdrantManager abstraction, Domain models with type safety, 10x writer throughput improvement                                                                                                                                                  |
| 1.1.1   | Feb 2026 | **Refactoring:** Resume-safe Phase 1 (DB-backed URL dedup, resume flags), removed mock fetch/embedding paths, persisted source_dump metadata, added explicit existing-DB safety logging                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 1.2.0   | Feb 2026 | **Meilisearch Full-Text Search:** Added Meilisearch as discrete service for full-text document search. MeilisearchManager (lazy-init, graceful degradation). Writer pipeline indexes metadata to Meilisearch alongside Qdrant embeddings. BatchWriter buffers Meilisearch writes. Search API endpoint (`/api/search/meili`). Frontend search bar overlaid on map (semi-transparent, left side). Search results dropdown with click-to-zoom. GPU-based dimming of non-matched points via ScatterplotLayer alpha channel (no point removal). Sidebar search results limit control. Backfill script (`scripts/meili_sync.py`) for existing documents. |
| 1.2.1   | Feb 2026 | **Refactoring:** Enriched mapping metadata by doc_id, added metadata columns to Parquet mapping exports, and removed placeholder score defaults to surface missing data issues early.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 1.2.2   | Feb 2026 | **Writer Commit Fix + Text Storage:** Fixed missing `conn.commit()` in BatchWriter/Writer causing `document_texts` table to remain empty (0 rows for 38k docs). Removed `summary` computation from ingestion (leave empty; full text lives in `document_texts`). Added `scripts/backfill_texts.py` to populate `document_texts` from parsed archive for existing DB rows. PostProcessor scoring depends on `document_texts` for deferred scoring.                                                                                                                                                                                        |
| 1.2.3   | Feb 2026 | **Extraction Quality:** Switched trafilatura to `favor_recall=True`, `include_tables=True`, `deduplicate=True` for complete content extraction. Added `_clean_extracted_text()` post-processor to strip code blocks, CSS/JS artifacts, cookie banners, nav lists, ad CTAs, and non-alphabetic lines. Applied to both `_extract_worker_fn` (ProcessPoolExecutor path) and `ContentExtractor.extract` (BatchWorker path).                                                                                                                                                                                                                    |
| 1.2.4   | Feb 2026 | **Refactoring:** Added memory governor (18GB cap) that exits gracefully when exceeded. Qdrant writes moved to async batch worker to avoid writer stalls when Qdrant is unstable.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

---

## Appendix D: MVP Implementation Summary

This appendix documents the actual implementation delivered in the initial MVP release.

### Implemented Components

| Component                           | Location                             | Status   | Notes                                        |
| ----------------------------------- | ------------------------------------ | -------- | -------------------------------------------- |
| **Batch Ingestion (Stage 1)** | `phase1_offline/pipeline.py`       | Complete | Two-stage architecture, async + multiprocess |
| AsyncFetcher                        | `phase1_offline/worker.py`         | Complete | aiohttp + playwright for JS sites            |
| ExtractPool                         | `phase1_offline/worker.py`         | Complete | Multiprocess CPU-bound extraction            |
| ConcurrentEmbedder                  | `phase1_offline/worker.py`         | Complete | GPU batching with OOM protection             |
| Writer (Store-Only)                 | `phase1_offline/writer.py`         | Complete | No scoring, writes to document_texts         |
| **Post-Processing (Stage 2)** | `phase1_offline/post_processor.py` | Complete | Deferred scoring + dedup + FPS               |
| SLOP Dump Adapter                   | `phase1_offline/dump_adapters.py`  | Complete | Marginalia archive format                    |
| Content Extraction                  | `phase1_offline/worker.py`         | Complete | trafilatura + readability fallback           |
| **Quality Scoring Package**   | `curation/scoring/`                | Complete | Modular, protocol-based, extensible          |
| ScoringPipeline                     | `curation/scoring/pipeline.py`     | Complete | Orchestrates metrics + aggregation           |
| Individual Metrics                  | `curation/scoring/metrics/`        | Complete | Citation, depth, methodology, etc.           |
| Deduplication                       | `phase1_offline/deduplication.py`  | Complete | SimHash + MinHash                            |
| **Repository Layer**          | `common/repositories.py`           | Complete | DocumentRepository, MetricsRepository, etc.  |
| SQLite Storage                      | `common/database.py`               | Complete | Context manager, thread-safe                 |
| QdrantManager                       | `common/qdrant_manager.py`         | Complete | Unified Qdrant interface                     |
| Domain Models                       | `common/models.py`                 | Complete | Type-safe dataclasses                        |
| **UMAP Mapping Package**      | `mapping/`                         | Complete | Protocol-based, swappable components         |
| MappingPipeline                     | `mapping/pipeline.py`              | Complete | Orchestrates projection + clustering         |
| Web Visualizer                      | `visualizer/server.py`             | Complete | Interactive canvas with pan/zoom             |
| Health Monitor                      | `monitor/health.py`                | Complete | Gini, quality distribution, alerts           |
| Structured Logging                  | `common/logging/logger.py`         | Complete | JSON logs to `logs/`                       |

### Not Yet Implemented (Phase 2)

| Component              | Guide Section | Notes                          |
| ---------------------- | ------------- | ------------------------------ |
| Active Exploration     | §4           | Gap detection, query synthesis |
| Golden Set Calibration | §3.4         | Weight training from exemplars |
| FPS Sampling           | §3.3         | Quality-weighted selection     |
| Hyperbolic Projection  | §6.3.5       | Hierarchical visualization     |
| Wayback Fallback       | §3.1.4       | Dead link recovery             |

### Architecture Deviations (v1.1)

| Guide Specification               | Actual Implementation                         | Rationale                                 |
| --------------------------------- | --------------------------------------------- | ----------------------------------------- |
| Single-stage pipeline             | Two-stage (ingest + post-process)             | 10x writer throughput, re-scorable corpus |
| Scoring in writer                 | Deferred to PostProcessor                     | Separation of concerns, flexibility       |
| Thread-based workers              | Async fetcher + multiprocess extraction       | Better I/O saturation + CPU utilization   |
| CLI commands (`./batch-sample`) | Python scripts (`scripts/batch_process.py`) | Simpler development workflow              |
| YAML configs                      | JSON config (`config.json`)                 | Native Python support                     |
| Raw HTML archival                 | Extract-first (text only)                     | 5x storage reduction per guide v0.6       |
| Direct SQL queries                | Repository pattern                            | Type safety, testability, maintainability |
| Monolithic scoring                | Modular scoring package                       | Extensibility, protocol-based design      |

### Key Implementation Details (v1.1)

#### Two-Stage Architecture

```python
# Stage 1: Batch ingestion (scripts/batch_process.py)
- AsyncFetcher: aiohttp event loop, 100 concurrent connections
- ExtractPool: ProcessPoolExecutor with 8 workers
- ConcurrentEmbedder: GPU batching with OOM protection
- Writer: Store-only path, quality_profile_used='pending'

# Stage 2: Post-processing (scripts/post_process.py)
- PostProcessor: Batch scoring from document_texts table
- Deferred: Content type detection, quality scoring, Wilson score
- Optional: FPS selection, deduplication
- Updates: quality_score, detected_content_type, status
```

#### Async HTTP Fetching

```python
# AsyncFetcher in worker.py
- aiohttp.ClientSession for HTTP (100 concurrent)
- playwright.async_api for JS-heavy sites (4 concurrent)
- Exponential backoff on failures
- Graceful shutdown with queue draining
```

#### Multiprocess Extraction

```python
# ExtractPool in worker.py
- ProcessPoolExecutor saturates CPU cores
- Module-level _extract_worker_fn for pickling compatibility
- Trafilatura + readability fallback
- Per-process stats tracking
```

#### Repository Pattern

```python
# common/repositories.py
- DocumentRepository: Type-safe document operations
- DocumentTextRepository: Deferred scoring text storage
- MetricsRepository: Coverage metrics and monitoring
- Each method opens/closes own connection (thread-safe)
- Optional conn parameter for transactions
```

#### Protocol-Based Scoring

```python
# curation/scoring/
- ScoringMetric protocol: Extensible metric system
- ScoreAggregator protocol: Pluggable aggregation
- MetricRegistry: Runtime metric discovery
- Content-type-specific weight profiles
- Backward compatible via _compat.py facade
```

#### Thread-Safe Database Access

```python
# database.py
- Absolute path resolution (avoids CWD issues)
- Context manager for transactions with auto-commit/rollback
- Repository pattern prevents connection leaks
```

#### Interactive Visualizer

```
Features implemented:
- Canvas-based 2D rendering (not SVG/DOM)
- Scroll-to-zoom, drag-to-pan
- Intra-cluster connection lines
- Click-to-open website
- Rich tooltips with excerpt
- Cluster/quality filtering
```

### Configuration (`config.json`)

Key sections added beyond the guide:

```json
{
  "batch_processing": {
    "workers": 4,
    "worker_batch_size": 10,
    "url_queue_size": 1000,
    "embed_queue_size": 500
  },
  "mapping": {
    "umap": { "n_neighbors": 15, "min_dist": 0.1 },
    "hdbscan": { "min_cluster_size": 15 }
  },
  "visualizer": {
    "host": "0.0.0.0",
    "port": 8080
  }
}
```

### Module Documentation

Each module contains a `README.md` with:

- Architecture diagrams
- Key classes and functions
- Configuration options
- Usage examples

See individual directories for details.

---

## Appendix E: Cartographic Techniques Summary

This version integrates five key techniques from geospatial science:

| Technique          | Origin                             | Application               | Benefit                                                |
| ------------------ | ---------------------------------- | ------------------------- | ------------------------------------------------------ |
| Topic-cluster CV   | Spatial cross-validation           | Golden Set calibration    | Prevents 28-40% accuracy overestimate                  |
| QADI metrics       | ISO 19157 validation               | Verification diagnostics  | Separates quantity from allocation errors              |
| Wilson score       | Binomial confidence intervals      | Hidden gem ranking        | Sample-size-aware quality confidence                   |
| HDBSCAN clustering | Spatial clustering (DBSCAN family) | Quality topology analysis | Finds content farms, authority clusters, orphaned gems |
| Gini coefficient   | Geographic equity metrics          | Coverage monitoring       | Tracks topic/domain balance                            |

These techniques are battle-tested on global-scale mapping challenges (OpenStreetMap, humanitarian mapping) and translate directly to web content curation.
