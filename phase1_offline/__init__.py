"""
Phase 1 Offline Batch Processing Pipeline for Truth Atlas.

This package implements the batch dump processing phase, which:
1. Reads URLs from dump files (SLOP, JSONL, plain text)
2. Fetches content asynchronously (aiohttp) or uses pre-fetched HTML
3. Extracts text via multiprocess pool (trafilatura)
4. Computes embeddings with batch GPU processing
5. Stores to SQLite and Qdrant (scoring deferred)
6. Post-processes: quality scoring, FPS selection, deduplication

Components:
    - dump_adapters: Parsers for various dump formats
    - fps_sampler: Farthest Point Sampling with Lazy Greedy
    - deduplication: SimHash, MinHash, Bloom filter
    - producer: URL extraction and Bloom filter dedup
    - worker: AsyncFetcher, ExtractPool, ConcurrentEmbedder
    - writer: Store-only persistence (scoring deferred)
    - pipeline: Orchestration and CLI
    - post_processor: Standalone scoring/filtering/FPS/dedup

Usage:
    python -m phase1_offline.pipeline --dump dataset/sample-m.tar
    python scripts/post_process.py
"""

from phase1_offline.pipeline import BatchPipeline, MultiDumpPipeline
from phase1_offline.producer import Producer, MultiProducer
from phase1_offline.worker import (
    BatchWorker,
    WorkerPool,
    AsyncFetcher,
    ExtractPool,
    ConcurrentEmbedder,
)
from phase1_offline.writer import Writer, BatchWriter
from phase1_offline.dump_adapters import adapter_registry, DumpRecord
from phase1_offline.fps_sampler import FarthestPointSampler, StreamingFPSSampler
from phase1_offline.deduplication import (
    Deduplicator,
    BloomFilter,
    SimHash,
    MinHash,
    URLCanonicalizer
)
from phase1_offline.post_processor import PostProcessor

__all__ = [
    # Pipeline
    'BatchPipeline',
    'MultiDumpPipeline',

    # Producer
    'Producer',
    'MultiProducer',

    # Worker
    'BatchWorker',
    'WorkerPool',
    'AsyncFetcher',
    'ExtractPool',
    'ConcurrentEmbedder',

    # Writer
    'Writer',
    'BatchWriter',

    # Dump Adapters
    'adapter_registry',
    'DumpRecord',

    # FPS Sampling
    'FarthestPointSampler',
    'StreamingFPSSampler',

    # Deduplication
    'Deduplicator',
    'BloomFilter',
    'SimHash',
    'MinHash',
    'URLCanonicalizer',

    # Post-Processing
    'PostProcessor',
]
