"""
Phase 1 Offline Batch Processing Pipeline for Truth Atlas.

This package implements the batch dump processing phase, which:
1. Reads URLs from dump files (SLOP, JSONL, plain text)
2. Fetches and extracts content
3. Applies quality-weighted Farthest Point Sampling for coverage
4. Uses calibrated scoring from the golden set
5. Writes to SQLite and Qdrant

Components:
    - dump_adapters: Parsers for various dump formats
    - fps_sampler: Farthest Point Sampling with Lazy Greedy
    - deduplication: SimHash, MinHash, Bloom filter
    - producer: URL extraction and filtering
    - worker: Content fetching, extraction, embedding
    - writer: Quality scoring, FPS selection, persistence
    - pipeline: Orchestration and CLI

Usage:
    python -m phase1_offline.pipeline --dump dataset/sample-m.tar --workers 8
"""

from phase1_offline.pipeline import BatchPipeline, MultiDumpPipeline
from phase1_offline.producer import Producer, MultiProducer
from phase1_offline.worker import BatchWorker, WorkerPool
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
]
