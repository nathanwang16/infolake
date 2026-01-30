"""
Phase 1 Batch Processing Pipeline for Truth Atlas.

Orchestrates the producer-consumer pipeline for processing dump files:
    Producer → url_queue → Workers → embed_queue → Writer → Atlas

Usage:
    python -m phase1_offline.pipeline --dump dataset/sample-m.tar --workers 8
    python -m phase1_offline.pipeline --dump dataset/sample-m.tar --workers 8 --limit 10000
    python -m phase1_offline.pipeline --dumps dataset/*.tar --workers 12

Features:
- Multi-format dump support (SLOP, JSONL, plain text)
- Quality-weighted Farthest Point Sampling
- Calibrated quality scoring
- Comprehensive statistics and progress tracking
- Graceful shutdown handling
"""

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, Any, List, Optional

from src.logging.logger import get_logger, setup_logger
from common.config import config
from common.database import db
from phase1_offline.producer import Producer, MultiProducer
from phase1_offline.worker import BatchWorker, WorkerPool
from phase1_offline.writer import Writer, BatchWriter

logger = get_logger("pipeline")


@dataclass
class PipelineStats:
    """Aggregated pipeline statistics."""
    start_time: float
    end_time: Optional[float] = None
    producer_stats: Optional[Dict[str, Any]] = None
    worker_stats: Optional[Dict[str, Any]] = None
    writer_stats: Optional[Dict[str, Any]] = None
    
    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def duration_human(self) -> str:
        secs = int(self.duration_seconds)
        hours, remainder = divmod(secs, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @property
    def throughput(self) -> float:
        """Documents per second."""
        if self.writer_stats and self.duration_seconds > 0:
            return self.writer_stats.get('accepted', 0) / self.duration_seconds
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'duration_seconds': self.duration_seconds,
            'duration_human': self.duration_human,
            'throughput_docs_per_sec': round(self.throughput, 2),
            'producer': self.producer_stats,
            'workers': self.worker_stats,
            'writer': self.writer_stats,
        }


class ProgressMonitor(threading.Thread):
    """
    Background thread for progress reporting.
    
    Periodically logs pipeline statistics and queue sizes.
    """
    
    def __init__(
        self,
        url_queue: Queue,
        embed_queue: Queue,
        worker_pool: Optional[WorkerPool],
        writer: Optional[Writer],
        interval: int = 30
    ):
        super().__init__(name="ProgressMonitor", daemon=True)
        self.url_queue = url_queue
        self.embed_queue = embed_queue
        self.worker_pool = worker_pool
        self.writer = writer
        self.interval = interval
        self.running = True
        self._start_time = time.time()
    
    def run(self):
        """Main monitoring loop."""
        while self.running:
            time.sleep(self.interval)
            if not self.running:
                break
            self._report_progress()
    
    def _report_progress(self):
        """Logs current progress."""
        elapsed = time.time() - self._start_time
        
        url_qsize = self.url_queue.qsize()
        embed_qsize = self.embed_queue.qsize()
        
        worker_stats = self.worker_pool.get_stats() if self.worker_pool else {}
        writer_stats = self.writer.get_stats() if self.writer else {}
        
        processed = worker_stats.get('processed', 0)
        accepted = writer_stats.get('accepted', 0)
        
        throughput = accepted / max(elapsed, 1)
        
        logger.info(
            f"Progress: {elapsed:.0f}s | "
            f"url_q={url_qsize} | embed_q={embed_qsize} | "
            f"processed={processed} | accepted={accepted} | "
            f"throughput={throughput:.1f} docs/s"
        )
    
    def stop(self):
        """Stops the monitor."""
        self.running = False


class BatchPipeline:
    """
    Main pipeline orchestrator.
    
    Manages the producer-consumer pipeline lifecycle:
    1. Initialize queues and components
    2. Start writer (consumer of results)
    3. Start workers (producers of results)
    4. Start producer (producer of URLs)
    5. Wait for completion or handle shutdown
    """
    
    def __init__(
        self,
        dump_path: str,
        num_workers: Optional[int] = None,
        limit: int = 0,
        url_queue_size: Optional[int] = None,
        embed_queue_size: Optional[int] = None
    ):
        if dump_path is None:
            raise ValueError("dump_path is required")
        
        self.dump_path = Path(dump_path)
        self.num_workers = num_workers or config.get("batch_processing.workers", 4)
        self.limit = limit
        
        # Queue configuration
        self.url_queue_size = url_queue_size or config.get("batch_processing.url_queue_size", 10000)
        self.embed_queue_size = embed_queue_size or config.get("batch_processing.embed_queue_size", 5000)
        
        # Components (initialized in setup)
        self.url_queue: Optional[Queue] = None
        self.embed_queue: Optional[Queue] = None
        self.producer: Optional[Producer] = None
        self.worker_pool: Optional[WorkerPool] = None
        self.writer: Optional[Writer] = None
        self.monitor: Optional[ProgressMonitor] = None
        
        # State
        self._shutdown_requested = False
        self._stats = PipelineStats(start_time=time.time())
        self._writer_thread = None
        
        logger.info(f"Pipeline initialized: dump={dump_path}, workers={self.num_workers}, limit={limit}")
    
    def setup(self):
        """Initializes all pipeline components."""
        logger.info("Setting up pipeline components...")
        
        # Create queues
        self.url_queue = Queue(maxsize=self.url_queue_size)
        self.embed_queue = Queue(maxsize=self.embed_queue_size)
        
        # Create producer
        self.producer = Producer(
            url_queue=self.url_queue,
            dump_path=str(self.dump_path),
            limit=self.limit
        )
        
        # Create worker pool
        self.worker_pool = WorkerPool(
            url_queue=self.url_queue,
            embed_queue=self.embed_queue,
            num_workers=self.num_workers
        )
        
        # Create writer
        self.writer = BatchWriter(embed_queue=self.embed_queue)
        
        # Create progress monitor
        self.monitor = ProgressMonitor(
            url_queue=self.url_queue,
            embed_queue=self.embed_queue,
            worker_pool=self.worker_pool,
            writer=self.writer
        )
        
        logger.info("Pipeline components initialized")
    
    def run(self) -> PipelineStats:
        """
        Runs the complete pipeline.
        
        Returns:
            PipelineStats with aggregated statistics
        """
        self.setup()
        self._install_signal_handlers()
        
        logger.info("Starting pipeline...")
        
        try:
            # Start writer first (consumer)
            # NOT daemon - we need it to flush buffers on shutdown
            self._writer_thread = threading.Thread(target=self.writer.start, name="Writer")
            self._writer_thread.daemon = False
            self._writer_thread.start()
            
            # Start workers
            self.worker_pool.start()
            
            # Start progress monitor
            self.monitor.start()
            
            # Run producer in main thread (blocking)
            self.producer.start()
            
            logger.info("Producer finished. Waiting for queues to drain...")
            
            # Wait for URL queue to drain
            self._wait_for_queue(self.url_queue, "url_queue")
            
            # Signal workers to finish their current items
            time.sleep(2)
            
            # Wait for embed queue to drain
            self._wait_for_queue(self.embed_queue, "embed_queue")
            
            logger.info("Queues drained. Initiating shutdown...")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self._shutdown()
        
        return self._stats
    
    def _wait_for_queue(self, queue: Queue, name: str, timeout: int = 300):
        """Waits for a queue to empty with timeout."""
        start = time.time()
        
        while not queue.empty():
            if self._shutdown_requested:
                break
            
            elapsed = time.time() - start
            if elapsed > timeout:
                logger.warning(f"{name} did not drain within {timeout}s timeout")
                break
            
            remaining = queue.qsize()
            if remaining > 0 and int(elapsed) % 10 == 0:
                logger.info(f"Waiting for {name}: {remaining} items remaining")
            
            time.sleep(1)
    
    def _install_signal_handlers(self):
        """Installs signal handlers for graceful shutdown."""
        def handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            self._shutdown_requested = True
            self._shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    def _shutdown(self):
        """Gracefully shuts down all components."""
        logger.info("Shutting down pipeline...")
        
        # Stop monitor
        if self.monitor:
            self.monitor.stop()
        
        # Stop producer
        if self.producer:
            self.producer.stop()
        
        # Stop workers (this also drains the embedder)
        if self.worker_pool:
            self.worker_pool.stop()
        
        # Wait for embed_queue to drain before stopping writer
        if self.embed_queue:
            drain_start = time.time()
            while not self.embed_queue.empty():
                if time.time() - drain_start > 30:
                    logger.warning("Embed queue drain timeout")
                    break
                time.sleep(0.1)
            logger.info(f"Embed queue drained in {time.time() - drain_start:.1f}s")
        
        # Stop writer (flushes buffers) and wait for it to finish
        if self.writer:
            self.writer.stop()
            if hasattr(self, '_writer_thread') and self._writer_thread.is_alive():
                logger.info("Waiting for writer thread to finish...")
                self._writer_thread.join(timeout=30.0)
                if self._writer_thread.is_alive():
                    logger.warning("Writer thread did not finish in time")
        
        # Collect final statistics
        self._collect_stats()
        
        logger.info("Pipeline shutdown complete")
    
    def _collect_stats(self):
        """Collects final statistics from all components."""
        self._stats.end_time = time.time()
        
        if self.producer:
            self._stats.producer_stats = self.producer.get_stats()
        
        if self.worker_pool:
            self._stats.worker_stats = self.worker_pool.get_stats()
        
        if self.writer:
            self._stats.writer_stats = self.writer.get_stats()
        
        # Log final summary
        stats_dict = self._stats.to_dict()
        logger.info(f"Pipeline completed in {self._stats.duration_human}")
        logger.info(f"Final stats: {stats_dict}")


class MultiDumpPipeline(BatchPipeline):
    """
    Pipeline for processing multiple dump files.
    
    Maintains global deduplication across all dumps.
    """
    
    def __init__(
        self,
        dump_paths: List[str],
        num_workers: Optional[int] = None,
        limit_per_dump: int = 0,
        total_limit: int = 0
    ):
        if dump_paths is None or not dump_paths:
            raise ValueError("dump_paths is required")
        
        # Use first dump path for parent init
        super().__init__(
            dump_path=dump_paths[0],
            num_workers=num_workers,
            limit=total_limit
        )
        
        self.dump_paths = [Path(p) for p in dump_paths]
        self.limit_per_dump = limit_per_dump
        self.total_limit = total_limit
    
    def setup(self):
        """Initializes components with multi-dump producer."""
        logger.info(f"Setting up multi-dump pipeline for {len(self.dump_paths)} dumps...")
        
        # Create queues
        self.url_queue = Queue(maxsize=self.url_queue_size)
        self.embed_queue = Queue(maxsize=self.embed_queue_size)
        
        # Create multi-dump producer
        self.producer = MultiProducer(
            url_queue=self.url_queue,
            dump_paths=[str(p) for p in self.dump_paths],
            limit_per_dump=self.limit_per_dump,
            total_limit=self.total_limit
        )
        
        # Create worker pool
        self.worker_pool = WorkerPool(
            url_queue=self.url_queue,
            embed_queue=self.embed_queue,
            num_workers=self.num_workers
        )
        
        # Create writer
        self.writer = BatchWriter(embed_queue=self.embed_queue)
        
        # Create progress monitor
        self.monitor = ProgressMonitor(
            url_queue=self.url_queue,
            embed_queue=self.embed_queue,
            worker_pool=self.worker_pool,
            writer=self.writer
        )
        
        logger.info(f"Multi-dump pipeline initialized with {len(self.dump_paths)} dumps")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Truth Atlas Batch Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single dump
    python -m phase1_offline.pipeline --dump dataset/sample-m.tar
    
    # Process with limited URLs for testing
    python -m phase1_offline.pipeline --dump dataset/sample-m.tar --limit 1000
    
    # Process multiple dumps
    python -m phase1_offline.pipeline --dumps dataset/dump1.tar dataset/dump2.tar
    
    # Adjust worker count
    python -m phase1_offline.pipeline --dump dataset/sample-m.tar --workers 16
"""
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dump",
        help="Path to single dump file"
    )
    input_group.add_argument(
        "--dumps",
        nargs='+',
        help="Paths to multiple dump files"
    )
    
    # Processing options
    parser.add_argument(
        "--workers",
        type=int,
        default=config.get("batch_processing.workers", 4),
        help="Number of worker threads"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit total URLs to process (0 = no limit)"
    )
    parser.add_argument(
        "--limit-per-dump",
        type=int,
        default=0,
        help="Limit URLs per dump file (0 = no limit)"
    )
    
    # Queue options
    parser.add_argument(
        "--url-queue-size",
        type=int,
        default=config.get("batch_processing.url_queue_size", 10000),
        help="URL queue capacity"
    )
    parser.add_argument(
        "--embed-queue-size",
        type=int,
        default=config.get("batch_processing.embed_queue_size", 5000),
        help="Embedding queue capacity"
    )
    
    # Output options
    parser.add_argument(
        "--stats-file",
        help="Output file for statistics JSON"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger("pipeline", console_output=True)
    
    # Validate dump paths
    if args.dump:
        dump_path = Path(args.dump)
        if not dump_path.exists():
            logger.error(f"Dump file not found: {dump_path}")
            sys.exit(1)
        
        pipeline = BatchPipeline(
            dump_path=str(dump_path),
            num_workers=args.workers,
            limit=args.limit,
            url_queue_size=args.url_queue_size,
            embed_queue_size=args.embed_queue_size
        )
    else:
        # Multiple dumps
        for path in args.dumps:
            if not Path(path).exists():
                logger.error(f"Dump file not found: {path}")
                sys.exit(1)
        
        pipeline = MultiDumpPipeline(
            dump_paths=args.dumps,
            num_workers=args.workers,
            limit_per_dump=args.limit_per_dump,
            total_limit=args.limit
        )
    
    # Run pipeline
    try:
        stats = pipeline.run()
        
        # Output statistics
        if args.stats_file:
            import json
            with open(args.stats_file, 'w') as f:
                json.dump(stats.to_dict(), f, indent=2)
            logger.info(f"Statistics written to {args.stats_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Duration: {stats.duration_human}")
        print(f"Throughput: {stats.throughput:.1f} docs/sec")
        
        if stats.producer_stats:
            print(f"\nProducer:")
            print(f"  Total records: {stats.producer_stats.get('total_records', 0):,}")
            print(f"  Queued: {stats.producer_stats.get('queued', 0):,}")
            print(f"  Filtered: {stats.producer_stats.get('filtered_out', 0):,}")
        
        if stats.worker_stats:
            print(f"\nWorkers:")
            print(f"  Processed: {stats.worker_stats.get('processed', 0):,}")
            print(f"  Success: {stats.worker_stats.get('success', 0):,}")
        
        if stats.writer_stats:
            print(f"\nWriter:")
            print(f"  Received: {stats.writer_stats.get('received', 0):,}")
            print(f"  Accepted: {stats.writer_stats.get('accepted', 0):,}")
            print(f"  Acceptance rate: {stats.writer_stats.get('acceptance_rate', 0):.1%}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
