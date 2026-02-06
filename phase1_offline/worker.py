"""
Worker module for Phase 1 batch processing pipeline.

Workers process URLs from the queue:
1. Fetch content (if not pre-fetched)
2. Extract text using trafilatura
3. Apply filters (language, length, quality)
4. Compute embeddings
5. Push results to the writer queue

Key features:
- Graceful handling of pre-fetched HTML (from SLOP format)
- Language detection (English only for MVP)
- Retry logic with exponential backoff
- Thread-safe embedding model sharing
- Batch embedding for efficiency
"""

import hashlib
import re
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from common.logging.logger import get_logger
from common.config import config

logger = get_logger("worker")


# Thread-safe embedding model singleton
_model_lock = threading.Lock()
_embedding_model = None


def get_embedding_model():
    """
    Returns shared embedding model instance.
    Thread-safe lazy initialization.
    """
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    with _model_lock:
        if _embedding_model is not None:
            return _embedding_model
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = config.get("embedding.model")
            device = config.get("embedding.device")
            
            logger.info(f"Loading embedding model {model_name} on {device}...")
            _embedding_model = SentenceTransformer(model_name, device=device)
            logger.info(f"Embedding model loaded successfully")
            
            return _embedding_model
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None


@dataclass
class WorkerResult:
    """Result from worker processing."""
    url: str
    text: str
    metadata: Dict[str, Any]
    raw_html_hash: str
    embedding: List[float]
    quality_metrics: Optional[Dict[str, float]] = None


class ContentExtractor:
    """
    Extracts main content from HTML using trafilatura.
    Falls back to readability if trafilatura fails.
    """
    
    def __init__(self):
        self._trafilatura = None
        self._readability = None
        self._load_extractors()
    
    def _load_extractors(self):
        """Lazy load extraction libraries."""
        try:
            import trafilatura
            self._trafilatura = trafilatura
        except ImportError:
            logger.warning("trafilatura not installed")
            raise ImportError()
        
        try:
            from readability import Document
            self._readability = Document
        except ImportError:
            logger.warning("readability-lxml not installed")
            
    
    def extract(self, html: str, url: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Extracts text and metadata from HTML.
        
        Args:
            html: HTML content
            url: Source URL (for metadata extraction)
            
        Returns:
            Tuple of (text, metadata dict)
        """
        if html is None:
            raise ValueError("html is required")
        if url is None:
            raise ValueError("url is required")
        
        text = None
        metadata = {'url': url}
        
        # Try trafilatura first
        if self._trafilatura:
            try:
                text = self._trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=False,
                    url=url
                )
                
                # Extract metadata
                meta = self._trafilatura.extract_metadata(html)
                if meta:
                    meta_dict = meta.as_dict()
                    metadata.update({
                        'title': meta_dict.get('title'),
                        'author': meta_dict.get('author'),
                        'date': meta_dict.get('date'),
                        'description': meta_dict.get('description'),
                        'sitename': meta_dict.get('sitename'),
                    })
            except Exception as e:
                logger.debug(f"Trafilatura failed for {url}: {e}")
        
        # Fallback to readability
        if not text and self._readability:
            try:
                doc = self._readability(html)
                text = doc.summary()
                metadata['title'] = doc.title()
                
                # Strip HTML tags from readability output
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
            except Exception as e:
                logger.debug(f"Readability failed for {url}: {e}")
        
        # Final cleanup
        if text:
            # Remove excessive whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = text.strip()
        
        return text, metadata


class LanguageDetector:
    """
    Detects document language.
    
    Uses langdetect library with fallback to simple heuristics.
    """
    
    def __init__(self, target_language: str = 'en'):
        if target_language is None:
            raise ValueError("target_language is required")
        
        self.target = target_language
        self._detector = None
        
        try:
            from langdetect import detect, detect_langs
            self._detector = detect
        except ImportError:
            logger.warning("langdetect not installed, using heuristics only")
    
    def is_target_language(self, text: str) -> bool:
        """
        Checks if text is in the target language.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is in target language
        """
        if text is None:
            raise ValueError("text is required")
        
        # Skip very short texts
        if len(text) < 50:
            return True  # Assume OK for short texts
        
        # Use langdetect if available
        if self._detector:
            try:
                # Use first 500 chars for speed
                sample = text[:500]
                detected = self._detector(sample)
                return detected == self.target
            except Exception:
                # Detection failed, use heuristics
                pass
        
        # Simple English heuristic: check for common English words
        common_english = {'the', 'and', 'is', 'it', 'to', 'of', 'a', 'in', 'that'}
        words = set(text.lower().split()[:100])
        english_ratio = len(words & common_english) / max(len(words), 1)
        
        return english_ratio > 0.1


class ContentFilter:
    """
    Filters content based on quality heuristics.
    """
    
    # Minimum text length (characters)
    MIN_LENGTH = 100
    
    # Maximum text length (truncate beyond this)
    MAX_LENGTH = 100_000
    
    # Maximum link density (ratio of links to text)
    MAX_LINK_DENSITY = 0.3
    
    # Spam indicators
    SPAM_PATTERNS = [
        r'buy now|order now|add to cart|limited time offer',
        r'click here|download now|sign up today',
        r'xxx|casino|poker|lottery|prize winner',
        r'enlarge|enhancement|weight loss|miracle cure',
        r'(?:.)\\1{5,}',  # Repeated characters
    ]
    
    def __init__(self):
        self._spam_regex = re.compile(
            '|'.join(self.SPAM_PATTERNS), 
            re.IGNORECASE
        )
        self._stats = {
            'too_short': 0,
            'spam_detected': 0,
            'passed': 0,
        }
    
    def filter(self, text: str) -> Tuple[bool, str, Optional[str]]:
        """
        Filters text content.
        
        Args:
            text: Text to filter
            
        Returns:
            Tuple of (passed, filtered_text, rejection_reason)
        """
        if text is None:
            raise ValueError("text is required")
        
        # Length check
        if len(text) < self.MIN_LENGTH:
            self._stats['too_short'] += 1
            return False, text, 'too_short'
        
        # Truncate if too long
        if len(text) > self.MAX_LENGTH:
            text = text[:self.MAX_LENGTH]
        
        # Spam detection
        if self._spam_regex.search(text[:1000]):  # Check first 1000 chars
            self._stats['spam_detected'] += 1
            return False, text, 'spam_detected'
        
        self._stats['passed'] += 1
        return True, text, None
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()


class BatchWorker(threading.Thread):
    """
    Worker thread for batch processing.
    
    Processes URLs from the queue, extracts content, computes embeddings,
    and pushes results to the writer queue.
    
    Uses shared embedding batch processor for concurrent embedding.
    """
    
    def __init__(
        self,
        worker_id: int,
        url_queue: Queue,
        embed_queue: Queue,
        embed_batch_queue: Optional[Queue] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        super().__init__(name=f"Worker-{worker_id}")
        
        if url_queue is None:
            raise ValueError("url_queue is required")
        if embed_queue is None:
            raise ValueError("embed_queue is required")
        
        self.worker_id = worker_id
        self.url_queue = url_queue
        self.embed_queue = embed_queue
        self.embed_batch_queue = embed_batch_queue  # Shared queue for batch embedding
        
        # Read all params from config
        self.batch_size = batch_size or config.get("batch_processing.worker_batch_size")
        self.max_retries = max_retries or config.get("batch_processing.max_retries")
        
        self.running = True
        self.daemon = True
        
        # Components - read config
        target_lang = config.get("language.target")
        self.extractor = ContentExtractor()
        self.language_detector = LanguageDetector(target_language=target_lang)
        self.content_filter = ContentFilter()
        
        # Stats
        self._stats = {
            'processed': 0,
            'fetch_errors': 0,
            'extract_errors': 0,
            'language_filtered': 0,
            'content_filtered': 0,
            'embed_errors': 0,
            'success': 0,
        }
        
        # Batch buffer for embeddings
        self._batch_buffer: List[Dict[str, Any]] = []
    
    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} started")
        
        while self.running:
            try:
                # Get item from queue with timeout
                item = self.url_queue.get(timeout=1.0)
            except Empty:
                # Flush any pending batches
                if self._batch_buffer:
                    self._flush_batch()
                continue
            
            try:
                self._process_item(item)
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
            finally:
                self.url_queue.task_done()
        
        # Final batch flush
        if self._batch_buffer:
            self._flush_batch()
        
        logger.info(f"Worker {self.worker_id} stopped. Stats: {self._stats}")
    
    def _process_item(self, item):
        """Processes a single queue item."""
        from phase1_offline.producer import QueueItem
        
        self._stats['processed'] += 1
        url = item.url
        
        # Check if text is pre-extracted (e.g., C4 dataset)
        if item.metadata and item.metadata.get('pre_extracted') and item.metadata.get('text'):
            text = item.metadata['text']
            raw_hash = hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()
            metadata = {'url': url, 'title': None}
            if item.metadata:
                metadata.update(item.metadata)
        else:
            # 1. Fetch HTML (or use pre-fetched)
            if item.html:
                html = item.html
            else:
                html = self._fetch(url)
                if not html:
                    self._stats['fetch_errors'] += 1
                    return
            
            # 2. Compute raw HTML hash
            raw_hash = hashlib.sha256(html.encode('utf-8', errors='ignore')).hexdigest()
            
            # 3. Extract text
            text, metadata = self.extractor.extract(html, url)
            if not text:
                self._stats['extract_errors'] += 1
                return
            
            # Merge item metadata
            if item.metadata:
                metadata.update(item.metadata)
        
        # 4. Language filter
        if not self.language_detector.is_target_language(text):
            self._stats['language_filtered'] += 1
            return
        
        # 5. Content filter
        passed, text, reason = self.content_filter.filter(text)
        if not passed:
            self._stats['content_filtered'] += 1
            return
        
        # 6. Add to batch buffer for embedding
        self._batch_buffer.append({
            'url': url,
            'text': text,
            'metadata': metadata,
            'raw_hash': raw_hash,
        })
        
        # Flush batch if full
        if len(self._batch_buffer) >= self.batch_size:
            self._flush_batch()
    
    def _fetch(self, url: str) -> Optional[str]:
        """
        Fetches URL content with retries.
        
        Uses requests library with fallback to simpler methods.
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed, using mock data")
            return f"<html><body>Mock content for {url}</body></html>"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    timeout=10,
                    headers={
                        'User-Agent': 'TruthAtlas/1.0 (Research Crawler)',
                        'Accept': 'text/html,application/xhtml+xml',
                        'Accept-Language': 'en-US,en;q=0.9',
                    },
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code in (403, 429):
                    # Rate limited or blocked, backoff
                    time.sleep(2 ** attempt)
                else:
                    logger.debug(f"HTTP {response.status_code} for {url}")
                    return None
                    
            except requests.Timeout:
                logger.debug(f"Timeout for {url} (attempt {attempt + 1})")
            except requests.RequestException as e:
                logger.debug(f"Request error for {url}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(0.5 * (2 ** attempt))
        
        return None
    
    def _flush_batch(self):
        """Sends batch items to shared embedder queue or embeds directly."""
        if not self._batch_buffer:
            return
        
        # If we have a shared embedding queue, use it for concurrent processing
        if self.embed_batch_queue is not None:
            for item in self._batch_buffer:
                # Send to concurrent embedder
                self.embed_batch_queue.put({
                    'url': item['url'],
                    'text': item['text'],
                    'metadata': item['metadata'],
                    'raw_html_hash': item['raw_hash'],
                })
                self._stats['success'] += 1
            self._batch_buffer = []
            return
        
        # Fallback: embed directly in worker thread
        model = get_embedding_model()
        
        if model is None:
            # Mock embeddings for testing
            for item in self._batch_buffer:
                embedding = np.random.rand(384).tolist()
                self._send_result(item, embedding)
            self._batch_buffer = []
            return
        
        try:
            # Batch encode
            texts = [item['text'] for item in self._batch_buffer]
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=len(texts)
            )
            
            # Send results
            for item, embedding in zip(self._batch_buffer, embeddings):
                self._send_result(item, embedding.tolist())
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            self._stats['embed_errors'] += len(self._batch_buffer)
        
        self._batch_buffer = []
    
    def _send_result(self, item: Dict, embedding: List[float]):
        """Sends processed result to writer queue."""
        result = {
            'url': item['url'],
            'text': item['text'],
            'metadata': item['metadata'],
            'raw_html_hash': item['raw_hash'],
            'embedding': embedding,
        }
        
        self.embed_queue.put(result)
        self._stats['success'] += 1
    
    def stop(self):
        """Signals worker to stop."""
        self.running = False
    
    def get_stats(self) -> Dict[str, int]:
        """Returns worker statistics."""
        return {
            **self._stats,
            'content_filter_stats': self.content_filter.get_stats(),
        }


class ConcurrentEmbedder(threading.Thread):
    """
    Dedicated thread for batch embedding with concurrent model access.
    
    Collects items from workers and processes them in batches for efficiency.
    """
    
    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        batch_size: Optional[int] = None,
        batch_timeout: float = 0.3
    ):
        super().__init__(name="ConcurrentEmbedder", daemon=True)
        
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size or config.get("embedding.batch_size")
        self.batch_timeout = batch_timeout
        self.running = True
        self._draining = False  # Flag for draining mode
        
        self._stats = {'batches': 0, 'items': 0, 'errors': 0}
    
    def run(self):
        """Main embedding loop - collects batches and embeds."""
        batch = []
        last_flush = time.time()
        
        # Continue while running OR draining remaining items
        while self.running or self._draining or not self.input_queue.empty():
            try:
                item = self.input_queue.get(timeout=0.1)
                batch.append(item)
                self.input_queue.task_done()
            except Empty:
                # If draining and queue is empty, we're done
                if self._draining and self.input_queue.empty():
                    break
            
            # Flush if batch is full or timeout reached
            should_flush = (
                len(batch) >= self.batch_size or
                (batch and time.time() - last_flush > self.batch_timeout) or
                (self._draining and batch)  # Always flush when draining
            )
            
            if should_flush and batch:
                self._process_batch(batch)
                batch = []
                last_flush = time.time()
        
        # Final flush
        if batch:
            self._process_batch(batch)
        
        logger.info(f"ConcurrentEmbedder finished: {self._stats}")
    
    def _process_batch(self, batch: List[Dict]):
        """Embeds a batch of items."""
        model = get_embedding_model()
        
        if model is None:
            # Mock embeddings
            for item in batch:
                item['embedding'] = np.random.rand(384).tolist()
                self.output_queue.put(item)
            self._stats['items'] += len(batch)
            return
        
        try:
            texts = [item['text'] for item in batch]
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=len(texts)
            )
            
            for item, emb in zip(batch, embeddings):
                item['embedding'] = emb.tolist()
                self.output_queue.put(item)
            
            self._stats['batches'] += 1
            self._stats['items'] += len(batch)
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            self._stats['errors'] += len(batch)
    
    def stop(self):
        """Signals embedder to stop accepting new items."""
        self.running = False
    
    def drain_and_stop(self, timeout: float = 10.0):
        """Drains remaining items and stops. Call after workers have stopped."""
        self._draining = True
        self.running = False
        # Wait for embedder to finish
        self.join(timeout=timeout)
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()


class WorkerPool:
    """
    Manages a pool of worker threads with concurrent embedding.
    
    Workers extract content in parallel, then a dedicated embedder
    processes batches for GPU efficiency.
    """
    
    def __init__(
        self,
        url_queue: Queue,
        embed_queue: Queue,
        num_workers: Optional[int] = None
    ):
        if url_queue is None:
            raise ValueError("url_queue is required")
        if embed_queue is None:
            raise ValueError("embed_queue is required")
        
        # Read from config
        num_workers = num_workers or config.get("batch_processing.workers")
        
        # Shared queue for items ready to embed
        self.pre_embed_queue: Queue = Queue(maxsize=1000)
        
        self.workers: List[BatchWorker] = []
        for i in range(num_workers):
            worker = BatchWorker(
                worker_id=i,
                url_queue=url_queue,
                embed_queue=embed_queue,
                embed_batch_queue=self.pre_embed_queue
            )
            self.workers.append(worker)
        
        # Concurrent embedder
        self.embedder = ConcurrentEmbedder(
            input_queue=self.pre_embed_queue,
            output_queue=embed_queue
        )
    
    def start(self):
        """Starts all workers and embedder."""
        # Start embedder first
        self.embedder.start()
        
        for worker in self.workers:
            worker.start()
        logger.info(f"Started {len(self.workers)} workers with concurrent embedder")
    
    def stop(self):
        """Stops all workers, then drains and stops embedder."""
        logger.info(f"Stopping {len(self.workers)} workers...")
        
        # First stop workers so they finish current items
        for worker in self.workers:
            worker.stop()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        # Now drain the embedder queue
        logger.info("Draining embedder queue...")
        self.embedder.drain_and_stop(timeout=30.0)
        
        logger.info("WorkerPool stopped")
    
    def join(self, timeout: Optional[float] = None):
        """Waits for all workers and embedder to finish."""
        for worker in self.workers:
            worker.join(timeout=timeout)
        if self.embedder.is_alive():
            self.embedder.join(timeout=timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """Aggregates stats from all workers and embedder."""
        totals = {
            'processed': 0,
            'fetch_errors': 0,
            'extract_errors': 0,
            'language_filtered': 0,
            'content_filtered': 0,
            'embed_errors': 0,
            'success': 0,
        }
        
        for worker in self.workers:
            worker_stats = worker.get_stats()
            for key in totals:
                if key in worker_stats:
                    totals[key] += worker_stats[key]
        
        totals['num_workers'] = len(self.workers)
        totals['active_workers'] = sum(1 for w in self.workers if w.is_alive())
        totals['embedder'] = self.embedder.get_stats()
        
        return totals
