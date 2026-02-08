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

    Extraction aims for maximum recall of human-readable text
    while stripping ads, navigation, code blocks, and boilerplate.
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

        # Try trafilatura first (favor_recall for completeness)
        if self._trafilatura:
            try:
                text = self._trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    favor_recall=True,
                    deduplicate=True,
                    url=url,
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

        # Post-process for cleanliness
        if text:
            text = clean_extracted_text(text)

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
            logger.error("requests not installed; cannot fetch URL content")
            self.running = False
            raise RuntimeError("requests dependency is required for synchronous fetching")
        
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
            logger.error("Embedding model unavailable; cannot generate embeddings")
            self.running = False
            raise RuntimeError("Embedding model is required for batch processing")
        
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
        """Embeds a batch of items with OOM protection."""
        model = get_embedding_model()

        if model is None:
            logger.error("Embedding model unavailable; cannot generate embeddings")
            self.running = False
            raise RuntimeError("Embedding model is required for concurrent embedding")

        self._encode_with_oom_retry(model, batch)

    def _encode_with_oom_retry(self, model, batch: List[Dict], current_batch_size: int = None):
        """Encodes batch, retrying with half batch size on OOM."""
        if current_batch_size is None:
            current_batch_size = len(batch)

        try:
            texts = [item['text'] for item in batch]
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=current_batch_size
            )

            for item, emb in zip(batch, embeddings):
                item['embedding'] = emb.tolist()
                self.output_queue.put(item)

            self._stats['batches'] += 1
            self._stats['items'] += len(batch)

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'MPS' in str(e):
                half = max(1, current_batch_size // 2)
                logger.warning(f"OOM with batch_size={current_batch_size}, retrying with {half}")
                if half < current_batch_size:
                    self._encode_with_oom_retry(model, batch, half)
                else:
                    logger.error(f"OOM even with batch_size=1, dropping {len(batch)} items")
                    self._stats['errors'] += len(batch)
            else:
                logger.error(f"Batch embedding failed: {e}")
                self._stats['errors'] += len(batch)
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


def _clean_extracted_text(text: str) -> str:
    """
    Cleans extracted text by removing non-human-language artifacts.

    Strips: code blocks, inline code, cookie/privacy banners, navigation
    lists, excessive formatting, ad copy, and other boilerplate.

    Must be a module-level function (used by ProcessPoolExecutor workers).
    """
    import re

    # 1. Remove code blocks (fenced markdown, indented blocks of code-like content)
    text = re.sub(r'```[\s\S]*?```', '', text)

    # 2. Remove inline code/markup artifacts
    text = re.sub(r'`[^`]+`', '', text)

    # 3. Remove HTML/XML fragments that survived extraction
    text = re.sub(r'<[^>]{1,200}>', '', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)

    # 4. Remove CSS/JS artifacts (selectors, property blocks, function calls, media queries)
    text = re.sub(r'[.#]?[a-zA-Z_][\w-]*\s*\{[^}]*\}', '', text)
    text = re.sub(r'@media[^}]*\{[^}]*\}', '', text)
    text = re.sub(r'@media[^\n]*', '', text)
    text = re.sub(r'[a-zA-Z_]\w*\s*\([^)]{0,100}\)\s*\{', '', text)
    text = re.sub(r'(?m)^[.#@][a-zA-Z_][\w-]*\s*$', '', text)

    # 5. Remove common cookie/privacy/consent banner text
    cookie_patterns = [
        r'(?i)(?:we use|this (?:site|website) uses?) cookies?[^\n]{0,300}',
        r'(?i)by (?:continuing|clicking|using)[^\n]{0,200}(?:cookies?|consent|privacy)',
        r'(?i)cookie (?:policy|settings|preferences|notice)[^\n]{0,200}',
        r'(?i)accept (?:all )?cookies?[^\n]{0,100}',
        r'(?i)manage (?:cookie|consent|privacy)[^\n]{0,100}',
        r'(?i)gdpr[^\n]{0,200}',
    ]
    for pat in cookie_patterns:
        text = re.sub(pat, '', text)

    # 6. Remove navigation-like lists (many short items separated by | or newlines)
    text = re.sub(
        r'(?:^|\n)(?:[^\n]{1,30}\s*\|\s*){3,}[^\n]{1,30}(?:\n|$)',
        '\n', text,
    )

    # 7. Remove ad/CTA lines
    text = re.sub(
        r'(?im)^.*(?:subscribe|sign up|newsletter|download now|buy now|add to cart|'
        r'free trial|limited time|click here|learn more|get started).*$',
        '', text,
    )

    # 8. Remove lines that are mostly non-alphabetic (code, data tables, hashes)
    cleaned_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        alpha_ratio = alpha_chars / max(len(stripped), 1)
        # Keep lines that are at least 40% alphabetic or very short (headings)
        if alpha_ratio >= 0.4 or len(stripped) < 20:
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # 9. Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.strip()

    return text


# Expose as module-level alias for ContentExtractor
clean_extracted_text = _clean_extracted_text


def _extract_worker_fn(item_tuple):
    """
    Top-level extraction function for ProcessPoolExecutor.

    Must be module-level (not a method) for pickling.
    Each subprocess imports trafilatura independently.

    Uses favor_recall for maximum content completeness, deduplicate
    to strip repeated boilerplate, and post-processing to remove
    non-human-language artifacts.

    Args:
        item_tuple: (url, html, metadata) - the item to extract

    Returns:
        (url, text, metadata, raw_html_hash) or None on failure
    """
    import hashlib
    import re

    url, html, metadata = item_tuple

    # Handle pre-extracted text passthrough
    if metadata and metadata.get('pre_extracted') and metadata.get('text'):
        text = metadata['text']
        text = _clean_extracted_text(text)
        raw_hash = hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()
        if len(text) < 100:
            return None
        return (url, text, metadata, raw_hash)

    if not html:
        return None

    # Compute raw HTML hash
    raw_hash = hashlib.sha256(html.encode('utf-8', errors='ignore')).hexdigest()

    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
            deduplicate=True,
            url=url,
        )

        if not text:
            return None

        # Extract metadata from trafilatura
        try:
            meta = trafilatura.extract_metadata(html)
            if meta:
                meta_dict = meta.as_dict()
                metadata = metadata or {}
                metadata.update({
                    'title': meta_dict.get('title'),
                    'author': meta_dict.get('author'),
                    'date': meta_dict.get('date'),
                    'description': meta_dict.get('description'),
                    'sitename': meta_dict.get('sitename'),
                })
        except Exception:
            pass

    except Exception:
        # Fallback: try readability
        try:
            from readability import Document
            doc = Document(html)
            text = doc.summary()
            if metadata is None:
                metadata = {}
            metadata['title'] = doc.title()
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        except Exception:
            return None

    if not text:
        return None

    # Post-process for cleanliness
    text = _clean_extracted_text(text)

    # Min length guard
    if len(text) < 100:
        return None

    # Max length truncation
    if len(text) > 100_000:
        text = text[:100_000]

    return (url, text, metadata or {}, raw_hash)


class AsyncFetcher(threading.Thread):
    """
    Async HTTP fetcher running in a dedicated thread with its own asyncio event loop.

    Uses aiohttp with configurable concurrency for high-throughput URL fetching.
    Passes through items with pre-fetched HTML or pre-extracted text.
    Optional playwright fallback for JS-rendered pages.
    """

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        concurrency: Optional[int] = None,
        timeout: Optional[int] = None,
        use_playwright: bool = False,
        playwright_concurrency: int = 4,
    ):
        super().__init__(name="AsyncFetcher", daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.concurrency = concurrency or config.get("batch_processing.fetch_concurrency")
        self.timeout = timeout or config.get("batch_processing.fetch_timeout")
        self.use_playwright = use_playwright
        self.playwright_concurrency = playwright_concurrency
        self.running = True
        self._draining = False

        self._stats = {
            'fetched': 0,
            'fetch_errors': 0,
            'playwright_fetched': 0,
            'pre_fetched': 0,
            'pre_extracted': 0,
        }

    def run(self):
        """Creates new asyncio event loop in thread and runs fetch loop."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._fetch_loop())
        finally:
            loop.close()
        logger.info(f"AsyncFetcher finished: {self._stats}")

    async def _fetch_loop(self):
        """Main async fetch loop."""
        import asyncio
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed, falling back to synchronous fetching")
            self._sync_fallback_loop()
            return

        connector = aiohttp.TCPConnector(
            limit=self.concurrency,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        client_timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {
            'User-Agent': 'TruthAtlas/1.0 (Research Crawler)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        # Playwright browser (optional)
        playwright_sem = None
        browser = None
        if self.use_playwright:
            try:
                from playwright.async_api import async_playwright
                pw = await async_playwright().start()
                browser = await pw.chromium.launch(headless=True)
                playwright_sem = asyncio.Semaphore(self.playwright_concurrency)
                logger.info(f"Playwright browser launched (concurrency={self.playwright_concurrency})")
            except Exception as e:
                logger.warning(f"Playwright not available: {e}")
                browser = None

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=client_timeout,
            headers=headers,
        ) as session:
            pending_tasks = set()
            max_pending = self.concurrency

            while self.running or self._draining or not self.input_queue.empty() or pending_tasks:
                # Fill up pending tasks from input queue
                while len(pending_tasks) < max_pending:
                    try:
                        item = self.input_queue.get_nowait()
                    except Exception:
                        break
                    task = asyncio.create_task(
                        self._fetch_one(session, item, browser, playwright_sem)
                    )
                    pending_tasks.add(task)
                    task.add_done_callback(pending_tasks.discard)

                if not pending_tasks:
                    if self._draining and self.input_queue.empty():
                        break
                    await asyncio.sleep(0.05)
                    continue

                # Wait for at least one task to complete
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, timeout=0.5, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    try:
                        task.result()  # propagate exceptions to log
                    except Exception as e:
                        logger.debug(f"Fetch task error: {e}")

        if browser:
            await browser.close()

    async def _fetch_one(self, session, item, browser=None, playwright_sem=None):
        """Fetch a single item."""
        from phase1_offline.producer import QueueItem

        url = item.url
        metadata = item.metadata or {}

        # Passthrough for pre-extracted text
        if metadata.get('pre_extracted') and metadata.get('text'):
            self._stats['pre_extracted'] += 1
            self.output_queue.put((url, None, metadata))
            self.input_queue.task_done()
            return

        # Passthrough for pre-fetched HTML
        if item.html:
            self._stats['pre_fetched'] += 1
            self.output_queue.put((url, item.html, metadata))
            self.input_queue.task_done()
            return

        # Fetch via aiohttp
        html = await self._aiohttp_fetch(session, url)

        if html:
            self._stats['fetched'] += 1
            self.output_queue.put((url, html, metadata))
            self.input_queue.task_done()
            return

        # Playwright fallback for JS-rendered pages
        if browser and playwright_sem:
            html = await self._playwright_fetch(browser, playwright_sem, url)
            if html:
                self._stats['playwright_fetched'] += 1
                self.output_queue.put((url, html, metadata))
                self.input_queue.task_done()
                return

        self._stats['fetch_errors'] += 1
        self.input_queue.task_done()

    async def _aiohttp_fetch(self, session, url: str, max_retries: int = 2) -> Optional[str]:
        """Fetch URL with aiohttp and retry logic."""
        import aiohttp
        import asyncio

        for attempt in range(max_retries):
            try:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status == 200:
                        return await response.text(errors='replace')
                    elif response.status in (429, 503):
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return None
            except asyncio.TimeoutError:
                pass
            except aiohttp.ClientError:
                pass
            except Exception:
                return None

            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))

        return None

    async def _playwright_fetch(self, browser, sem, url: str) -> Optional[str]:
        """Fetch URL with playwright for JS-rendered pages."""
        import asyncio

        async with sem:
            page = None
            try:
                page = await browser.new_page()
                await page.goto(url, timeout=self.timeout * 1000, wait_until='networkidle')
                html = await page.content()
                return html
            except asyncio.TimeoutError:
                return None
            except Exception:
                return None
            finally:
                if page:
                    await page.close()

    def _sync_fallback_loop(self):
        """Synchronous fallback when aiohttp is not installed."""
        try:
            import requests
        except ImportError:
            logger.error("requests not installed; cannot run synchronous fetcher")
            self.running = False
            raise RuntimeError("requests dependency is required for synchronous fetching")

        while self.running or self._draining or not self.input_queue.empty():
            try:
                item = self.input_queue.get(timeout=0.5)
            except Empty:
                if self._draining and self.input_queue.empty():
                    break
                continue

            from phase1_offline.producer import QueueItem
            url = item.url
            metadata = item.metadata or {}

            if metadata.get('pre_extracted') and metadata.get('text'):
                self._stats['pre_extracted'] += 1
                self.output_queue.put((url, None, metadata))
                self.input_queue.task_done()
                continue

            if item.html:
                self._stats['pre_fetched'] += 1
                self.output_queue.put((url, item.html, metadata))
                self.input_queue.task_done()
                continue

            # Synchronous fetch
            try:
                response = requests.get(url, timeout=self.timeout, headers={
                    'User-Agent': 'TruthAtlas/1.0 (Research Crawler)',
                })
                if response.status_code == 200:
                    self._stats['fetched'] += 1
                    self.output_queue.put((url, response.text, metadata))
                else:
                    self._stats['fetch_errors'] += 1
            except Exception:
                self._stats['fetch_errors'] += 1

            self.input_queue.task_done()

    def stop(self):
        """Signals fetcher to stop."""
        self.running = False

    def drain_and_stop(self, timeout: float = 30.0):
        """Drains remaining items and stops."""
        self._draining = True
        self.running = False
        self.join(timeout=timeout)

    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()


class ExtractPool:
    """
    Coordinator that dispatches HTML extraction to a ProcessPoolExecutor.

    Reads (url, html, metadata) tuples from html_queue, submits to process pool,
    collects results and puts (url, text, metadata, raw_html_hash) on text_queue.
    """

    def __init__(
        self,
        html_queue: Queue,
        text_queue: Queue,
        num_processes: Optional[int] = None,
    ):
        self.html_queue = html_queue
        self.text_queue = text_queue
        self.num_processes = num_processes or config.get("batch_processing.extract_processes")
        self._executor = None
        self._thread = None
        self.running = True
        self._draining = False

        self._stats = {
            'extracted': 0,
            'extract_errors': 0,
            'passthrough': 0,
        }

    def start(self):
        """Starts the coordinator thread and process pool."""
        from concurrent.futures import ProcessPoolExecutor
        self._executor = ProcessPoolExecutor(max_workers=self.num_processes)
        self._thread = threading.Thread(target=self._run, name="ExtractPool", daemon=True)
        self._thread.start()
        logger.info(f"ExtractPool started with {self.num_processes} processes")

    def _run(self):
        """Main coordinator loop."""
        from concurrent.futures import as_completed
        futures = {}
        max_inflight = self.num_processes * 4  # Keep pool busy

        while self.running or self._draining or not self.html_queue.empty() or futures:
            # Submit new work
            while len(futures) < max_inflight:
                try:
                    item = self.html_queue.get_nowait()
                except Empty:
                    break

                url, html, metadata = item

                # Passthrough for pre-extracted text
                if metadata and metadata.get('pre_extracted') and metadata.get('text'):
                    import hashlib
                    text = metadata['text']
                    raw_hash = hashlib.sha256(
                        text.encode('utf-8', errors='ignore')
                    ).hexdigest()
                    self.text_queue.put({
                        'url': url,
                        'text': text,
                        'metadata': metadata,
                        'raw_html_hash': raw_hash,
                    })
                    self._stats['passthrough'] += 1
                    self.html_queue.task_done()
                    continue

                future = self._executor.submit(
                    _extract_worker_fn, (url, html, metadata)
                )
                futures[future] = url
                self.html_queue.task_done()

            if not futures:
                if self._draining and self.html_queue.empty():
                    break
                time.sleep(0.05)
                continue

            # Collect completed futures
            done = []
            for future in list(futures):
                if future.done():
                    done.append(future)

            if not done:
                time.sleep(0.02)
                continue

            for future in done:
                url = futures.pop(future)
                try:
                    result = future.result()
                    if result:
                        r_url, text, metadata, raw_hash = result
                        self.text_queue.put({
                            'url': r_url,
                            'text': text,
                            'metadata': metadata,
                            'raw_html_hash': raw_hash,
                        })
                        self._stats['extracted'] += 1
                    else:
                        self._stats['extract_errors'] += 1
                except Exception as e:
                    logger.debug(f"Extraction failed for {url}: {e}")
                    self._stats['extract_errors'] += 1

        logger.info(f"ExtractPool finished: {self._stats}")

    def stop(self):
        """Signals pool to stop."""
        self.running = False

    def drain_and_stop(self, timeout: float = 60.0):
        """Drains remaining items and stops."""
        self._draining = True
        self.running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._executor:
            self._executor.shutdown(wait=True)

    def join(self, timeout: Optional[float] = None):
        """Waits for coordinator thread."""
        if self._thread:
            self._thread.join(timeout=timeout)

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
