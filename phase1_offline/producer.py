"""
Producer module for Phase 1 batch processing pipeline.

Reads dump files (SLOP, JSONL, plain text) and pushes URLs to the processing queue.
Applies URL filtering, deduplication, and pattern blacklisting.

Key features:
- Auto-detection of dump formats via adapter registry
- URL canonicalization and Bloom filter deduplication
- Pattern-based filtering (spam, social media, login pages)
- Pre-fetched HTML passthrough when available (SLOP format)
"""

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Optional, Set, Dict, Any, List
from urllib.parse import urlparse

from common.logging.logger import get_logger
from common.config import config
from common.database import db
from common.models import DumpJob
from common.repositories import JobRepository
from phase1_offline.dump_adapters import adapter_registry, DumpRecord
from phase1_offline.deduplication import BloomFilter, URLCanonicalizer

logger = get_logger("producer")


@dataclass
class QueueItem:
    """Item placed on the URL queue for workers."""
    url: str
    html: Optional[str] = None  # Pre-fetched HTML if available
    metadata: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None


class URLFilter:
    """
    URL pattern filter for quality control.
    
    Filters out:
    - Known spam/low-quality domains
    - Social media URLs
    - E-commerce/cart pages
    - Login/authentication pages
    - Short URL services
    """
    
    # Domain blacklist (substrings)
    DOMAIN_BLACKLIST = {
        # Social media
        'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com',
        'pinterest.com', 'linkedin.com', 'snapchat.com', 'whatsapp.com',
        'reddit.com', 'tumblr.com', 'discord.com', 'telegram.org',
        
        # E-commerce
        'amazon.com', 'ebay.com', 'aliexpress.com', 'wish.com',
        'alibaba.com', 'etsy.com', 'shopify.com',
        
        # URL shorteners
        'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly',
        'is.gd', 'buff.ly', 'j.mp', 'soo.gd',
        
        # Generic low-quality
        'blogspot.', 'wordpress.com', 'weebly.com', 'wix.com',
        'medium.com', 'substack.com',  # May reconsider these
        
        # Spam indicators
        'click.', 'track.', 'ads.', 'pixel.', 'analytics.',
    }
    
    # Path pattern blacklist (regex)
    PATH_BLACKLIST_PATTERNS = [
        r'/login', r'/signin', r'/signup', r'/register',
        r'/cart', r'/checkout', r'/basket', r'/order',
        r'/account', r'/profile', r'/settings', r'/preferences',
        r'/password', r'/forgot', r'/reset',
        r'/unsubscribe', r'/opt-out',
        r'/ads?/', r'/sponsor', r'/promo',
        r'/share\?', r'/tweet\?', r'/pin\?',
        r'\.(jpg|jpeg|png|gif|webp|svg|pdf|mp3|mp4|avi|mov)$',
        r'/tag/', r'/category/', r'/author/',  # Often low-content
        r'/page/\d+', r'/p/\d+',  # Pagination pages
        r'\?replytocom=', r'#comment-',  # Comment permalinks
    ]
    
    # File extension blacklist
    EXTENSION_BLACKLIST = {
        '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.ico',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
        '.zip', '.rar', '.tar', '.gz', '.7z',
        '.exe', '.dmg', '.pkg', '.deb', '.rpm',
        '.css', '.js', '.json', '.xml', '.rss', '.atom',
    }
    
    def __init__(self, custom_blacklist: Optional[Set[str]] = None):
        self._path_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.PATH_BLACKLIST_PATTERNS
        ]
        self._custom_blacklist = custom_blacklist or set()
        self._stats = {
            'total_checked': 0,
            'domain_blocked': 0,
            'path_blocked': 0,
            'extension_blocked': 0,
            'custom_blocked': 0,
            'passed': 0,
        }
    
    def is_allowed(self, url: str) -> bool:
        """
        Checks if URL passes all filters.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is allowed, False if filtered
        """
        if url is None:
            raise ValueError("url is required")
        
        self._stats['total_checked'] += 1
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # Domain blacklist
            for blacklisted in self.DOMAIN_BLACKLIST:
                if blacklisted in domain:
                    self._stats['domain_blocked'] += 1
                    return False
            
            # Custom blacklist
            if domain in self._custom_blacklist:
                self._stats['custom_blocked'] += 1
                return False
            
            # Extension blacklist
            for ext in self.EXTENSION_BLACKLIST:
                if path.endswith(ext):
                    self._stats['extension_blocked'] += 1
                    return False
            
            # Path pattern blacklist
            for pattern in self._path_patterns:
                if pattern.search(path):
                    self._stats['path_blocked'] += 1
                    return False
            
            # Length sanity check
            if len(url) > 2000:
                return False
            
            self._stats['passed'] += 1
            return True
            
        except Exception as e:
            logger.warning(f"URL filter error for {url}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Returns filtering statistics."""
        return self._stats.copy()


class Producer:
    """
    Dump file producer for the batch processing pipeline.
    
    Reads URLs from dump files, applies filtering and deduplication,
    and pushes valid URLs to the processing queue.
    """
    
    def __init__(
        self,
        url_queue: Queue,
        dump_path: str,
        job_id: Optional[str] = None,
        limit: int = 0,
        expected_urls: int = 1_000_000,
        database=None,
        job_repo=None,
    ):
        if url_queue is None:
            raise ValueError("url_queue is required")
        if dump_path is None:
            raise ValueError("dump_path is required")

        self.url_queue = url_queue
        self.dump_path = Path(dump_path)
        self.job_id = job_id or str(uuid.uuid4())[:8]
        self.limit = limit
        self.running = False

        # DI
        self._database = database or db
        self._job_repo = job_repo or JobRepository(self._database)

        # Initialize components
        self.bloom_filter = BloomFilter(expected_urls)

        # Statistics
        self._stats = {
            'total_records': 0,
            'duplicates': 0,
            'queued': 0,
            'with_html': 0,
        }
        
        logger.info(f"Producer initialized for {dump_path} (job_id={self.job_id})")
    
    def start(self):
        """Starts reading from the dump file and populating the queue."""
        self.running = True
        logger.info(f"Producer starting for {self.dump_path}")
        
        # Register job in database
        self._register_job()
        
        try:
            # Iterate through dump records
            for record in adapter_registry.iterate(self.dump_path):
                if not self.running:
                    logger.info("Producer stopped by request")
                    break
                
                self._stats['total_records'] += 1

                # Canonicalize URL
                canonical_url = URLCanonicalizer.canonicalize(record.url)
                
                # Check Bloom filter for duplicates
                if canonical_url in self.bloom_filter:
                    self._stats['duplicates'] += 1
                    continue
                
                # Add to Bloom filter
                self.bloom_filter.add(canonical_url)
                
                # Create queue item
                item = QueueItem(
                    url=canonical_url,
                    html=record.html,
                    metadata=record.metadata or {},
                    job_id=self.job_id
                )
                
                if record.html:
                    self._stats['with_html'] += 1
                
                # Blocking put if queue is full
                self.url_queue.put(item)
                self._stats['queued'] += 1
                
                # Progress logging
                if self._stats['queued'] % 1000 == 0:
                    logger.info(f"Queued {self._stats['queued']} URLs "
                               f"(dupes={self._stats['duplicates']})")
                
                # Check limit
                if self.limit > 0 and self._stats['queued'] >= self.limit:
                    logger.info(f"Reached limit of {self.limit} URLs")
                    break
            
            self._finalize_job(success=True)
            logger.info(f"Producer finished. Final stats: {self._stats}")
            
        except Exception as e:
            logger.error(f"Producer failed: {e}")
            self._finalize_job(success=False, error=str(e))
            raise
        finally:
            self.running = False
    
    def stop(self):
        """Signals the producer to stop."""
        self.running = False
    
    def _register_job(self):
        """Registers the processing job in the database."""
        try:
            job = DumpJob(
                id=self.job_id,
                dump_name=self.dump_path.name,
                dump_path=str(self.dump_path),
            )
            self._job_repo.register_job(job)
        except Exception as e:
            logger.warning(f"Failed to register job: {e}")

    def _finalize_job(self, success: bool, error: Optional[str] = None):
        """Updates job status in database."""
        try:
            status = 'completed' if success else 'failed'
            self._job_repo.finalize_job(
                job_id=self.job_id,
                status=status,
                total_urls=self._stats['total_records'],
                filtered_urls=self._stats['queued'],
                error=error,
            )
        except Exception as e:
            logger.warning(f"Failed to finalize job: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns producer statistics."""
        return {
            **self._stats,
            'bloom_filter_size': len(self.bloom_filter),
        }


class MultiProducer:
    """
    Producer for multiple dump files.
    
    Iterates through multiple dumps sequentially, maintaining
    global deduplication across all files.
    """
    
    def __init__(
        self,
        url_queue: Queue,
        dump_paths: List[str],
        limit_per_dump: int = 0,
        total_limit: int = 0
    ):
        if url_queue is None:
            raise ValueError("url_queue is required")
        if dump_paths is None:
            raise ValueError("dump_paths is required")
        
        self.url_queue = url_queue
        self.dump_paths = [Path(p) for p in dump_paths]
        self.limit_per_dump = limit_per_dump
        self.total_limit = total_limit
        
        # Shared deduplication across dumps
        total_expected = len(dump_paths) * 1_000_000
        self.bloom_filter = BloomFilter(total_expected)
        
        self._total_queued = 0
        self.running = False
    
    def start(self):
        """Processes all dump files sequentially."""
        self.running = True
        
        for dump_path in self.dump_paths:
            if not self.running:
                break
            
            if self.total_limit > 0 and self._total_queued >= self.total_limit:
                logger.info(f"Reached total limit of {self.total_limit}")
                break
            
            logger.info(f"Processing dump: {dump_path}")
            
            # Create producer with shared filters
            producer = Producer(
                url_queue=self.url_queue,
                dump_path=str(dump_path),
                limit=self.limit_per_dump
            )
            
            # Share deduplication state
            producer.bloom_filter = self.bloom_filter
            
            try:
                producer.start()
                self._total_queued += producer._stats['queued']
            except Exception as e:
                logger.error(f"Failed to process {dump_path}: {e}")
                continue
        
        logger.info(f"MultiProducer finished. Total queued: {self._total_queued}")
    
    def stop(self):
        """Signals to stop processing."""
        self.running = False
