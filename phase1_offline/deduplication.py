"""
Deduplication module using multiple strategies.

Strategies:
1. URL canonicalization and Bloom filter (fast pre-filter)
2. SimHash - locality-sensitive hash for near-duplicate text detection
3. MinHash - Jaccard similarity approximation for set-based comparison
4. Embedding similarity - semantic deduplication via vector space

Configuration thresholds (from config.json):
    - simhash_threshold: 3 (Hamming distance for near-duplicates)
    - minhash_jaccard_threshold: 0.5 (Jaccard similarity threshold)
    - embedding_similarity_threshold: 0.95 (Cosine similarity threshold)
"""

import hashlib
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict, Any
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import numpy as np

from common.logging.logger import get_logger
from common.config import config

logger = get_logger("deduplication")


class BloomFilter:
    """
    Simple Bloom filter for URL deduplication.
    
    Space-efficient probabilistic data structure for membership testing.
    False positives possible, no false negatives.
    """
    
    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01):
        if expected_elements is None:
            raise ValueError("expected_elements is required")
        if false_positive_rate is None:
            raise ValueError("false_positive_rate is required")
        
        # Calculate optimal size and number of hash functions
        self.size = self._optimal_size(expected_elements, false_positive_rate)
        self.num_hashes = self._optimal_hashes(self.size, expected_elements)
        
        # Bit array (using bytearray for efficiency)
        self.bit_array = bytearray((self.size + 7) // 8)
        self._count = 0
        
        logger.info(f"BloomFilter initialized: size={self.size} bits, "
                   f"hashes={self.num_hashes}, expected_fpr={false_positive_rate:.3f}")
    
    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculates optimal bit array size."""
        import math
        return int(-n * math.log(p) / (math.log(2) ** 2))
    
    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        """Calculates optimal number of hash functions."""
        import math
        return max(1, int(m / n * math.log(2)))
    
    def _get_hash_values(self, item: str) -> List[int]:
        """Generates hash values using double hashing."""
        # Use MD5 for two base hashes
        h = hashlib.md5(item.encode('utf-8')).digest()
        h1 = struct.unpack('<Q', h[:8])[0]
        h2 = struct.unpack('<Q', h[8:])[0]
        
        # Generate k hash values using double hashing
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]
    
    def add(self, item: str):
        """Adds an item to the filter."""
        if item is None:
            raise ValueError("item is required")
        
        for pos in self._get_hash_values(item):
            byte_idx = pos // 8
            bit_idx = pos % 8
            self.bit_array[byte_idx] |= (1 << bit_idx)
        self._count += 1
    
    def contains(self, item: str) -> bool:
        """Checks if item might be in the filter."""
        if item is None:
            raise ValueError("item is required")
        
        for pos in self._get_hash_values(item):
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True
    
    def __len__(self) -> int:
        return self._count
    
    def __contains__(self, item: str) -> bool:
        return self.contains(item)


class URLCanonicalizer:
    """
    Normalizes URLs for consistent deduplication.
    
    Normalizations:
    - Lowercase scheme and host
    - Remove default ports
    - Sort query parameters
    - Remove tracking parameters
    - Normalize path slashes
    """
    
    # Common tracking parameters to remove
    TRACKING_PARAMS = {
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'fbclid', 'gclid', 'ref', 'source', 'mc_cid', 'mc_eid',
        '_ga', '_gid', 'ICID', 'icid', 'trk', 'trkCampaign',
    }
    
    @classmethod
    def canonicalize(cls, url: str) -> str:
        """Returns canonical form of URL."""
        if url is None:
            raise ValueError("url is required")
        
        try:
            parsed = urlparse(url.strip())
            
            # Lowercase scheme and host
            scheme = parsed.scheme.lower()
            host = parsed.netloc.lower()
            
            # Remove default ports
            if ':80' in host and scheme == 'http':
                host = host.replace(':80', '')
            elif ':443' in host and scheme == 'https':
                host = host.replace(':443', '')
            
            # Remove www prefix for consistency
            if host.startswith('www.'):
                host = host[4:]
            
            # Normalize path
            path = parsed.path
            if not path:
                path = '/'
            # Remove trailing slash except for root
            if path != '/' and path.endswith('/'):
                path = path.rstrip('/')
            
            # Filter and sort query parameters
            query_params = parse_qs(parsed.query, keep_blank_values=True)
            filtered_params = {
                k: v for k, v in query_params.items()
                if k.lower() not in cls.TRACKING_PARAMS
            }
            sorted_query = urlencode(sorted(filtered_params.items()), doseq=True)
            
            # Reconstruct URL
            return urlunparse((scheme, host, path, '', sorted_query, ''))
            
        except Exception as e:
            logger.warning(f"URL canonicalization failed for {url}: {e}")
            return url


class SimHash:
    """
    SimHash implementation for near-duplicate text detection.
    
    Locality-sensitive hash that maps similar documents to similar hashes.
    Near-duplicates can be found by comparing Hamming distances.
    """
    
    def __init__(self, num_bits: int = 64):
        if num_bits is None:
            raise ValueError("num_bits is required")
        
        self.num_bits = num_bits
    
    def _tokenize(self, text: str) -> List[str]:
        """Extracts word n-grams from text."""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Generate 3-grams for better precision
        ngrams = []
        for i in range(len(words) - 2):
            ngrams.append(' '.join(words[i:i+3]))
        
        # Also include individual words for short texts
        ngrams.extend(words)
        
        return ngrams
    
    def _hash_token(self, token: str) -> int:
        """Hashes a token to a 64-bit integer."""
        h = hashlib.md5(token.encode('utf-8')).digest()
        return struct.unpack('<Q', h[:8])[0]
    
    def compute(self, text: str) -> int:
        """Computes SimHash for text."""
        if text is None:
            raise ValueError("text is required")
        
        tokens = self._tokenize(text)
        
        if not tokens:
            return 0
        
        # Initialize vector of sums
        v = [0] * self.num_bits
        
        # Add weighted contributions from each token
        for token in tokens:
            h = self._hash_token(token)
            for i in range(self.num_bits):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # Convert to fingerprint
        fingerprint = 0
        for i in range(self.num_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Computes Hamming distance between two hashes."""
        xor = hash1 ^ hash2
        return bin(xor).count('1')
    
    def is_near_duplicate(self, hash1: int, hash2: int, threshold: int = 3) -> bool:
        """Checks if two hashes indicate near-duplicates."""
        if threshold is None:
            raise ValueError("threshold is required")
        return self.hamming_distance(hash1, hash2) <= threshold


class MinHash:
    """
    MinHash implementation for Jaccard similarity estimation.
    
    Estimates set similarity efficiently using random permutations.
    """
    
    def __init__(self, num_perm: int = 128):
        if num_perm is None:
            raise ValueError("num_perm is required")
        
        self.num_perm = num_perm
        # Pre-compute random hash parameters
        self._a = np.random.randint(1, 2**31 - 1, size=num_perm, dtype=np.uint64)
        self._b = np.random.randint(0, 2**31 - 1, size=num_perm, dtype=np.uint64)
        self._prime = 2**61 - 1  # Mersenne prime
    
    def _hash_func(self, x: int, i: int) -> int:
        """Computes hash function i for value x."""
        return ((self._a[i] * x + self._b[i]) % self._prime)
    
    def compute(self, text: str) -> np.ndarray:
        """Computes MinHash signature for text."""
        if text is None:
            raise ValueError("text is required")
        
        # Tokenize into shingles (character 5-grams)
        shingles = set()
        text_lower = text.lower()
        for i in range(len(text_lower) - 4):
            shingle = text_lower[i:i+5]
            # Hash shingle to integer
            h = struct.unpack('<I', hashlib.md5(shingle.encode()).digest()[:4])[0]
            shingles.add(h)
        
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.uint64)
        
        # Compute minimum hash for each permutation
        signature = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        
        for shingle in shingles:
            for i in range(self.num_perm):
                h = self._hash_func(shingle, i)
                if h < signature[i]:
                    signature[i] = h
        
        return signature
    
    @staticmethod
    def jaccard_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimates Jaccard similarity from MinHash signatures."""
        return np.mean(sig1 == sig2)
    
    def is_near_duplicate(self, sig1: np.ndarray, sig2: np.ndarray, threshold: float = 0.5) -> bool:
        """Checks if two signatures indicate near-duplicates."""
        if threshold is None:
            raise ValueError("threshold is required")
        return self.jaccard_similarity(sig1, sig2) >= threshold


@dataclass
class DeduplicationResult:
    """Result of deduplication check."""
    is_duplicate: bool
    reason: Optional[str] = None
    duplicate_of: Optional[str] = None
    similarity: Optional[float] = None


class Deduplicator:
    """
    Multi-strategy deduplication engine.
    
    Combines URL canonicalization, SimHash, MinHash, and embedding similarity
    for comprehensive near-duplicate detection.
    """
    
    def __init__(
        self,
        expected_documents: int = 1_000_000,
        simhash_threshold: Optional[int] = None,
        minhash_threshold: Optional[float] = None,
        embedding_threshold: Optional[float] = None
    ):
        if expected_documents is None:
            raise ValueError("expected_documents is required")
        
        # Load thresholds from config if not provided
        self.simhash_threshold = simhash_threshold or config.get(
            "deduplication.simhash_threshold"
        )
        self.minhash_threshold = minhash_threshold or config.get(
            "deduplication.minhash_jaccard_threshold"
        )
        self.embedding_threshold = embedding_threshold or config.get(
            "deduplication.embedding_similarity_threshold"
        )
        
        # Initialize components
        self.url_filter = BloomFilter(expected_documents)
        self.simhash = SimHash()
        self.minhash = MinHash()
        
        # Index for fast SimHash lookup (maps hash prefixes to doc IDs)
        self._simhash_index: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        self._minhash_signatures: Dict[str, np.ndarray] = {}
        
        # Statistics
        self._stats = {
            'url_duplicates': 0,
            'simhash_duplicates': 0,
            'minhash_duplicates': 0,
            'embedding_duplicates': 0,
            'unique_documents': 0,
        }
        
        logger.info(f"Deduplicator initialized: simhash_th={self.simhash_threshold}, "
                   f"minhash_th={self.minhash_threshold}, embed_th={self.embedding_threshold}")
    
    def check_url(self, url: str) -> DeduplicationResult:
        """
        Checks if URL has been seen before.
        
        Uses canonical URL form and Bloom filter for fast lookup.
        """
        if url is None:
            raise ValueError("url is required")
        
        canonical = URLCanonicalizer.canonicalize(url)
        
        if canonical in self.url_filter:
            self._stats['url_duplicates'] += 1
            return DeduplicationResult(
                is_duplicate=True,
                reason='url_duplicate',
                duplicate_of=canonical
            )
        
        return DeduplicationResult(is_duplicate=False)
    
    def register_url(self, url: str):
        """Registers a URL as processed."""
        if url is None:
            raise ValueError("url is required")
        
        canonical = URLCanonicalizer.canonicalize(url)
        self.url_filter.add(canonical)
    
    def check_content(
        self,
        doc_id: str,
        text: str,
        use_simhash: bool = True,
        use_minhash: bool = True
    ) -> DeduplicationResult:
        """
        Checks if content is a near-duplicate of existing documents.
        
        Args:
            doc_id: Document identifier
            text: Document text content
            use_simhash: Whether to check SimHash
            use_minhash: Whether to check MinHash
            
        Returns:
            DeduplicationResult indicating if duplicate and why
        """
        if doc_id is None:
            raise ValueError("doc_id is required")
        if text is None:
            raise ValueError("text is required")
        
        # SimHash check (fast)
        if use_simhash:
            simhash_val = self.simhash.compute(text)
            
            # Check against existing hashes using prefix bucketing
            prefix = simhash_val >> 48  # Top 16 bits as bucket key
            
            for existing_id, existing_hash in self._simhash_index.get(prefix, []):
                if self.simhash.is_near_duplicate(
                    simhash_val, existing_hash, self.simhash_threshold
                ):
                    self._stats['simhash_duplicates'] += 1
                    distance = SimHash.hamming_distance(simhash_val, existing_hash)
                    return DeduplicationResult(
                        is_duplicate=True,
                        reason='simhash_duplicate',
                        duplicate_of=existing_id,
                        similarity=1 - (distance / 64)
                    )
            
            # Also check nearby buckets (±1 in top bits)
            for delta in [-1, 1]:
                nearby_prefix = prefix + delta
                if nearby_prefix < 0 or nearby_prefix > 65535:
                    continue
                for existing_id, existing_hash in self._simhash_index.get(nearby_prefix, []):
                    if self.simhash.is_near_duplicate(
                        simhash_val, existing_hash, self.simhash_threshold
                    ):
                        self._stats['simhash_duplicates'] += 1
                        distance = SimHash.hamming_distance(simhash_val, existing_hash)
                        return DeduplicationResult(
                            is_duplicate=True,
                            reason='simhash_duplicate',
                            duplicate_of=existing_id,
                            similarity=1 - (distance / 64)
                        )
        
        # MinHash check (more accurate but slower)
        if use_minhash and len(self._minhash_signatures) < 50000:
            # Only use MinHash when signature store is manageable
            minhash_sig = self.minhash.compute(text)
            
            for existing_id, existing_sig in self._minhash_signatures.items():
                similarity = MinHash.jaccard_similarity(minhash_sig, existing_sig)
                if similarity >= self.minhash_threshold:
                    self._stats['minhash_duplicates'] += 1
                    return DeduplicationResult(
                        is_duplicate=True,
                        reason='minhash_duplicate',
                        duplicate_of=existing_id,
                        similarity=similarity
                    )
        
        return DeduplicationResult(is_duplicate=False)
    
    def register_content(self, doc_id: str, text: str):
        """Registers content for future deduplication checks."""
        if doc_id is None:
            raise ValueError("doc_id is required")
        if text is None:
            raise ValueError("text is required")
        
        # Store SimHash
        simhash_val = self.simhash.compute(text)
        prefix = simhash_val >> 48
        self._simhash_index[prefix].append((doc_id, simhash_val))
        
        # Store MinHash (limited to prevent memory issues)
        if len(self._minhash_signatures) < 50000:
            self._minhash_signatures[doc_id] = self.minhash.compute(text)
        
        self._stats['unique_documents'] += 1
    
    def check_embedding(
        self,
        embedding: np.ndarray,
        existing_embeddings: List[np.ndarray]
    ) -> DeduplicationResult:
        """
        Checks for semantic duplicates using embedding similarity.
        
        This is typically done via Qdrant in the writer, but can be
        used standalone for batch processing.
        """
        if embedding is None:
            raise ValueError("embedding is required")
        if existing_embeddings is None:
            raise ValueError("existing_embeddings is required")
        
        if not existing_embeddings:
            return DeduplicationResult(is_duplicate=False)
        
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        # Compute cosine similarity with all existing
        existing = np.array(existing_embeddings)
        existing = existing / (np.linalg.norm(existing, axis=1, keepdims=True) + 1e-10)
        
        similarities = np.dot(existing, embedding)
        max_similarity = np.max(similarities)
        
        if max_similarity >= self.embedding_threshold:
            self._stats['embedding_duplicates'] += 1
            max_idx = np.argmax(similarities)
            return DeduplicationResult(
                is_duplicate=True,
                reason='embedding_duplicate',
                duplicate_of=str(max_idx),
                similarity=float(max_similarity)
            )
        
        return DeduplicationResult(is_duplicate=False)
    
    def get_stats(self) -> Dict[str, int]:
        """Returns deduplication statistics."""
        return {
            **self._stats,
            'url_filter_size': len(self.url_filter),
            'simhash_index_buckets': len(self._simhash_index),
            'minhash_signatures': len(self._minhash_signatures),
        }


def compute_content_hash(text: str) -> str:
    """Computes MD5 hash of text content for exact duplicate detection."""
    if text is None:
        raise ValueError("text is required")
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def compute_simhash(text: str) -> int:
    """Convenience function for computing SimHash."""
    if text is None:
        raise ValueError("text is required")
    return SimHash().compute(text)


def compute_minhash(text: str) -> np.ndarray:
    """Convenience function for computing MinHash signature."""
    if text is None:
        raise ValueError("text is required")
    return MinHash().compute(text)
