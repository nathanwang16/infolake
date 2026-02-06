"""Tests for phase1_offline/deduplication.py — pure functions and data structures."""

import numpy as np
import pytest

from phase1_offline.deduplication import (
    BloomFilter,
    URLCanonicalizer,
    SimHash,
    MinHash,
    compute_content_hash,
)


# ── BloomFilter ───────────────────────────────────────────────

class TestBloomFilter:
    def test_add_and_contains(self):
        bf = BloomFilter(1000)
        bf.add("hello")
        assert "hello" in bf
        assert "world" not in bf

    def test_no_false_negatives(self):
        bf = BloomFilter(10000)
        urls = [f"https://example.com/page/{i}" for i in range(500)]
        for url in urls:
            bf.add(url)
        for url in urls:
            assert url in bf

    def test_len(self):
        bf = BloomFilter(1000)
        bf.add("a")
        bf.add("b")
        assert len(bf) == 2


# ── URLCanonicalizer ─────────────────────────────────────────

class TestURLCanonicalizer:
    def test_lowercase_host(self):
        result = URLCanonicalizer.canonicalize("https://EXAMPLE.COM/Page")
        assert "example.com" in result

    def test_removes_www(self):
        result = URLCanonicalizer.canonicalize("https://www.example.com/page")
        assert result == "https://example.com/page"

    def test_removes_tracking_params(self):
        result = URLCanonicalizer.canonicalize(
            "https://example.com/page?utm_source=twitter&id=5"
        )
        assert "utm_source" not in result
        assert "id=5" in result

    def test_removes_default_port(self):
        result = URLCanonicalizer.canonicalize("https://example.com:443/page")
        assert ":443" not in result

    def test_sorts_query_params(self):
        result = URLCanonicalizer.canonicalize(
            "https://example.com/?z=1&a=2"
        )
        assert result.index("a=2") < result.index("z=1")

    def test_strips_trailing_slash(self):
        result = URLCanonicalizer.canonicalize("https://example.com/page/")
        assert result.endswith("/page")

    def test_root_keeps_slash(self):
        result = URLCanonicalizer.canonicalize("https://example.com/")
        assert result.endswith("/")

    def test_raises_on_none(self):
        with pytest.raises(ValueError):
            URLCanonicalizer.canonicalize(None)


# ── SimHash ───────────────────────────────────────────────────

class TestSimHash:
    def test_identical_texts_same_hash(self):
        sh = SimHash()
        h1 = sh.compute("The quick brown fox jumps over the lazy dog")
        h2 = sh.compute("The quick brown fox jumps over the lazy dog")
        assert h1 == h2

    def test_similar_texts_low_hamming(self):
        sh = SimHash()
        h1 = sh.compute("The quick brown fox jumps over the lazy dog")
        h2 = sh.compute("The quick brown fox jumps over a lazy dog")
        assert SimHash.hamming_distance(h1, h2) < 10

    def test_different_texts_high_hamming(self):
        sh = SimHash()
        h1 = sh.compute("Quantum mechanics describes particle behavior at atomic scale")
        h2 = sh.compute("A chocolate cake recipe requires flour eggs and sugar")
        assert SimHash.hamming_distance(h1, h2) > 10

    def test_is_near_duplicate(self):
        sh = SimHash()
        h1 = sh.compute("The quick brown fox jumps over the lazy dog")
        h2 = sh.compute("The quick brown fox jumps over the lazy dog")
        assert sh.is_near_duplicate(h1, h2, threshold=3)

    def test_empty_text(self):
        sh = SimHash()
        assert sh.compute("") == 0

    def test_raises_on_none(self):
        sh = SimHash()
        with pytest.raises(ValueError):
            sh.compute(None)


# ── MinHash ───────────────────────────────────────────────────

class TestMinHash:
    def test_identical_texts_high_similarity(self):
        mh = MinHash(num_perm=128)
        sig1 = mh.compute("The quick brown fox jumps over the lazy dog")
        sig2 = mh.compute("The quick brown fox jumps over the lazy dog")
        similarity = MinHash.jaccard_similarity(sig1, sig2)
        assert similarity == 1.0

    def test_different_texts_low_similarity(self):
        mh = MinHash(num_perm=128)
        sig1 = mh.compute("The quick brown fox jumps over the lazy dog")
        sig2 = mh.compute("Quantum computing leverages superposition and entanglement")
        similarity = MinHash.jaccard_similarity(sig1, sig2)
        assert similarity < 0.5

    def test_signature_shape(self):
        mh = MinHash(num_perm=64)
        sig = mh.compute("Hello world this is a test document")
        assert sig.shape == (64,)

    def test_raises_on_none(self):
        mh = MinHash()
        with pytest.raises(ValueError):
            mh.compute(None)


# ── compute_content_hash ──────────────────────────────────────

class TestContentHash:
    def test_deterministic(self):
        assert compute_content_hash("hello") == compute_content_hash("hello")

    def test_different_inputs(self):
        assert compute_content_hash("hello") != compute_content_hash("world")

    def test_raises_on_none(self):
        with pytest.raises(ValueError):
            compute_content_hash(None)
