"""Integration tests for common/repositories.py using in-memory SQLite (via DI)."""

import json
import pytest

from common.models import DocumentCreate, CoverageMetrics, GoldenSetEntry, DumpJob


class TestDocumentRepository:
    def _make_doc(self, doc_id="doc-1", url="https://example.com/page"):
        return DocumentCreate(
            id=doc_id,
            url=url,
            title="Test Document",
            summary="A test summary",
            content_hash="abc123",
            domain="example.com",
            content_type="default",
            quality_score=0.75,
            quality_components=json.dumps({"writing_quality": 0.8}),
            quality_profile_used="default",
            raw_html_hash="def456",
            novelty_distance=0.15,
            source_phase="batch",
            content_length=500,
        )

    def test_insert_and_get_by_id(self, doc_repo):
        doc = self._make_doc()
        doc_repo.insert(doc)
        result = doc_repo.get_by_id("doc-1")
        assert result is not None
        assert result.url == "https://example.com/page"
        assert result.quality_score == 0.75

    def test_get_by_id_not_found(self, doc_repo):
        assert doc_repo.get_by_id("nonexistent") is None

    def test_get_list(self, doc_repo):
        for i in range(5):
            doc_repo.insert(self._make_doc(f"doc-{i}", f"https://example.com/{i}"))
        items = doc_repo.get_list(limit=3, offset=0)
        assert len(items) == 3

    def test_get_count(self, doc_repo):
        for i in range(3):
            doc_repo.insert(self._make_doc(f"doc-{i}", f"https://example.com/{i}"))
        assert doc_repo.get_count() == 3

    def test_insert_batch(self, doc_repo):
        docs = [self._make_doc(f"batch-{i}", f"https://example.com/batch/{i}") for i in range(10)]
        doc_repo.insert_batch(docs)
        assert doc_repo.get_count() == 10

    def test_search_text(self, doc_repo):
        doc_repo.insert(self._make_doc("doc-1", "https://example.com/ml"))
        # title-based search
        results = doc_repo.search_text("Test Document", limit=10)
        assert len(results) >= 1

    def test_get_random_ids(self, doc_repo):
        for i in range(5):
            doc_repo.insert(self._make_doc(f"doc-{i}", f"https://example.com/{i}"))
        ids = doc_repo.get_random_ids(3)
        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)

    def test_update_mapping(self, doc_repo):
        doc_repo.insert(self._make_doc("doc-1"))
        doc_repo.update_mapping("doc-1", cluster_id=5, importance_score=0.9)
        result = doc_repo.get_by_id("doc-1")
        assert result.cluster_id == 5
        assert result.importance_score == 0.9


class TestMetricsRepository:
    def test_insert_and_roundtrip(self, metrics_repo):
        metrics = CoverageMetrics(
            topic_gini=0.4,
            domain_gini=0.3,
            orphan_rate=0.1,
            high_quality_orphans=5,
            cluster_count=10,
            largest_cluster_pct=0.2,
        )
        metrics_repo.insert_coverage_metrics(metrics)
        # Verify no exception on insert — retrieval is via health.py queries


class TestGoldenSetRepository:
    def test_upsert_and_get_for_training(self, golden_set_repo):
        entry = GoldenSetEntry(
            url="https://example.com/good",
            label="5",
            content_type="default",
            notes="test",
            raw_metrics=json.dumps({"writing_quality": 0.9}),
            version=1,
            domain="example.com",
        )
        golden_set_repo.upsert(entry)
        rows = golden_set_repo.get_for_training("default")
        assert len(rows) == 1
        assert rows[0][0] == "5"  # label

    def test_upsert_idempotent(self, golden_set_repo):
        entry = GoldenSetEntry(
            url="https://example.com/good",
            label="5",
            content_type="default",
            notes="first",
            raw_metrics=json.dumps({}),
            version=1,
            domain="example.com",
        )
        golden_set_repo.upsert(entry)
        entry.notes = "updated"
        golden_set_repo.upsert(entry)
        rows = golden_set_repo.get_for_training("default")
        assert len(rows) == 1


class TestJobRepository:
    def test_register_and_finalize(self, job_repo):
        job = DumpJob(id="job-1", dump_name="test.tar", dump_path="/tmp/test.tar")
        job_repo.register_job(job)
        job_repo.finalize_job("job-1", status="completed", total_urls=100, filtered_urls=50)
        # No exception = success
