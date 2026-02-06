"""Tests for the modular mapping pipeline, protocols, and components."""

import pytest
import numpy as np

from mapping.protocols import Projector, Clusterer, AxisScorer
from mapping.axis_scorers import DomainAuthorityAxisScorer
from mapping.pipeline import MappingPipeline
from mapping.registry import ComponentRegistry


# ── Fake Components ────────────────────────────────────────────

class FakeProjector(Projector):
    """Returns first 2 columns of input as coordinates."""

    @property
    def name(self):
        return "fake"

    def fit(self, embeddings, **kwargs):
        pass

    def transform(self, embeddings):
        return embeddings[:, :2]


class FakeClusterer(Clusterer):
    """Assigns all points to cluster 0."""

    @property
    def name(self):
        return "fake"

    def fit_predict(self, coordinates):
        return np.zeros(len(coordinates), dtype=int)


class FakeAxisScorer(AxisScorer):
    """Returns a fixed importance score."""

    @property
    def name(self):
        return "fake"

    def compute(self, domain, quality_score, content_type=None,
                inbound_links=None, citations=None):
        return 0.75


# ── DomainAuthorityAxisScorer ──────────────────────────────────

class TestDomainAuthorityAxisScorer:
    def test_known_domain(self):
        s = DomainAuthorityAxisScorer()
        score = s.compute("arxiv.org", 0.8)
        assert score > 0.7

    def test_unknown_domain(self):
        s = DomainAuthorityAxisScorer()
        score = s.compute("randomsite.com", 0.5)
        assert 0.0 < score < 1.0

    def test_edu_domain(self):
        s = DomainAuthorityAxisScorer()
        score = s.compute("stanford.edu", 0.7)
        assert score > 0.6

    def test_domain_required(self):
        s = DomainAuthorityAxisScorer()
        with pytest.raises(ValueError):
            s.compute(None, 0.5)

    def test_quality_required(self):
        s = DomainAuthorityAxisScorer()
        with pytest.raises(ValueError):
            s.compute("example.com", None)

    def test_caching(self):
        s = DomainAuthorityAxisScorer()
        s1 = s.compute("example.com", 0.5)
        s2 = s.compute("example.com", 0.5)
        assert s1 == s2
        assert f"example.com:0.50" in s._cache

    def test_with_inbound_links(self):
        s = DomainAuthorityAxisScorer()
        base = s.compute("example.com", 0.5)
        s._cache.clear()
        with_links = s.compute("example.com", 0.5, inbound_links=1000)
        # High inbound links should boost score
        assert with_links > base

    def test_with_citations(self):
        s = DomainAuthorityAxisScorer()
        base = s.compute("example.com", 0.5)
        s._cache.clear()
        with_citations = s.compute("example.com", 0.5, citations=100)
        assert with_citations > base


# ── MappingPipeline ────────────────────────────────────────────

class TestMappingPipeline:
    def test_project_with_fake(self):
        pipe = MappingPipeline(projector=FakeProjector())
        data = np.random.rand(10, 5)
        coords = pipe.project(data)
        assert coords.shape == (10, 2)

    def test_cluster_with_fake(self):
        pipe = MappingPipeline(clusterer=FakeClusterer())
        coords = np.random.rand(10, 2)
        labels = pipe.cluster(coords)
        assert len(labels) == 10
        assert all(l == 0 for l in labels)

    def test_score_importance_with_fake(self):
        pipe = MappingPipeline(axis_scorer=FakeAxisScorer())
        score = pipe.score_importance("example.com", 0.5)
        assert score == 0.75

    def test_full_pipeline_with_fakes(self):
        pipe = MappingPipeline(
            projector=FakeProjector(),
            clusterer=FakeClusterer(),
            axis_scorer=FakeAxisScorer(),
        )
        data = np.random.rand(20, 5)
        coords = pipe.project(data)
        labels = pipe.cluster(coords)
        score = pipe.score_importance("test.com", 0.6)
        assert coords.shape == (20, 2)
        assert len(labels) == 20
        assert score == 0.75


# ── ComponentRegistry ──────────────────────────────────────────

class TestComponentRegistry:
    def test_register_projector(self):
        reg = ComponentRegistry()
        reg.register(FakeProjector())
        assert reg.get_projector("fake") is not None
        assert "fake" in reg.projector_names

    def test_register_clusterer(self):
        reg = ComponentRegistry()
        reg.register(FakeClusterer())
        assert reg.get_clusterer("fake") is not None

    def test_register_axis_scorer(self):
        reg = ComponentRegistry()
        reg.register(FakeAxisScorer())
        assert reg.get_axis_scorer("fake") is not None

    def test_lookup_missing(self):
        reg = ComponentRegistry()
        assert reg.get_projector("nonexistent") is None

    def test_register_unknown_type(self):
        reg = ComponentRegistry()
        with pytest.raises(TypeError):
            reg.register("not_a_component")
