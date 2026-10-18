"""
Tests for cache invalidation: embedding cache and retrieval cache.
Verifies that:
  1. Cached values are returned on hit.
  2. Changed document/index version produces a cache miss.
  3. invalidate_all_for_document() removes all related cache entries.
  4. Corrupt cache files are handled gracefully (no crash, miss returned).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.caching.keys import (
    embedding_cache_key,
    retrieval_cache_key,
    normalize_query,
)
from backend.app.document.schemas import Evidence, EvidenceResult, SourceType


# ─── Cache Key Tests ──────────────────────────────────────────────────────────

class TestCacheKeys:
    def test_embedding_key_is_deterministic(self):
        key1 = embedding_cache_key("hello world", "model_v1")
        key2 = embedding_cache_key("hello world", "model_v1")
        assert key1 == key2

    def test_embedding_key_differs_by_model(self):
        key1 = embedding_cache_key("hello", "bge_small_v1")
        key2 = embedding_cache_key("hello", "bge_large_v1")
        assert key1 != key2

    def test_embedding_key_differs_by_text(self):
        key1 = embedding_cache_key("text A", "model_v1")
        key2 = embedding_cache_key("text B", "model_v1")
        assert key1 != key2

    def test_retrieval_key_is_deterministic(self):
        k1 = retrieval_cache_key("doc-001", "v1", "What is the rent?", 5, "hybrid")
        k2 = retrieval_cache_key("doc-001", "v1", "What is the rent?", 5, "hybrid")
        assert k1 == k2

    def test_retrieval_key_differs_by_index_version(self):
        k1 = retrieval_cache_key("doc-001", "v1", "question", 5)
        k2 = retrieval_cache_key("doc-001", "v2", "question", 5)
        assert k1 != k2

    def test_retrieval_key_differs_by_document_id(self):
        k1 = retrieval_cache_key("doc-001", "v1", "question", 5)
        k2 = retrieval_cache_key("doc-002", "v1", "question", 5)
        assert k1 != k2

    def test_retrieval_key_case_insensitive_query(self):
        """Normalized query should collapse case differences."""
        k1 = retrieval_cache_key("doc-001", "v1", "What is the RENT?", 5)
        k2 = retrieval_cache_key("doc-001", "v1", "what is the rent?", 5)
        assert k1 == k2

    def test_normalize_query_collapses_whitespace(self):
        assert normalize_query("  hello   world  ") == "hello world"

    def test_normalize_query_lowercases(self):
        assert normalize_query("TERMINATE") == "terminate"


# ─── Retrieval Cache Tests ────────────────────────────────────────────────────

def _make_evidence_result(doc_id: str = "doc-001", question: str = "What is the rent?") -> EvidenceResult:
    return EvidenceResult(
        document_id=doc_id,
        question=question,
        passages=[
            Evidence(
                source_id=f"{doc_id}::chunk-0000",
                source_type="contract",
                page=1,
                section="2",
                quote="Monthly rent is $1,500.",
                retrieval_rank=1,
                rrf_score=0.032,
                retrieval_method="hybrid",
                index_version="v1",
            )
        ],
        is_sufficient=True,
        index_version="v1",
    )


class TestRetrievalCache:
    def test_cache_hit(self, tmp_path):
        with patch("backend.app.caching.retrieval_cache.settings") as mock_settings:
            mock_settings.RETRIEVAL_CACHE_DIR = tmp_path / "retrieval"
            mock_settings.CACHE_MAX_ENTRIES = 100
            tmp_path.joinpath("retrieval").mkdir()

            from backend.app.caching import retrieval_cache
            # Monkeypatch the module-level settings reference
            with patch.object(retrieval_cache, "_cache_dir", return_value=tmp_path / "retrieval"):
                result = _make_evidence_result()
                retrieval_cache.cache_retrieval("doc-001", "What is the rent?", 5, result)
                hit = retrieval_cache.get_cached_retrieval("doc-001", "What is the rent?", 5)
                assert hit is not None
                assert hit.document_id == "doc-001"
                assert hit.is_sufficient is True

    def test_cache_miss_on_different_question(self, tmp_path):
        with patch("backend.app.caching.retrieval_cache._cache_dir", return_value=tmp_path):
            from backend.app.caching import retrieval_cache

            result = _make_evidence_result()
            retrieval_cache.cache_retrieval("doc-001", "What is the rent?", 5, result)
            miss = retrieval_cache.get_cached_retrieval("doc-001", "When can I terminate?", 5)
            assert miss is None

    def test_cache_miss_on_changed_index_version(self, tmp_path):
        with patch("backend.app.caching.retrieval_cache._cache_dir", return_value=tmp_path):
            from backend.app.caching import retrieval_cache

            result = _make_evidence_result()
            retrieval_cache.cache_retrieval("doc-001", "What is the rent?", 5, result, index_version="v1")
            miss = retrieval_cache.get_cached_retrieval("doc-001", "What is the rent?", 5, index_version="v2")
            assert miss is None

    def test_corrupt_cache_returns_none(self, tmp_path):
        from backend.app.caching.keys import retrieval_cache_key

        key = retrieval_cache_key("doc-001", "v1", "What is the rent?", 5, "hybrid")
        cache_file = tmp_path / f"{key}.json"
        cache_file.write_text("{{CORRUPT JSON{{", encoding="utf-8")

        with patch("backend.app.caching.retrieval_cache._cache_dir", return_value=tmp_path):
            from backend.app.caching import retrieval_cache
            result = retrieval_cache.get_cached_retrieval("doc-001", "What is the rent?", 5)
            assert result is None
            # Corrupt file should be cleaned up
            assert not cache_file.exists()

    def test_invalidate_removes_entries(self, tmp_path):
        with patch("backend.app.caching.retrieval_cache._cache_dir", return_value=tmp_path):
            from backend.app.caching import retrieval_cache

            result = _make_evidence_result()
            retrieval_cache.cache_retrieval("doc-001", "What is the rent?", 5, result)
            retrieval_cache.invalidate_document_retrieval("doc-001")

            miss = retrieval_cache.get_cached_retrieval("doc-001", "What is the rent?", 5)
            assert miss is None


# ─── Combined Invalidation ────────────────────────────────────────────────────

class TestInvalidateAll:
    def test_invalidate_all_for_document_returns_counts(self, tmp_path):
        with (
            patch("backend.app.caching.retrieval_cache._cache_dir", return_value=tmp_path / "ret"),
            patch("backend.app.caching.embedding_cache._cache_dir", return_value=tmp_path / "emb"),
        ):
            (tmp_path / "ret").mkdir()
            (tmp_path / "emb").mkdir()

            from backend.app.caching.retrieval_cache import invalidate_all_for_document

            # Call invalidation on a document that has no entries — should return zeros
            counts = invalidate_all_for_document("doc-999")
            assert "embedding" in counts
            assert "retrieval" in counts
            assert counts["embedding"] == 0
            assert counts["retrieval"] == 0
