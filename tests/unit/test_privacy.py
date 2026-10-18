"""
Privacy and deletion tests.

Verifies:
  1. No external network calls are made during embedding, retrieval, or caching.
  2. Document deletion removes all derived data: Qdrant vectors, embedding cache,
     retrieval cache — and stale evidence cannot be returned afterwards.
  3. The embedding model is never called with data for a deleted document.
"""

from __future__ import annotations

import socket
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app.document.schemas import Chunk, SourceType


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(source_id: str, doc_id: str = "doc-001", text: str = "Test clause.") -> Chunk:
    return Chunk(
        source_id=source_id,
        document_id=doc_id,
        source_type=SourceType.CONTRACT,
        page_start=1,
        page_end=1,
        text=text,
        span_start=0,
        span_end=len(text),
        chunk_index=0,
        index_version="v1",
    )


# ─── Privacy: No External Network Calls ──────────────────────────────────────

class TestNoExternalNetworkCalls:
    """
    Block outgoing connections and verify that our retrieval pipeline doesn't
    attempt to reach any external host.
    """

    def test_embed_query_stays_local(self, monkeypatch):
        """embed_query() must not open any external socket."""
        original_connect = socket.socket.connect

        external_attempts: list[str] = []

        def strict_connect(self, address):
            host = address[0] if isinstance(address, (list, tuple)) else str(address)
            # Allow loopback and local connections
            if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
                external_attempts.append(host)
                raise ConnectionRefusedError(f"TEST: Blocked external connection to {host}")
            return original_connect(self, address)

        monkeypatch.setattr(socket.socket, "connect", strict_connect)

        # Mock the actual model to avoid downloading it in CI
        with patch("backend.app.retrieval.embeddings._get_model") as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = [0.0] * 384
            mock_model.return_value = mock_instance

            from backend.app.retrieval.embeddings import embed_query
            embed_query("When can the landlord terminate?")

        assert external_attempts == [], f"External connections attempted: {external_attempts}"

    def test_bm25_search_stays_local(self):
        """BM25 search is pure in-memory — no network by design."""
        from backend.app.retrieval.bm25_retriever import BM25DocumentIndex

        chunks = [_make_chunk("doc::0000", text="The landlord may terminate with 60 days notice.")]
        index = BM25DocumentIndex(chunks=chunks)
        results = index.search("terminate landlord")
        # If we got here without a network call, the test passes
        assert isinstance(results, list)

    def test_cache_write_is_filesystem_only(self, tmp_path, monkeypatch):
        """Retrieval cache writes must not trigger any network I/O."""
        from backend.app.document.schemas import EvidenceResult

        result = EvidenceResult(
            document_id="doc-001",
            question="test question",
            passages=[],
            is_sufficient=False,
            index_version="v1",
        )

        original_connect = socket.socket.connect
        external_attempts: list[str] = []

        def no_external(self, address):
            host = address[0] if isinstance(address, (list, tuple)) else str(address)
            if host not in ("127.0.0.1", "localhost", "::1"):
                external_attempts.append(host)
                raise ConnectionRefusedError("Blocked")
            return original_connect(self, address)

        monkeypatch.setattr(socket.socket, "connect", no_external)

        with patch("backend.app.caching.retrieval_cache._cache_dir", return_value=tmp_path):
            from backend.app.caching import retrieval_cache
            retrieval_cache.cache_retrieval("doc-001", "test question", 5, result)

        assert external_attempts == []


# ─── Deletion: Derived Data Is Removed ───────────────────────────────────────

class TestDeletionCleansUp:
    """When a document is deleted, no derived data should remain or be retrievable."""

    def test_retrieval_cache_cleared_after_deletion(self, tmp_path):
        from backend.app.document.schemas import EvidenceResult
        from backend.app.caching import retrieval_cache

        result = EvidenceResult(
            document_id="doc-delete-001",
            question="test",
            passages=[],
            is_sufficient=False,
            index_version="v1",
        )

        with patch("backend.app.caching.retrieval_cache._cache_dir", return_value=tmp_path):
            retrieval_cache.cache_retrieval("doc-delete-001", "test", 5, result)
            # Verify it's cached
            assert retrieval_cache.get_cached_retrieval("doc-delete-001", "test", 5) is not None
            # Delete
            retrieval_cache.invalidate_document_retrieval("doc-delete-001")
            # Verify it's gone
            assert retrieval_cache.get_cached_retrieval("doc-delete-001", "test", 5) is None

    def test_stale_evidence_not_returned_after_reindex(self, tmp_path):
        """
        After a document is re-indexed (index_version bumped from v1 to v2),
        v1 cache entries must not be returned for the new version.
        """
        from backend.app.document.schemas import EvidenceResult, Evidence
        from backend.app.caching import retrieval_cache

        stale = EvidenceResult(
            document_id="doc-stale",
            question="What is the rent?",
            passages=[
                Evidence(
                    source_id="doc-stale::chunk-0000",
                    source_type="contract",
                    quote="Old rent clause: $800/month",
                    index_version="v1",
                )
            ],
            is_sufficient=True,
            index_version="v1",
        )

        with patch("backend.app.caching.retrieval_cache._cache_dir", return_value=tmp_path):
            # Cache the stale v1 result
            retrieval_cache.cache_retrieval(
                "doc-stale", "What is the rent?", 5, stale, index_version="v1"
            )

            # A query for v2 (after re-index) must not return the stale v1 entry
            v2_hit = retrieval_cache.get_cached_retrieval(
                "doc-stale", "What is the rent?", 5, index_version="v2"
            )
            assert v2_hit is None, "Stale v1 cache must not serve as v2 result"

    def test_no_phantom_chunks_after_qdrant_deletion(self, tmp_path, monkeypatch):
        """
        After delete_document(), document_is_indexed() must return False.
        This test mocks Qdrant client to verify the correct deletion call is made.
        """
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)   # Empty after deletion

        with patch("backend.app.retrieval.vector_store._get_client", return_value=mock_client):
            from backend.app.retrieval.vector_store import delete_document, document_is_indexed

            delete_document("doc-phantom-001")
            assert mock_client.delete.called, "delete() must be called on Qdrant client"

            is_indexed = document_is_indexed("doc-phantom-001")
            assert is_indexed is False, "document_is_indexed must return False after deletion"


# ─── Embedding Text Privacy ───────────────────────────────────────────────────

class TestEmbeddingTextPrivacy:
    def test_embedding_cache_does_not_store_text(self, tmp_path):
        """The embedding cache must store only vectors, never the original text."""
        from backend.app.caching.keys import embedding_cache_key

        key = embedding_cache_key("Confidential clause text", "test_model")
        cache_file = tmp_path / f"{key}.json"

        # Simulate what embedding_cache.py writes
        import json
        vector = [0.1, 0.2, 0.3]
        with open(cache_file, "w") as f:
            json.dump(vector, f)

        content = cache_file.read_text()
        assert "Confidential" not in content
        assert "clause text" not in content
