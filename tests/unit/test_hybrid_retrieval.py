"""
Tests for hybrid retrieval: RRF fusion, BM25 retriever, deduplication, and ranker.
These are pure unit tests — no real Qdrant or embedding model is invoked.
External dependencies (sentence-transformers, qdrant-client) are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from backend.app.document.schemas import Chunk, SourceType
from backend.app.retrieval.ranker import (
    reciprocal_rank_fusion,
    deduplicate_results,
    _spans_overlap_ratio,
    rank_and_deduplicate,
)
from backend.app.retrieval.bm25_retriever import (
    BM25DocumentIndex,
    expand_query,
    _tokenize,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(
    source_id: str,
    text: str,
    doc_id: str = "doc-001",
    span_start: int = 0,
    span_end: int | None = None,
    clause_id: str | None = None,
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        source_id=source_id,
        document_id=doc_id,
        source_type=SourceType.CONTRACT,
        page_start=1,
        page_end=1,
        clause_id=clause_id,
        text=text,
        span_start=span_start,
        span_end=span_end if span_end is not None else span_start + len(text),
        chunk_index=chunk_index,
    )


# ─── BM25 Tokenizer and Query Expansion ──────────────────────────────────────

class TestBM25Tokenizer:
    def test_tokenize_lowercase(self):
        tokens = _tokenize("TERMINATE the LEASE Agreement")
        assert all(t == t.lower() for t in tokens)

    def test_tokenize_removes_punctuation(self):
        tokens = _tokenize("clause 4.1: rent ($1,500)")
        assert "clause" in tokens
        assert "rent" in tokens
        # Punctuation tokens should not appear
        assert "(" not in tokens
        assert ":" not in tokens

    def test_tokenize_empty(self):
        assert _tokenize("") == []

    def test_expand_query_adds_synonyms(self):
        expanded = expand_query("terminate the lease")
        assert "termination" in expanded
        assert "notice" in expanded

    def test_expand_query_deposit(self):
        expanded = expand_query("security deposit")
        assert "refund" in expanded or "bond" in expanded

    def test_expand_query_no_expansion_for_unknown(self):
        expanded = expand_query("arbitration clause")
        # Should still contain the original words
        assert "arbitration" in expanded
        assert "clause" in expanded


# ─── BM25DocumentIndex ────────────────────────────────────────────────────────

class TestBM25DocumentIndex:
    CHUNKS = [
        _make_chunk("doc::0000", "The landlord may terminate the lease with 60 days notice.", chunk_index=0),
        _make_chunk("doc::0001", "Security deposit of $3,000 is required upon signing.", chunk_index=1),
        _make_chunk("doc::0002", "Monthly rent is $1,500 due on the first of each month.", chunk_index=2),
        _make_chunk("doc::0003", "No pets are allowed without prior written consent.", chunk_index=3),
        _make_chunk("doc::0004", "The tenant must maintain the property in good condition.", chunk_index=4),
    ]

    def test_search_returns_relevant_results(self):
        index = BM25DocumentIndex(chunks=self.CHUNKS)
        results = index.search("terminate landlord notice")
        assert results, "Expected results for termination query"
        texts = [c.text for c, _ in results]
        assert any("terminate" in t.lower() for t in texts)

    def test_search_deposit_query(self):
        index = BM25DocumentIndex(chunks=self.CHUNKS)
        results = index.search("security deposit amount")
        assert results
        top_text = results[0][0].text.lower()
        assert "deposit" in top_text or "security" in top_text

    def test_search_top_k_limit(self):
        index = BM25DocumentIndex(chunks=self.CHUNKS)
        results = index.search("landlord tenant agreement", top_k=2)
        assert len(results) <= 2

    def test_empty_corpus_returns_empty(self):
        index = BM25DocumentIndex(chunks=[])
        results = index.search("termination clause")
        assert results == []

    def test_scores_are_sorted_descending(self):
        index = BM25DocumentIndex(chunks=self.CHUNKS)
        results = index.search("terminate landlord notice", top_k=5, min_score=0.0)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

class TestRRF:
    def test_rrf_merges_lists(self):
        chunk_a = _make_chunk("doc::0000", "Termination clause text", chunk_index=0)
        chunk_b = _make_chunk("doc::0001", "Security deposit text", chunk_index=1)
        chunk_c = _make_chunk("doc::0002", "Rent payment terms", chunk_index=2)

        vector_results = [(chunk_a, 0.95), (chunk_b, 0.80)]
        bm25_results = [(chunk_b, 8.5), (chunk_c, 7.1)]

        merged = reciprocal_rank_fusion(vector_results, bm25_results, top_k=5)
        assert len(merged) == 3
        ids = [c.source_id for c, _, _ in merged]
        assert "doc::0000" in ids
        assert "doc::0001" in ids
        assert "doc::0002" in ids

    def test_rrf_hybrid_method_for_overlapping(self):
        chunk = _make_chunk("doc::0000", "Clause text")
        vector_results = [(chunk, 0.9)]
        bm25_results = [(chunk, 5.0)]
        merged = reciprocal_rank_fusion(vector_results, bm25_results)
        assert merged[0][2] == "hybrid"

    def test_rrf_vector_only_method(self):
        chunk = _make_chunk("doc::0000", "Clause text")
        merged = reciprocal_rank_fusion([(chunk, 0.9)], [], top_k=1)
        assert merged[0][2] == "vector"

    def test_rrf_bm25_only_method(self):
        chunk = _make_chunk("doc::0000", "Clause text")
        merged = reciprocal_rank_fusion([], [(chunk, 5.0)], top_k=1)
        assert merged[0][2] == "bm25"

    def test_rrf_empty_inputs(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_rrf_score_higher_when_in_both_lists(self):
        chunk_a = _make_chunk("doc::0000", "Clause A")
        chunk_b = _make_chunk("doc::0001", "Clause B")
        # chunk_a appears in both lists; chunk_b only in vector
        vector_results = [(chunk_a, 0.9), (chunk_b, 0.85)]
        bm25_results = [(chunk_a, 7.0)]
        merged = reciprocal_rank_fusion(vector_results, bm25_results)
        # chunk_a should be ranked first because it scores in both lists
        assert merged[0][0].source_id == "doc::0000"

    def test_rrf_respects_top_k(self):
        chunks = [_make_chunk(f"doc::000{i}", f"text {i}") for i in range(10)]
        vector_results = [(c, float(10 - i)) for i, c in enumerate(chunks)]
        merged = reciprocal_rank_fusion(vector_results, [], top_k=3)
        assert len(merged) <= 3


# ─── Deduplication ────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_no_overlap_keeps_all(self):
        chunks = [
            (_make_chunk("doc::0000", "First clause", span_start=0, span_end=100), 0.5, "hybrid"),
            (_make_chunk("doc::0001", "Second clause", span_start=200, span_end=300), 0.4, "hybrid"),
        ]
        result = deduplicate_results(chunks)
        assert len(result) == 2

    def test_full_overlap_keeps_higher_rank(self):
        """Two chunks with identical spans (100% overlap) → keep the first (higher ranked)."""
        text = "The landlord may terminate the lease."
        chunk_a = _make_chunk("doc::0000", text, span_start=0, span_end=len(text))
        chunk_b = _make_chunk("doc::0001", text, span_start=0, span_end=len(text))
        ranked = [(chunk_a, 0.6, "hybrid"), (chunk_b, 0.3, "hybrid")]
        result = deduplicate_results(ranked)
        assert len(result) == 1
        assert result[0][0].source_id == "doc::0000"

    def test_partial_overlap_below_threshold_keeps_both(self):
        """20% overlap is below the 50% threshold → both kept."""
        chunk_a = _make_chunk("doc::0000", "A" * 100, span_start=0, span_end=100)
        chunk_b = _make_chunk("doc::0001", "B" * 100, span_start=80, span_end=180)
        ranked = [(chunk_a, 0.9, "vector"), (chunk_b, 0.8, "vector")]
        result = deduplicate_results(ranked, overlap_threshold=0.5)
        assert len(result) == 2

    def test_partial_overlap_above_threshold_removes_lower(self):
        """70% overlap is above the 50% threshold → remove lower-ranked."""
        chunk_a = _make_chunk("doc::0000", "A" * 100, span_start=0, span_end=100)
        chunk_b = _make_chunk("doc::0001", "B" * 100, span_start=30, span_end=130)
        # Overlap region: 30-100 = 70 chars; shorter span = 100 → ratio = 0.70
        ranked = [(chunk_a, 0.9, "vector"), (chunk_b, 0.8, "vector")]
        result = deduplicate_results(ranked, overlap_threshold=0.5)
        assert len(result) == 1
        assert result[0][0].source_id == "doc::0000"

    def test_different_documents_not_deduplicated(self):
        """Chunks from different documents should never suppress each other."""
        chunk_a = _make_chunk("doc-001::0000", "Same text span", doc_id="doc-001", span_start=0, span_end=100)
        chunk_b = _make_chunk("doc-002::0000", "Same text span", doc_id="doc-002", span_start=0, span_end=100)
        ranked = [(chunk_a, 0.9, "hybrid"), (chunk_b, 0.8, "hybrid")]
        result = deduplicate_results(ranked)
        assert len(result) == 2


# ─── Span Overlap Ratio ───────────────────────────────────────────────────────

class TestSpanOverlapRatio:
    def test_no_overlap(self):
        from backend.app.retrieval.ranker import _spans_overlap_ratio
        assert _spans_overlap_ratio(0, 10, 20, 30) == 0.0

    def test_complete_overlap(self):
        from backend.app.retrieval.ranker import _spans_overlap_ratio
        assert _spans_overlap_ratio(0, 100, 0, 100) == 1.0

    def test_partial_overlap(self):
        from backend.app.retrieval.ranker import _spans_overlap_ratio
        # Overlap region: 50-100 = 50 chars; shorter span = 100 → 0.5
        ratio = _spans_overlap_ratio(0, 100, 50, 150)
        assert abs(ratio - 0.5) < 1e-9

    def test_adjacent_no_overlap(self):
        from backend.app.retrieval.ranker import _spans_overlap_ratio
        assert _spans_overlap_ratio(0, 50, 50, 100) == 0.0
