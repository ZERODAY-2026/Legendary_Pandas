"""
Reciprocal Rank Fusion (RRF) ranker and overlap-aware deduplication.

RRF formula:  score(d) = Σ  1 / (k + rank_i(d))
  where k = settings.RRF_K (default 60) and rank_i is the 1-based rank
  of document d in result list i.

Deduplication:
  After ranking, overlapping chunks (same document_id, overlapping span)
  that contribute only redundant context are collapsed: the higher-ranked
  chunk wins and absorbs lower-ranked neighbours whose span overlaps it
  by more than 50 %.
"""

from __future__ import annotations

import logging
from typing import Literal

from backend.app.config import settings
from backend.app.document.schemas import Chunk

logger = logging.getLogger(__name__)

RetrievalMethod = Literal["vector", "bm25", "hybrid"]


def reciprocal_rank_fusion(
    vector_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    k: int | None = None,
    top_k: int | None = None,
) -> list[tuple[Chunk, float, RetrievalMethod]]:
    """Merge vector and BM25 ranked lists using Reciprocal Rank Fusion.

    Args:
        vector_results: (Chunk, cosine-score) from vector_store.search_similar().
        bm25_results:   (Chunk, BM25-score) from BM25DocumentIndex.search().
        k: RRF constant (default: settings.RRF_K = 60).
        top_k: Max results to return (default: settings.HYBRID_TOP_K).

    Returns:
        List of (Chunk, rrf_score, method) sorted by rrf_score descending.
        method is "vector", "bm25", or "hybrid" indicating which list(s)
        contributed to that chunk's score.
    """
    rrf_k = k or settings.RRF_K
    n = top_k or settings.HYBRID_TOP_K

    # Build a map: source_id -> (chunk, rrf_score, present_in_sets)
    scores: dict[str, list] = {}  # source_id -> [chunk, score, methods_set]

    for rank, (chunk, _) in enumerate(vector_results, start=1):
        sid = chunk.source_id
        if sid not in scores:
            scores[sid] = [chunk, 0.0, set()]
        scores[sid][1] += 1.0 / (rrf_k + rank)
        scores[sid][2].add("vector")

    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        sid = chunk.source_id
        if sid not in scores:
            scores[sid] = [chunk, 0.0, set()]
        scores[sid][1] += 1.0 / (rrf_k + rank)
        scores[sid][2].add("bm25")

    # Convert to result list
    merged: list[tuple[Chunk, float, RetrievalMethod]] = []
    for chunk, rrf_score, methods in scores.values():
        if "vector" in methods and "bm25" in methods:
            method: RetrievalMethod = "hybrid"
        elif "vector" in methods:
            method = "vector"
        else:
            method = "bm25"
        merged.append((chunk, rrf_score, method))

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:n]


def _spans_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Fraction of the shorter span that is covered by the overlap."""
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_end <= overlap_start:
        return 0.0
    overlap_len = overlap_end - overlap_start
    shorter = min(a_end - a_start, b_end - b_start)
    return overlap_len / shorter if shorter > 0 else 0.0


def deduplicate_results(
    ranked: list[tuple[Chunk, float, RetrievalMethod]],
    overlap_threshold: float = 0.5,
) -> list[tuple[Chunk, float, RetrievalMethod]]:
    """Remove redundant overlapping chunks from a ranked list.

    A chunk is suppressed if its text span overlaps a higher-ranked chunk
    from the same document by more than overlap_threshold (default 50 %).

    This preserves useful adjacent context while eliminating copies of the
    same passage produced by overlapping chunk windows.

    Args:
        ranked: Output of reciprocal_rank_fusion(), sorted by score desc.
        overlap_threshold: Fraction of the shorter span that counts as "redundant".

    Returns:
        Deduplicated list, same ordering.
    """
    kept: list[tuple[Chunk, float, RetrievalMethod]] = []
    for candidate_chunk, candidate_score, candidate_method in ranked:
        suppress = False
        for kept_chunk, _, _ in kept:
            if kept_chunk.document_id != candidate_chunk.document_id:
                continue
            ratio = _spans_overlap_ratio(
                kept_chunk.span_start,
                kept_chunk.span_end,
                candidate_chunk.span_start,
                candidate_chunk.span_end,
            )
            if ratio >= overlap_threshold:
                suppress = True
                break
        if not suppress:
            kept.append((candidate_chunk, candidate_score, candidate_method))
    return kept


def rank_and_deduplicate(
    scored_chunks: list[tuple[Chunk, float]],
    top_k: int = 5,
    min_score: float = 0.5,
) -> list[tuple[Chunk, float]]:
    """Legacy BM25-only ranking used by the old search.py interface.

    Kept for backwards compatibility with existing code that calls this function.
    New code should use reciprocal_rank_fusion() + deduplicate_results() directly.
    """
    valid = [sc for sc in scored_chunks if sc[1] >= min_score]
    valid.sort(key=lambda x: x[1], reverse=True)

    seen_spans: set[tuple[str, int, int]] = set()
    deduped: list[tuple[Chunk, float]] = []
    for chunk, score in valid:
        key = (chunk.document_id, chunk.span_start, chunk.span_end)
        if key not in seen_spans:
            seen_spans.add(key)
            deduped.append((chunk, score))
        if len(deduped) >= top_k:
            break

    return deduped

