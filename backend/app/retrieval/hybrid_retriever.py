"""
Hybrid retriever combining vector similarity and BM25 via Reciprocal Rank Fusion.

This is the primary retrieval interface consumed by Member 2's LangGraph agent.

Usage:
    from backend.app.retrieval.hybrid_retriever import retrieve_evidence

    result = retrieve_evidence(
        document_id="doc-001",
        question="When can the landlord terminate?",
        top_k=5,
    )

The function returns an EvidenceResult — see backend/app/document/schemas.py.
It does NOT generate any legal answer itself. That is Member 2's responsibility.
"""

from __future__ import annotations

import logging

from backend.app.config import settings
from backend.app.document.schemas import Chunk, Evidence, EvidenceResult, SourceType

logger = logging.getLogger(__name__)


def _chunks_for_document(document_id: str) -> list[Chunk]:
    """Fetch all indexed chunks for a document from the Qdrant store.

    Uses a broad vector search (with a zero vector) to enumerate all points,
    then falls back to a scroll if the collection supports it.
    This is only called by build_bm25_for_document() to construct the in-memory
    BM25 index — not on the hot retrieval path.
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        from backend.app.retrieval.vector_store import _get_client, _payload_to_chunk

        client = _get_client()
        doc_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )
        all_chunks: list[Chunk] = []
        next_offset = None

        while True:
            points, next_offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                scroll_filter=doc_filter,
                limit=200,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                if p.payload:
                    all_chunks.append(_payload_to_chunk(p.payload))
            if next_offset is None:
                break

        # Sort by chunk_index for consistent BM25 corpus ordering
        all_chunks.sort(key=lambda c: c.chunk_index)
        return all_chunks

    except Exception as exc:  # pragma: no cover
        logger.error("Failed to fetch chunks for BM25 index: %s", exc)
        return []


def retrieve_evidence(
    document_id: str,
    question: str,
    top_k: int | None = None,
    method: str = "hybrid",
) -> EvidenceResult:
    """Main retrieval interface for Member 2.

    Performs hybrid retrieval (vector + BM25) and returns the top-k most
    relevant passages with full provenance metadata.

    Args:
        document_id: The ID of the contract to search.
        question: The user's natural-language question.
        top_k: Number of passages to return (defaults to settings.HYBRID_TOP_K).
        method: "hybrid" (default), "vector", or "bm25" — for evaluation/ablation.

    Returns:
        EvidenceResult with:
            - passages: list of Evidence objects (ranked, deduplicated)
            - is_sufficient: False when no passages cleared relevance threshold
            - reason: human-readable explanation when is_sufficient is False
            - index_version: the version tag of the index used

    Raises:
        RuntimeError: If the required dependencies are not installed.

    Design guarantee:
        This function NEVER calls any external API or generative model.
        It is strictly a retrieval function.
    """
    k = top_k or settings.HYBRID_TOP_K

    # ── 1. Check cache first ──────────────────────────────────────────────────
    try:
        from backend.app.caching.retrieval_cache import get_cached_retrieval, cache_retrieval

        cached = get_cached_retrieval(document_id=document_id, question=question, top_k=k)
        if cached is not None:
            logger.debug("Cache hit for document '%s'", document_id)
            return cached
    except ImportError:
        cache_retrieval = None  # caching module not yet available
        cached = None

    # ── 2. Vector retrieval ───────────────────────────────────────────────────
    vector_results: list[tuple[Chunk, float]] = []
    index_ver = "v1"

    if method in ("hybrid", "vector"):
        try:
            from backend.app.retrieval.embeddings import embed_query
            from backend.app.retrieval.vector_store import search_similar

            query_vector = embed_query(question)
            vector_results = search_similar(query_vector, document_id, top_k=k)
            if vector_results:
                index_ver = vector_results[0][0].index_version

        except Exception as exc:
            logger.warning("Vector retrieval failed: %s", exc)

    # ── 3. BM25 retrieval ─────────────────────────────────────────────────────
    bm25_results: list[tuple[Chunk, float]] = []

    if method in ("hybrid", "bm25"):
        try:
            from backend.app.retrieval.bm25_retriever import BM25DocumentIndex

            chunks = _chunks_for_document(document_id)
            if chunks:
                bm25_index = BM25DocumentIndex(chunks=chunks)
                bm25_results = bm25_index.search(question, top_k=k)

        except Exception as exc:
            logger.warning("BM25 retrieval failed: %s", exc)

    # ── 4. Merge and rank using RRF ───────────────────────────────────────────
    from backend.app.retrieval.ranker import reciprocal_rank_fusion, deduplicate_results

    if method == "vector":
        # Fake BM25 list so RRF still works with single-source input
        ranked = [(c, s, "vector") for c, s in vector_results[:k]]
    elif method == "bm25":
        ranked = [(c, s, "bm25") for c, s in bm25_results[:k]]
    else:
        ranked = reciprocal_rank_fusion(vector_results, bm25_results, top_k=k)

    deduped = deduplicate_results(ranked)

    # ── 5. Build EvidenceResult ───────────────────────────────────────────────
    if not deduped:
        result = EvidenceResult(
            document_id=document_id,
            question=question,
            passages=[],
            is_sufficient=False,
            reason="No relevant passages found for this question in the indexed document.",
            index_version=index_ver,
        )
    else:
        passages = []
        for rank_idx, (chunk, rrf_score, ret_method) in enumerate(deduped, start=1):
            passages.append(
                Evidence(
                    source_id=chunk.source_id,
                    source_type=chunk.source_type.value,  # type: ignore[arg-type]
                    page=chunk.page_start,
                    section=chunk.clause_id,
                    quote=chunk.text,
                    retrieval_rank=rank_idx,
                    rrf_score=round(rrf_score, 6),
                    retrieval_method=ret_method,
                    index_version=chunk.index_version,
                )
            )

        result = EvidenceResult(
            document_id=document_id,
            question=question,
            passages=passages,
            is_sufficient=True,
            index_version=index_ver,
        )

    # ── 6. Store in cache ─────────────────────────────────────────────────────
    try:
        if callable(cache_retrieval):  # type: ignore[arg-type]
            cache_retrieval(document_id=document_id, question=question, top_k=k, result=result)
    except Exception:
        pass  # Cache write failures must never break retrieval

    return result
