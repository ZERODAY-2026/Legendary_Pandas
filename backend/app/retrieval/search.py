"""
Query processor, query expansion, and local BM25 retrieval coordinator.

NOTE: This module is the LEGACY BM25-only retrieval path.
New code should use hybrid_retriever.retrieve_evidence() instead,
which combines vector similarity (Qdrant) + BM25 via Reciprocal Rank Fusion.
"""

from typing import NamedTuple
from backend.app.config import settings
from backend.app.document.schemas import Chunk, Evidence
from backend.app.retrieval.index import BM25DocumentIndex
from backend.app.retrieval.ranker import rank_and_deduplicate


class RetrievalResult(NamedTuple):
    passages: list[Chunk]
    evidence_items: list[Evidence]
    is_sufficient: bool
    reason: str | None = None


# Basic concept expansion dictionary for rental contracts
CONCEPT_EXPANSIONS: dict[str, list[str]] = {
    "terminate": ["terminate", "termination", "notice", "breach", "expiry", "end of lease"],
    "termination": ["termination", "terminate", "notice", "breach", "expiry", "end of lease"],
    "deposit": ["deposit", "security deposit", "holding deposit", "refund", "deductions"],
    "rent": ["rent", "monthly", "payment", "due date", "late fee", "grace period"],
    "repairs": ["repairs", "maintenance", "damage", "wear and tear", "landlord obligations"],
    "sublet": ["sublet", "sublease", "assignment", "guests", "occupants"],
    "pets": ["pets", "animals", "dog", "cat", "pet fee"],
}


def expand_query(query: str) -> list[str]:
    """Tokenize and expand query terms using concept synonyms."""
    import re

    words = re.findall(r"\w+", query.lower())
    expanded = set(words)
    for w in words:
        if w in CONCEPT_EXPANSIONS:
            expanded.update(CONCEPT_EXPANSIONS[w])
    return list(expanded)


def search_chunks(
    index: BM25DocumentIndex,
    query: str,
    top_k: int = 5,
    min_score: float = 0.5,
) -> RetrievalResult:
    """Retrieve relevant chunks for a user query against the BM25 index (legacy)."""
    if not index.chunks or not index.bm25:
        return RetrievalResult(
            passages=[],
            evidence_items=[],
            is_sufficient=False,
            reason="No indexed chunks available in the active document.",
        )

    expanded_terms = expand_query(query)
    scores = index.bm25.get_scores(expanded_terms)

    scored_chunks = list(zip(index.chunks, scores))
    ranked = rank_and_deduplicate(scored_chunks, top_k=top_k, min_score=min_score)

    if not ranked:
        return RetrievalResult(
            passages=[],
            evidence_items=[],
            is_sufficient=False,
            reason="No document passages cleared the relevance threshold for the question.",
        )

    passages = [chunk for chunk, _ in ranked]
    evidence_items = [
        Evidence(
            source_id=chunk.source_id,  # updated: was chunk.document_id
            source_type=chunk.source_type.value,
            page=chunk.page_start,
            section=chunk.clause_id,
            quote=chunk.text,
        )
        for chunk in passages
    ]

    return RetrievalResult(
        passages=passages,
        evidence_items=evidence_items,
        is_sufficient=True,
    )

