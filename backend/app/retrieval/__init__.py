"""
Retrieval engine — Phase 2 & 3 hybrid RAG subsystem.

Primary public interface (consumed by Member 2's LangGraph agent):
    from backend.app.retrieval.hybrid_retriever import retrieve_evidence
    from backend.app.document.schemas import EvidenceResult

    result = retrieve_evidence(
        document_id="doc-001",
        question="When can the landlord terminate?",
        top_k=5,
    )

Internal modules:
    embeddings.py      — local BAAI/bge-small-en-v1.5 embedding via sentence-transformers
    vector_store.py    — Qdrant local persistent store (no server needed)
    bm25_retriever.py  — BM25 in-memory index using rank-bm25
    hybrid_retriever.py — RRF fusion of vector + BM25
    ranker.py          — RRF algorithm + overlap-aware deduplication
    search.py          — legacy BM25-only search (kept for backwards compat)
"""
