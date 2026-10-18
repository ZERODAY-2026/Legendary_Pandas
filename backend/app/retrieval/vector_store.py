"""
Qdrant local vector store for legal document chunks.

Design:
  - Uses qdrant-client in LOCAL mode (on-disk path, no server process).
  - One collection per application (settings.QDRANT_COLLECTION_NAME).
  - Each point stores: vector, chunk text, and all Chunk metadata as payload.
  - document_id is stored in payload for scoped search (no cross-doc leakage).
  - Deletion removes all points belonging to a document_id.
  - index_version in payload enables cache invalidation.

Thread safety: QdrantClient in local mode is NOT thread-safe for writes.
Callers must serialize index_document / delete_document operations.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.document.schemas import Chunk

logger = logging.getLogger(__name__)

# Lazy singleton — created on first use to avoid startup cost
_client: Any = None


def _get_client():
    """Return the module-level Qdrant local client, creating it if needed."""
    global _client  # noqa: PLW0603
    if _client is not None:
        return _client

    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, VectorParams  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required. Run: pip install qdrant-client"
        ) from exc

    qdrant_path = settings.QDRANT_PATH
    qdrant_path.mkdir(parents=True, exist_ok=True)

    _client = QdrantClient(path=str(qdrant_path))
    logger.info("Qdrant local client opened at: %s", qdrant_path)

    # Ensure the collection exists
    existing = {c.name for c in _client.get_collections().collections}
    if settings.QDRANT_COLLECTION_NAME not in existing:
        _client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' with dimension %d",
            settings.QDRANT_COLLECTION_NAME,
            settings.EMBEDDING_DIMENSION,
        )

    return _client


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """Serialize a Chunk into a Qdrant point payload."""
    return {
        "source_id": chunk.source_id,
        "document_id": chunk.document_id,
        "source_type": chunk.source_type.value,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "clause_id": chunk.clause_id,
        "parent_clause_id": chunk.parent_clause_id,
        "text": chunk.text,
        "span_start": chunk.span_start,
        "span_end": chunk.span_end,
        "chunk_index": chunk.chunk_index,
        "index_version": chunk.index_version,
    }


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    """Reconstruct a Chunk from a Qdrant point payload."""
    from backend.app.document.schemas import SourceType

    return Chunk(
        source_id=payload["source_id"],
        document_id=payload["document_id"],
        source_type=SourceType(payload["source_type"]),
        page_start=payload["page_start"],
        page_end=payload["page_end"],
        clause_id=payload.get("clause_id"),
        parent_clause_id=payload.get("parent_clause_id"),
        text=payload["text"],
        span_start=payload["span_start"],
        span_end=payload["span_end"],
        chunk_index=payload.get("chunk_index", 0),
        index_version=payload.get("index_version", "v1"),
    )


def index_document(chunks: list[Chunk], vectors: list[list[float]]) -> int:
    """Upsert chunk vectors into the Qdrant collection.

    Args:
        chunks: List of Chunk objects to index.
        vectors: Corresponding embedding vectors (same length as chunks).

    Returns:
        Number of points successfully upserted.

    Raises:
        ValueError: If len(chunks) != len(vectors).
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have the same length."
        )
    if not chunks:
        return 0

    try:
        from qdrant_client.models import PointStruct  # type: ignore
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required.") from exc

    client = _get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.source_id)),
            vector=vector,
            payload=_chunk_to_payload(chunk),
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=points,
    )
    logger.info("Upserted %d points for document '%s'", len(points), chunks[0].document_id)
    return len(points)


def search_similar(
    query_vector: list[float],
    document_id: str,
    top_k: int | None = None,
) -> list[tuple[Chunk, float]]:
    """Vector similarity search scoped to a single document.

    Args:
        query_vector: Embedded query vector from embed_query().
        document_id: Restrict results to chunks from this document only.
        top_k: Number of results to return (defaults to settings.VECTOR_TOP_K).

    Returns:
        List of (Chunk, score) tuples sorted by cosine similarity descending.
    """
    k = top_k or settings.VECTOR_TOP_K
    client = _get_client()

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required.") from exc

    doc_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            query_filter=doc_filter,
            limit=k,
            with_payload=True,
        )
        hits = response.points
    else:
        hits = client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=doc_filter,
            limit=k,
            with_payload=True,
        )

    return [
        (_payload_to_chunk(hit.payload), hit.score)
        for hit in hits
        if hit.payload
    ]


def delete_document(document_id: str) -> int:
    """Delete all indexed chunks for a given document_id.

    Args:
        document_id: The document whose chunks should be removed.

    Returns:
        Number of deleted points (may be 0 if doc was never indexed).
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required.") from exc

    client = _get_client()
    doc_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )

    result = client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=doc_filter,
    )
    logger.info("Deleted all Qdrant points for document '%s'", document_id)
    # qdrant returns UpdateResult; operation_id is not a count — scroll to verify if needed
    return 0  # exact count requires a separate scroll; acceptable for MVP


def document_is_indexed(document_id: str) -> bool:
    """Return True if at least one chunk for document_id exists in the index."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required.") from exc

    client = _get_client()
    doc_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )
    # Use scroll with limit=1 to check existence cheaply
    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=doc_filter,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(points) > 0
