"""
Local embedding generator using BAAI/bge-small-en-v1.5 via sentence-transformers.

Design constraints:
  - Model is loaded ONCE at module level (lazy, thread-safe singleton).
  - No external API calls — inference runs fully offline on CPU.
  - BGE models require a query prefix for asymmetric retrieval:
      queries:   "Represent this sentence for searching relevant passages: <query>"
      passages:  no prefix needed
  - Supports batch encoding for efficient indexing of large chunk sets.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

from backend.app.config import settings

logger = logging.getLogger(__name__)

# BGE asymmetric search prefix (applied to queries, not passages)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_model():
    """Lazy singleton: load the embedding model once and cache it in-process."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local embeddings. "
            "Run: pip install sentence-transformers"
        ) from exc

    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(
        settings.EMBEDDING_MODEL_NAME,
        device=settings.DEVICE,
    )
    logger.info("Embedding model loaded. Dimension: %d", settings.EMBEDDING_DIMENSION)
    return model


def embed_passages(texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed a list of passage texts (document chunks) into dense vectors.

    Args:
        texts: List of passage strings to embed.
        batch_size: Inference batch size (defaults to settings.EMBEDDING_BATCH_SIZE).

    Returns:
        List of float vectors, one per input text.  Length == len(texts).
    """
    if not texts:
        return []

    model = _get_model()
    bs = batch_size or settings.EMBEDDING_BATCH_SIZE

    embeddings = model.encode(
        list(texts),
        batch_size=bs,
        normalize_embeddings=True,   # cosine similarity == dot product after normalization
        show_progress_bar=False,
    )
    return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]


def embed_query(query: str) -> list[float]:
    """Embed a single query string with the BGE asymmetric prefix.

    Args:
        query: The user's natural-language question.

    Returns:
        A single float vector of length settings.EMBEDDING_DIMENSION.
    """
    prefixed = _BGE_QUERY_PREFIX + query
    model = _get_model()
    embedding = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


def get_embedding_model_version() -> str:
    """Return a stable version string for the current embedding model.

    Used as part of cache keys so that changing the model invalidates
    all cached embeddings automatically.
    """
    return settings.EMBEDDING_MODEL_NAME.replace("/", "__")
