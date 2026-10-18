"""
Cache key construction utilities.

Cache keys are deterministic strings derived from:
  - Embedding cache:  sha256(text) + embedding_model_version
  - Retrieval cache:  document_id + index_version + normalize(question) + top_k + method

Keys are filesystem-safe (hex strings) and version-tagged so changing the
model or re-indexing a document automatically invalidates affected entries.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_query(query: str) -> str:
    """Normalize a query string for cache key comparison.

    Steps:
      1. Unicode NFKC normalization (collapses ligatures / fancy quotes).
      2. Lowercase.
      3. Collapse whitespace.
      4. Strip leading/trailing whitespace.

    This ensures that "When can the landlord terminate?"
    and "when can the landlord terminate?" map to the same cache entry.
    """
    normalized = unicodedata.normalize("NFKC", query)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def embedding_cache_key(text: str, model_version: str) -> str:
    """Stable cache key for a single embedding.

    Format: "emb_{sha256(text)}_{model_version}"
    The model_version is included so switching from bge-small to bge-large
    automatically misses all existing cache entries.
    """
    text_hash = _sha256(text)
    safe_model = re.sub(r"[^a-z0-9_\-]", "_", model_version.lower())
    return f"emb_{text_hash}_{safe_model}"


def retrieval_cache_key(
    document_id: str,
    index_version: str,
    question: str,
    top_k: int,
    method: str = "hybrid",
) -> str:
    """Stable cache key for a retrieval result.

    Incorporates:
      - document_id: isolates results per contract
      - index_version: invalidated when the document is re-indexed
      - normalized question: case/whitespace insensitive
      - top_k and method: different configurations produce separate entries

    Format: "ret_{sha256(canonical_string)}"
    """
    canonical = "|".join([
        document_id,
        index_version,
        normalize_query(question),
        str(top_k),
        method,
    ])
    return f"ret_{_sha256(canonical)}"
