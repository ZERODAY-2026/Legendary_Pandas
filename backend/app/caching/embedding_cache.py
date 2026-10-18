"""
Local embedding cache using the filesystem.

Each cached embedding is stored as a JSON file:
    <EMBEDDING_CACHE_DIR>/<cache_key>.json

The cache key encodes the text hash and embedding model version (see keys.py),
so switching the model automatically misses all existing entries.

Privacy guarantee:
  - Cache files contain only the vector (list of floats).
  - The original text is NEVER written to disk through this cache.
  - Files stay on the local machine only; no network calls.

Invalidation:
  - Call invalidate_document(document_id) to delete all embedding cache entries
    whose key prefix matches chunks belonging to that document.
  - Since keys are based on text hashes, not document IDs, bulk invalidation
    is done by tracking a metadata file per document.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

from backend.app.caching.keys import embedding_cache_key
from backend.app.config import settings
from backend.app.retrieval.embeddings import embed_passages, embed_query, get_embedding_model_version

logger = logging.getLogger(__name__)


def _cache_dir() -> Path:
    d = settings.EMBEDDING_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key_path(key: str) -> Path:
    return _cache_dir() / f"{key}.json"


def _doc_manifest_path(document_id: str) -> Path:
    """Path to the per-document manifest that lists cache keys for that doc."""
    return _cache_dir() / f"_manifest_{document_id}.json"


# ─── Read / write individual entries ─────────────────────────────────────────

def _read_cached(key: str) -> list[float] | None:
    path = _key_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Corrupt embedding cache file %s: %s", path.name, exc)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _write_cached(key: str, vector: list[float]) -> None:
    path = _key_path(key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vector, f)
    except OSError as exc:
        logger.warning("Failed to write embedding cache file %s: %s", path.name, exc)


def _update_manifest(document_id: str, keys: list[str]) -> None:
    """Append keys to the document manifest (used for bulk invalidation)."""
    manifest_path = _doc_manifest_path(document_id)
    existing: list[str] = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    combined = list(set(existing + keys))
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(combined, f)
    except OSError as exc:
        logger.warning("Failed to update embedding manifest for %s: %s", document_id, exc)


# ─── Public API ──────────────────────────────────────────────────────────────

def embed_passages_cached(
    texts: Sequence[str],
    document_id: str | None = None,
) -> list[list[float]]:
    """Embed passage texts, using the local cache to avoid recomputing.

    Args:
        texts: List of text strings to embed.
        document_id: Optional — if provided, new cache entries are tracked
                     in the per-document manifest for easy bulk invalidation.

    Returns:
        List of embedding vectors (one per text, same order).
    """
    model_version = get_embedding_model_version()
    keys = [embedding_cache_key(t, model_version) for t in texts]

    results: list[list[float] | None] = [_read_cached(k) for k in keys]

    # Find which texts need fresh computation
    missing_indices = [i for i, r in enumerate(results) if r is None]
    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        new_vectors = embed_passages(missing_texts)

        new_keys_written: list[str] = []
        for idx, vector in zip(missing_indices, new_vectors):
            results[idx] = vector
            _write_cached(keys[idx], vector)
            new_keys_written.append(keys[idx])

        if document_id and new_keys_written:
            _update_manifest(document_id, new_keys_written)

        logger.debug(
            "Embedding cache: %d hits, %d misses",
            len(texts) - len(missing_indices),
            len(missing_indices),
        )

    return [r for r in results if r is not None]


def embed_query_cached(query: str) -> list[float]:
    """Embed a query string with caching.

    Queries are cached without a document_id since they are not bound to
    a specific document and don't need bulk invalidation.
    """
    model_version = get_embedding_model_version()
    key = embedding_cache_key("QUERY:" + query, model_version)

    cached = _read_cached(key)
    if cached is not None:
        return cached

    vector = embed_query(query)
    _write_cached(key, vector)
    return vector


def invalidate_document_embeddings(document_id: str) -> int:
    """Delete all embedding cache entries for a given document.

    Args:
        document_id: The document whose embeddings should be evicted.

    Returns:
        Number of cache files deleted.
    """
    manifest_path = _doc_manifest_path(document_id)
    if not manifest_path.exists():
        return 0

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            keys: list[str] = json.load(f)
    except Exception:
        return 0

    deleted = 0
    for key in keys:
        path = _key_path(key)
        try:
            path.unlink(missing_ok=True)
            deleted += 1
        except OSError:
            pass

    try:
        manifest_path.unlink(missing_ok=True)
    except OSError:
        pass

    logger.info("Evicted %d embedding cache entries for document '%s'", deleted, document_id)
    return deleted
