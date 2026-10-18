"""
Local retrieval result cache using the filesystem.

Each cached result is a JSON-serialized EvidenceResult stored at:
    <RETRIEVAL_CACHE_DIR>/<cache_key>.json

The cache key encodes: document_id + index_version + normalized(question) + top_k + method.
Changing any of these (including re-indexing the document which bumps index_version)
produces a different key and a cache miss.

Privacy guarantee:
  - Cached files live on the local machine only; no external transmission.
  - Files contain question text and retrieved passages (both acceptable since
    they are already stored in Qdrant on-disk).

Invalidation:
  - Call invalidate_document_retrieval(document_id) to purge all retrieval cache
    entries for that document.  This is tracked via a per-document manifest.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from backend.app.caching.keys import retrieval_cache_key
from backend.app.config import settings
from backend.app.document.schemas import EvidenceResult

logger = logging.getLogger(__name__)


def _cache_dir() -> Path:
    d = settings.RETRIEVAL_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key_path(key: str) -> Path:
    return _cache_dir() / f"{key}.json"


def _doc_manifest_path(document_id: str) -> Path:
    return _cache_dir() / f"_manifest_{document_id}.json"


def _update_manifest(document_id: str, key: str) -> None:
    manifest_path = _doc_manifest_path(document_id)
    existing: list[str] = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    if key not in existing:
        existing.append(key)
        # Enforce max entries per document to prevent unbounded growth
        if len(existing) > settings.CACHE_MAX_ENTRIES:
            existing = existing[-settings.CACHE_MAX_ENTRIES :]
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(existing, f)
        except OSError as exc:
            logger.warning("Failed to update retrieval manifest for %s: %s", document_id, exc)


# ─── Public API ──────────────────────────────────────────────────────────────

def get_cached_retrieval(
    document_id: str,
    question: str,
    top_k: int,
    index_version: str = "v1",
    method: str = "hybrid",
) -> EvidenceResult | None:
    """Return a cached EvidenceResult if one exists, otherwise None.

    Args:
        document_id: The contract that was searched.
        question: The original question (will be normalized internally).
        top_k: Number of results that were requested.
        index_version: The index version tag; bumping this misses old entries.
        method: Retrieval method used ("hybrid", "vector", "bm25").

    Returns:
        Deserialized EvidenceResult, or None on cache miss or any read error.
    """
    key = retrieval_cache_key(document_id, index_version, question, top_k, method)
    path = _key_path(key)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EvidenceResult.model_validate(data)
    except Exception as exc:
        logger.warning("Corrupt retrieval cache file %s: %s", path.name, exc)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def cache_retrieval(
    document_id: str,
    question: str,
    top_k: int,
    result: EvidenceResult,
    index_version: str | None = None,
    method: str = "hybrid",
) -> None:
    """Persist an EvidenceResult to the retrieval cache.

    Args:
        document_id: The contract that was searched.
        question: The original question.
        top_k: Number of results.
        result: The EvidenceResult to cache.
        index_version: Index version (defaults to result.index_version).
        method: Retrieval method used.
    """
    iv = index_version or result.index_version
    key = retrieval_cache_key(document_id, iv, question, top_k, method)
    path = _key_path(key)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f)
        _update_manifest(document_id, key)
    except OSError as exc:
        logger.warning("Failed to write retrieval cache entry %s: %s", key, exc)


def invalidate_document_retrieval(document_id: str) -> int:
    """Delete all retrieval cache entries for a document.

    Must be called whenever a document is re-indexed or deleted to prevent
    stale evidence from being served.

    Args:
        document_id: The document to evict.

    Returns:
        Number of cache entries deleted.
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

    logger.info("Evicted %d retrieval cache entries for document '%s'", deleted, document_id)
    return deleted


def invalidate_all_for_document(document_id: str) -> dict[str, int]:
    """Invalidate both embedding and retrieval caches for a document.

    Call this when a document is updated, re-indexed, or deleted.

    Returns:
        Dict with counts: {"embedding": N, "retrieval": M}
    """
    from backend.app.caching.embedding_cache import invalidate_document_embeddings

    emb_count = invalidate_document_embeddings(document_id)
    ret_count = invalidate_document_retrieval(document_id)
    return {"embedding": emb_count, "retrieval": ret_count}
