"""
BM25 keyword retriever using rank-bm25.

Operates entirely in-memory over the chunks for one document.
Built fresh on first retrieval for a document; callers should cache the
BM25DocumentIndex object to avoid re-tokenizing on every query.

Design notes:
  - Tokenization: lowercase alphanumeric tokens (legal terms stay intact).
  - Query expansion: simple synonym table for common rental-law concepts.
  - The returned list is sorted by BM25 Okapi score descending.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

from backend.app.config import settings
from backend.app.document.schemas import Chunk

logger = logging.getLogger(__name__)


# ─── Rental-law concept expansion table ──────────────────────────────────────
CONCEPT_EXPANSIONS: dict[str, list[str]] = {
    "terminate": ["terminate", "termination", "notice", "breach", "expiry", "end of lease"],
    "termination": ["termination", "terminate", "notice", "breach", "expiry", "end of lease"],
    "deposit": ["deposit", "security deposit", "holding deposit", "refund", "deductions"],
    "security": ["security", "deposit", "bond", "guarantee"],
    "rent": ["rent", "monthly", "payment", "due date", "late fee", "grace period"],
    "repairs": ["repairs", "maintenance", "damage", "wear and tear", "landlord obligations"],
    "sublet": ["sublet", "sublease", "assignment", "guests", "occupants"],
    "sublease": ["sublease", "sublet", "assignment", "transfer"],
    "pets": ["pets", "animals", "dog", "cat", "pet fee", "pet deposit"],
    "landlord": ["landlord", "lessor", "owner", "property owner"],
    "tenant": ["tenant", "lessee", "renter", "occupant"],
    "notice": ["notice", "notification", "written notice", "days notice"],
    "penalty": ["penalty", "late fee", "charge", "fine", "interest"],
    "utilities": ["utilities", "electricity", "water", "gas", "heating", "internet"],
    "renewal": ["renewal", "renew", "extend", "extension", "holdover"],
    "inspection": ["inspection", "entry", "access", "visit", "walkthrough"],
}


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer (preserves hyphenated terms)."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def expand_query(query: str) -> list[str]:
    """Tokenize and expand query tokens using the rental-law synonym table."""
    tokens = _tokenize(query)
    expanded: set[str] = set(tokens)
    for token in tokens:
        if token in CONCEPT_EXPANSIONS:
            expanded.update(CONCEPT_EXPANSIONS[token])
    # Also tokenize each multi-word synonym
    final: list[str] = []
    for term in expanded:
        final.extend(_tokenize(term))
    return list(set(final))


@dataclass
class BM25DocumentIndex:
    """In-memory BM25 index over all chunks for a single document."""

    chunks: list[Chunk]
    _bm25: object = field(init=False, repr=False, default=None)
    _tokenized_corpus: list[list[str]] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self._tokenized_corpus = [_tokenize(c.text) for c in self.chunks]
        if self._tokenized_corpus:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            except ImportError:
                logger.warning(
                    "rank-bm25 not installed; BM25 retrieval will be unavailable."
                )
                self._bm25 = None

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return top-k (Chunk, BM25-score) pairs for the given query.

        Args:
            query: Natural-language question.
            top_k: Number of results (defaults to settings.BM25_TOP_K).
            min_score: Minimum BM25 score to include (defaults to settings.BM25_MIN_SCORE_THRESHOLD).

        Returns:
            List of (Chunk, score) sorted by score descending.
        """
        k = top_k or settings.BM25_TOP_K
        threshold = min_score if min_score is not None else settings.BM25_MIN_SCORE_THRESHOLD

        if not self.chunks or self._bm25 is None:
            return []

        query_tokens = expand_query(query)
        scores: list[float] = self._bm25.get_scores(query_tokens)

        # Zip with chunks, filter, sort
        scored = [(chunk, float(score)) for chunk, score in zip(self.chunks, scores)]
        filtered = [(c, s) for c, s in scored if s >= threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:k]
