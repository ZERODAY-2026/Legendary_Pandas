"""Provenance tracking and validation utilities.

Ensures every evidence citation and claim has verified provenance back to
an ingested document chunk.
"""

import hashlib
from typing import Optional
from backend.app.document.schemas import Chunk, EvidencePassage, SourceType


def compute_content_hash(text: str) -> str:
    """Computes SHA-256 hash of text for tamper-detection and provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ProvenanceRecord:
    """Encapsulates provenance verification for a chunk or evidence passage."""

    @staticmethod
    def create_evidence_from_chunk(
        chunk: Chunk,
        exact_quote: str,
        start_char: int,
        end_char: int,
        evidence_id: str,
        relevance_score: Optional[float] = None
    ) -> EvidencePassage:
        """Constructs an EvidencePassage strictly inheriting the immutable source_type from chunk."""
        return EvidencePassage(
            evidence_id=evidence_id,
            document_id=chunk.document_id,
            source_type=chunk.source_type,
            page_number=chunk.page_number,
            start_char=start_char,
            end_char=end_char,
            exact_quote=exact_quote,
            clause_number=chunk.clause_number,
            relevance_score=relevance_score,
        )

    @staticmethod
    def verify_evidence_provenance(evidence: EvidencePassage, full_document_text: str) -> bool:
        """Verifies that the evidence passage exists verbatim in the document."""
        return evidence.verify_verbatim(full_document_text)
