"""
Tests for clause-aware overlapping chunker.
Covers: span invariant, overlap, clause detection, metadata preservation,
boundary conditions, and malformed input.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend.app.document.chunker import chunk_document, _build_clause_segments, _split_segment_with_overlap
from backend.app.document.schemas import DocumentMetadata, ExtractedDocument, SourceType
from backend.app.ingestion.txt_parser import parse_txt


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_CONTRACT = """\
RESIDENTIAL LEASE AGREEMENT

This lease agreement is entered into between the Landlord and the Tenant.

1. TERM
The tenancy shall commence on the 1st of January 2024 and shall continue for
a period of twelve (12) months unless terminated earlier in accordance with
the terms herein.

2. RENT
The monthly rent shall be $1,500 USD payable on the first day of each month.
A late fee of $50 shall be charged for payments received after the 5th day.

3. SECURITY DEPOSIT
A security deposit of $3,000 shall be due upon signing this agreement.
The deposit will be returned within 30 days of the end of the tenancy, less
any deductions for damages beyond normal wear and tear.

4. TERMINATION
Either party may terminate this agreement by providing 60 days written notice.
The landlord may terminate immediately for non-payment of rent or breach of
the terms of this agreement.

5. PETS
No pets are permitted without prior written consent of the landlord.

6. GOVERNING LAW
This agreement shall be governed by the laws of the State of California.
"""


def _make_extracted_doc(text: str, doc_id: str = "test-doc") -> ExtractedDocument:
    """Helper to create an ExtractedDocument from a text string."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False)
    tmp.write(text)
    tmp.flush()
    return parse_txt(Path(tmp.name), doc_id)


# ─── Span Invariant Tests ─────────────────────────────────────────────────────

class TestSpanInvariant:
    """The core correctness property: raw_text[span_start:span_end] == chunk.text."""

    def test_span_invariant_basic(self):
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks = chunk_document(doc)
        assert chunks, "Expected at least one chunk"
        for chunk in chunks:
            actual = doc.raw_text[chunk.span_start : chunk.span_end]
            assert actual == chunk.text, (
                f"Span invariant violated for chunk {chunk.source_id!r}:\n"
                f"  expected: {chunk.text[:80]!r}\n"
                f"  got:      {actual[:80]!r}"
            )

    def test_span_invariant_short_document(self):
        doc = _make_extracted_doc("Short contract with one clause.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert doc.raw_text[chunk.span_start : chunk.span_end] == chunk.text

    def test_span_invariant_repeated_chunking(self):
        """Chunking the same document twice must produce the same chunks."""
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks_a = chunk_document(doc)
        chunks_b = chunk_document(doc)
        assert len(chunks_a) == len(chunks_b)
        for a, b in zip(chunks_a, chunks_b):
            assert a.text == b.text
            assert a.span_start == b.span_start
            assert a.span_end == b.span_end


# ─── Overlap Tests ────────────────────────────────────────────────────────────

class TestOverlap:
    def test_overlapping_chunks_share_context(self):
        """When a clause is split into multiple sub-chunks, consecutive chunks share text."""
        # Use a tiny chunk size to force splitting of clauses (longest clause is ~220 chars)
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks = chunk_document(doc, max_chunk_chars=180, chunk_overlap=90)

        # Look for adjacent chunks with overlapping spans
        overlapping_pairs = 0
        for i in range(len(chunks) - 1):
            a, b = chunks[i], chunks[i + 1]
            if a.document_id == b.document_id and a.span_end > b.span_start:
                overlapping_pairs += 1

        # We expect overlap for a 180-char chunk on a multi-clause contract
        assert overlapping_pairs > 0, "Expected at least one overlapping pair with small chunk size"

    def test_chunk_size_respected(self):
        """No chunk should exceed max_chunk_chars (LangChain guarantees this approximately)."""
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        max_size = 400
        chunks = chunk_document(doc, max_chunk_chars=max_size, chunk_overlap=80)
        for chunk in chunks:
            # Allow a small overshoot due to separator keep logic
            assert len(chunk.text) <= max_size * 1.2, (
                f"Chunk too large: {len(chunk.text)} chars (limit {max_size})"
            )


# ─── Metadata Tests ───────────────────────────────────────────────────────────

class TestChunkMetadata:
    def test_document_id_preserved(self):
        doc = _make_extracted_doc(SAMPLE_CONTRACT, "my-contract-42")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.document_id == "my-contract-42"

    def test_source_type_preserved(self):
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.source_type == SourceType.CONTRACT

    def test_page_numbers_valid(self):
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.page_start >= 1
            assert chunk.page_end >= chunk.page_start

    def test_source_id_unique_per_chunk(self):
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks = chunk_document(doc)
        source_ids = [c.source_id for c in chunks]
        assert len(source_ids) == len(set(source_ids)), "source_id values must be unique"

    def test_clause_id_populated_for_detected_sections(self):
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks = chunk_document(doc)
        # At least some chunks should have a clause_id from section detection
        chunks_with_clause = [c for c in chunks if c.clause_id is not None]
        assert chunks_with_clause, "Expected clause IDs from numbered sections"

    def test_chunk_index_sequential(self):
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks = chunk_document(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks))), "chunk_index should be sequential from 0"


# ─── Boundary / Edge Cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_document_returns_no_chunks(self):
        doc = _make_extracted_doc("")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_whitespace_only_document(self):
        doc = _make_extracted_doc("   \n\n  \t  \n")
        chunks = chunk_document(doc)
        # All-whitespace content should be skipped
        assert all(len(c.text.strip()) > 0 for c in chunks)

    def test_single_short_clause(self):
        doc = _make_extracted_doc("1. TERM\nThe lease is for 6 months.")
        chunks = chunk_document(doc)
        assert len(chunks) >= 1
        full_text = " ".join(c.text for c in chunks)
        assert "TERM" in full_text or "lease" in full_text

    def test_min_chunk_filter(self):
        """Chunks shorter than min_chunk_chars should be dropped."""
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks = chunk_document(doc, min_chunk_chars=50)
        for chunk in chunks:
            assert len(chunk.text.strip()) >= 50

    def test_no_empty_chunks(self):
        doc = _make_extracted_doc(SAMPLE_CONTRACT)
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.text.strip() != "", "Empty chunks must not be produced"
