"""
Tests for PDF/TXT extraction pipelines.
Covers: valid files, error cases, metadata accuracy, span integrity.
"""

from __future__ import annotations

import io
import struct
import tempfile
from pathlib import Path

import pytest

from backend.app.config import settings
from backend.app.ingestion.txt_parser import parse_txt
from backend.app.ingestion.validator import validate_file, ValidationError
from backend.app.document.schemas import SourceType


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _write_tmp_txt(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False)
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


# ─── TXT Parser Tests ─────────────────────────────────────────────────────────

class TestTxtParser:
    def test_basic_extraction(self):
        p = _write_tmp_txt("Hello, world!")
        doc = parse_txt(p, "doc-001")
        assert doc.raw_text == "Hello, world!"
        assert doc.metadata.document_id == "doc-001"
        assert doc.metadata.file_type == "txt"

    def test_page_spans_cover_full_text(self):
        text = "Line one\nLine two\nLine three"
        p = _write_tmp_txt(text)
        doc = parse_txt(p, "doc-002")
        for page_num, (start, end) in doc.page_spans.items():
            assert doc.raw_text[start:end] == doc.page_texts[page_num]

    def test_single_page_for_txt(self):
        p = _write_tmp_txt("Some contract text.")
        doc = parse_txt(p, "doc-003")
        assert doc.metadata.page_count == 1
        assert 1 in doc.page_texts

    def test_char_count_matches_raw_text(self):
        text = "RENTAL AGREEMENT between landlord and tenant."
        p = _write_tmp_txt(text)
        doc = parse_txt(p, "doc-004")
        assert doc.metadata.char_count == len(text)

    def test_empty_file_parses_without_crash(self):
        p = _write_tmp_txt("")
        doc = parse_txt(p, "doc-005")
        assert doc.raw_text == ""
        assert doc.metadata.char_count == 0

    def test_source_type_defaults_to_contract(self):
        p = _write_tmp_txt("Some text")
        doc = parse_txt(p, "doc-006")
        assert doc.metadata.source_type == SourceType.CONTRACT

    def test_span_invariant(self):
        """For every page: raw_text[span_start:span_end] == page_texts[page]."""
        text = "LEASE AGREEMENT\n\nThis lease governs the rental of the premises."
        p = _write_tmp_txt(text)
        doc = parse_txt(p, "doc-007")
        for page_num, (start, end) in doc.page_spans.items():
            assert doc.raw_text[start:end] == doc.page_texts[page_num], (
                f"Span invariant violated for page {page_num}"
            )


# ─── Validator Tests ──────────────────────────────────────────────────────────

class TestValidator:
    def test_valid_txt_file(self):
        content = b"This is a contract."
        result = validate_file(Path("contract.txt"), "contract.txt", content)
        assert result == "txt"

    def test_valid_pdf_magic(self):
        content = b"%PDF-1.4 binary content..."
        result = validate_file(Path("contract.pdf"), "contract.pdf", content)
        assert result == "pdf"

    def test_invalid_pdf_magic_raises(self):
        content = b"Not a real PDF"
        with pytest.raises(ValidationError, match="PDF header"):
            validate_file(Path("fake.pdf"), "fake.pdf", content)

    def test_unsupported_extension_raises(self):
        content = b"Some content"
        with pytest.raises(ValidationError, match="Unsupported file extension"):
            validate_file(Path("doc.docx"), "doc.docx", content)

    def test_file_size_limit_when_configured(self, monkeypatch):
        # Configure a temporary limit to verify size enforcement works when set
        monkeypatch.setattr(settings, "MAX_FILE_SIZE_BYTES", 100)
        content = b"x" * 101
        with pytest.raises(ValidationError, match="maximum allowed size"):
            validate_file(Path("big.txt"), "big.txt", content)

    def test_large_file_allowed_by_default(self):
        # When limit is removed (None), files larger than 25MB are accepted
        content = b"%PDF-1.4 " + (b"x" * (26 * 1024 * 1024))
        result = validate_file(Path("big.pdf"), "big.pdf", content)
        assert result == "pdf"

    def test_non_utf8_txt_accepted(self):
        """Latin-1 encoded files should be accepted (fallback encoding)."""
        content = "Téléphone: 01 23 45".encode("latin-1")
        # latin-1 fails UTF-8 decode but should be caught by fallback
        result = validate_file(Path("latin.txt"), "latin.txt", content)
        assert result == "txt"


# ─── PDF Parser Tests (skipped if fitz not installed) ─────────────────────────

try:
    import fitz  # noqa: F401
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


@pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF not installed")
class TestPdfParser:
    def _make_minimal_pdf(self) -> Path:
        """Create a minimal valid PDF with one page of text using PyMuPDF."""
        import fitz

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_name = tmp.name
        tmp.close()
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "RENTAL AGREEMENT\n\n1. TERM\nThe lease runs for 12 months.")
        doc.save(tmp_name)
        doc.close()
        return Path(tmp_name)

    def test_pdf_extraction_returns_text(self):
        from backend.app.ingestion.pdf_parser import parse_pdf

        pdf_path = self._make_minimal_pdf()
        doc = parse_pdf(pdf_path, "pdf-doc-001")
        assert "RENTAL AGREEMENT" in doc.raw_text

    def test_pdf_span_invariant(self):
        from backend.app.ingestion.pdf_parser import parse_pdf

        pdf_path = self._make_minimal_pdf()
        doc = parse_pdf(pdf_path, "pdf-doc-002")
        for page_num, (start, end) in doc.page_spans.items():
            assert doc.raw_text[start:end] == doc.page_texts[page_num], (
                f"Span invariant violated for page {page_num}"
            )

    def test_pdf_metadata_page_count(self):
        from backend.app.ingestion.pdf_parser import parse_pdf

        pdf_path = self._make_minimal_pdf()
        doc = parse_pdf(pdf_path, "pdf-doc-003")
        assert doc.metadata.page_count >= 1

    def test_encrypted_pdf_raises(self):
        """Encrypted PDFs should raise PDFExtractionError cleanly."""
        from backend.app.ingestion.pdf_parser import parse_pdf, PDFExtractionError

        import fitz
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_name = tmp.name
        tmp.close()
        orig = fitz.open()
        orig.new_page()
        orig.save(tmp_name, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret")
        orig.close()
        with pytest.raises(PDFExtractionError, match="Encrypted"):
            parse_pdf(Path(tmp_name), "enc-doc")
