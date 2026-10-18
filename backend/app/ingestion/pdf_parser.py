"""
PDF document parser using PyMuPDF (fitz).
Extracts text page-by-page, retaining page numbers and exact character spans.
Fails safely on empty, encrypted, corrupted, and image-only PDFs.
"""

from pathlib import Path
from backend.app.document.schemas import DocumentMetadata, ExtractedDocument, SourceType


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails due to corruption, encryption, or absence of text."""

    pass


def parse_pdf(file_path: Path, document_id: str) -> ExtractedDocument:
    """Parse PDF file using PyMuPDF locally and preserve page offsets.

    Will be fully tested and verified in Phase 2.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise PDFExtractionError("PyMuPDF (fitz) is required for local PDF extraction.")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise PDFExtractionError(f"Corrupted or invalid PDF file: {e}")

    if doc.is_encrypted:
        raise PDFExtractionError("Encrypted PDFs are not supported. Please provide an unencrypted document.")

    if len(doc) == 0:
        raise PDFExtractionError("PDF contains no pages.")

    page_texts: dict[int, str] = {}
    page_spans: dict[int, tuple[int, int]] = {}
    cumulative_text: list[str] = []
    current_offset = 0
    total_non_whitespace_chars = 0

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        text = page.get_text("text")

        total_non_whitespace_chars += len(text.strip())
        page_texts[page_num] = text
        start_span = current_offset
        end_span = start_span + len(text)
        page_spans[page_num] = (start_span, end_span)

        cumulative_text.append(text)
        current_offset = end_span

    if total_non_whitespace_chars == 0:
        raise PDFExtractionError(
            "PDF contains no extractable text (image-only or scanned document). "
            "OCR is not yet supported in this offline MVP."
        )

    full_raw_text = "".join(cumulative_text)
    file_size = file_path.stat().st_size if file_path.exists() else 0

    metadata = DocumentMetadata(
        document_id=document_id,
        filename=file_path.name,
        file_type="pdf",
        file_size_bytes=file_size,
        page_count=len(doc),
        char_count=len(full_raw_text),
        source_type=SourceType.CONTRACT,
    )

    return ExtractedDocument(
        metadata=metadata,
        raw_text=full_raw_text,
        page_texts=page_texts,
        page_spans=page_spans,
    )
