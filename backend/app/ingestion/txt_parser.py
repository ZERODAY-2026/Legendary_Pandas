"""
Plain text document parser using Python standard library.
Preserves page-equivalent boundaries, spans, and metadata.
"""

from pathlib import Path
from backend.app.document.schemas import DocumentMetadata, ExtractedDocument, SourceType


def parse_txt(file_path: Path, document_id: str) -> ExtractedDocument:
    """Parse TXT file into ExtractedDocument model."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    file_size = file_path.stat().st_size if file_path.exists() else len(content.encode("utf-8"))

    # TXT files represent a single logical page 1 by default
    page_texts = {1: content}
    page_spans = {1: (0, len(content))}

    metadata = DocumentMetadata(
        document_id=document_id,
        filename=file_path.name,
        file_type="txt",
        file_size_bytes=file_size,
        page_count=1,
        char_count=len(content),
        source_type=SourceType.CONTRACT,
    )

    return ExtractedDocument(
        metadata=metadata,
        raw_text=content,
        page_texts=page_texts,
        page_spans=page_spans,
    )
