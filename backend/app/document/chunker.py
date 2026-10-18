"""
Contract-aware chunker with clause detection and overlapping sub-chunks.

Pipeline:
  1. Detect clause/section boundaries using section_detector.
  2. For each clause segment, split with LangChain RecursiveCharacterTextSplitter
     using the configured chunk_size / overlap.
  3. Preserve exact span offsets so raw_text[chunk.span_start:chunk.span_end] == chunk.text
     for every chunk (the span invariant).
  4. Carry forward: document_id, source_type, page_start, page_end, clause_id, parent_clause_id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.app.config import settings
from backend.app.document.schemas import Chunk, ExtractedDocument, SourceType
from backend.app.document.section_detector import detect_sections


# ─── Separators ordered from coarsest to finest (legal-document aware) ──────
_LEGAL_SEPARATORS = [
    r"\n\n",          # blank line (paragraph / clause boundary)
    r"\n(?=\d+\.)",   # new numbered clause on next line
    r"\n",            # single newline
    r"(?<=\.)\s",     # sentence boundary
    r"\s",            # any whitespace
    r"",              # character level (last resort)
]


def _page_for_offset(offset: int, page_spans: dict[int, tuple[int, int]]) -> int:
    """Return the 1-indexed page number that contains the given character offset."""
    for page_num, (start, end) in page_spans.items():
        if start <= offset < end:
            return page_num
    # Fallback: return the last page
    return max(page_spans.keys(), default=1)


@dataclass
class _ClauseSegment:
    """A contiguous region of raw_text belonging to one detected clause."""
    clause_id: str | None
    title: str
    raw_start: int   # inclusive offset into raw_text
    raw_end: int     # exclusive offset into raw_text


def _build_clause_segments(
    raw_text: str, page_spans: dict[int, tuple[int, int]]
) -> list[_ClauseSegment]:
    """
    Merge detected sections into non-overlapping segments that cover the full text.
    Any gap before the first detected clause becomes an anonymous preamble segment.
    """
    sections = detect_sections(raw_text)
    segments: list[_ClauseSegment] = []

    cursor = 0
    for i, section in enumerate(sections):
        section_start = section.start_char
        # Gap before this section → preamble / between-clause text
        if cursor < section_start and (section_start - cursor) > settings.MIN_CHUNK_CHARS:
            segments.append(
                _ClauseSegment(
                    clause_id=None,
                    title="preamble" if cursor == 0 else "inter-clause",
                    raw_start=cursor,
                    raw_end=section_start,
                )
            )

        # Next section starts the end boundary of this one
        next_start = sections[i + 1].start_char if i + 1 < len(sections) else len(raw_text)
        segments.append(
            _ClauseSegment(
                clause_id=section.clause_id,
                title=section.title,
                raw_start=section_start,
                raw_end=next_start,
            )
        )
        cursor = next_start

    # Trailing text after last section
    if cursor < len(raw_text) and (len(raw_text) - cursor) > settings.MIN_CHUNK_CHARS:
        segments.append(
            _ClauseSegment(
                clause_id=None,
                title="appendix",
                raw_start=cursor,
                raw_end=len(raw_text),
            )
        )

    # If no sections were detected at all, treat the whole document as one segment
    if not segments:
        segments.append(
            _ClauseSegment(
                clause_id=None,
                title="document",
                raw_start=0,
                raw_end=len(raw_text),
            )
        )

    return segments


def _split_segment_with_overlap(
    segment: _ClauseSegment,
    raw_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """
    Split a clause segment into overlapping sub-chunks.

    Returns a list of (text, abs_start, abs_end) tuples where abs_start/abs_end
    are absolute offsets into raw_text.  The span invariant is preserved:
        raw_text[abs_start:abs_end] == text
    """
    segment_text = raw_text[segment.raw_start : segment.raw_end]

    if len(segment_text) <= chunk_size:
        return [(segment_text, segment.raw_start, segment.raw_end)]

    splitter = RecursiveCharacterTextSplitter(
        separators=_LEGAL_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=True,
        keep_separator=True,
    )
    lc_chunks = splitter.split_text(segment_text)

    result: list[tuple[str, int, int]] = []
    search_from = 0  # tracks position within segment_text to find each chunk
    for chunk_text in lc_chunks:
        if not chunk_text.strip():
            continue
        # Locate this chunk inside the segment text (first occurrence after search_from)
        rel_start = segment_text.find(chunk_text, search_from)
        if rel_start == -1:
            # Fuzzy fallback: strip and try again
            stripped = chunk_text.strip()
            rel_start = segment_text.find(stripped, search_from)
            if rel_start == -1:
                continue
            chunk_text = stripped

        rel_end = rel_start + len(chunk_text)
        abs_start = segment.raw_start + rel_start
        abs_end = segment.raw_start + rel_end
        result.append((chunk_text, abs_start, abs_end))
        # Move forward by chunk_size - overlap to allow the next search to start
        # slightly before rel_end (accounts for overlap)
        search_from = max(rel_start, rel_end - chunk_overlap)

    return result


def chunk_document(
    extracted_doc: ExtractedDocument,
    max_chunk_chars: int | None = None,
    min_chunk_chars: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Clause-aware, overlapping chunker for legal documents.

    Args:
        extracted_doc: Output from pdf_parser or txt_parser.
        max_chunk_chars: Override for settings.CHUNK_SIZE_CHARS.
        min_chunk_chars: Override for settings.MIN_CHUNK_CHARS.
        chunk_overlap: Override for settings.CHUNK_OVERLAP_CHARS.

    Returns:
        List of Chunk objects. Each satisfies:
            extracted_doc.raw_text[c.span_start:c.span_end] == c.text
    """
    chunk_size = max_chunk_chars or settings.CHUNK_SIZE_CHARS
    overlap = chunk_overlap or settings.CHUNK_OVERLAP_CHARS
    min_chars = min_chunk_chars or settings.MIN_CHUNK_CHARS

    raw_text = extracted_doc.raw_text
    if not raw_text.strip():
        return []

    doc_id = extracted_doc.metadata.document_id
    source_type = extracted_doc.metadata.source_type
    page_spans = extracted_doc.page_spans

    segments = _build_clause_segments(raw_text, page_spans)
    chunks: list[Chunk] = []
    chunk_index = 0

    for segment in segments:
        sub_chunks = _split_segment_with_overlap(segment, raw_text, chunk_size, overlap)

        for chunk_text, abs_start, abs_end in sub_chunks:
            if len(chunk_text.strip()) < min_chars:
                if not chunks and segment == segments[-1] and (chunk_text, abs_start, abs_end) == sub_chunks[-1]:
                    pass
                else:
                    continue

            page_s = _page_for_offset(abs_start, page_spans)
            page_e = _page_for_offset(max(abs_end - 1, abs_start), page_spans)

            # Build a stable source_id: doc + sequential index
            source_id = f"{doc_id}::chunk-{chunk_index:04d}"
            clause_label = segment.clause_id or segment.title

            chunks.append(
                Chunk(
                    source_id=source_id,
                    document_id=doc_id,
                    source_type=source_type,
                    page_start=page_s,
                    page_end=page_e,
                    clause_id=clause_label,
                    parent_clause_id=segment.clause_id,
                    text=chunk_text,
                    span_start=abs_start,
                    span_end=abs_end,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    return chunks

