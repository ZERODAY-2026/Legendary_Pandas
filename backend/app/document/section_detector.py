"""
Section, clause number, and heading detector for rental agreements.

Detects all common legal document heading patterns:
  - Numbered clauses:  "1.", "1.2.", "1.2.3"
  - Named sections:    "Section 3", "Article IV", "Clause 5.1"
  - Uppercase headers: "RENT PAYMENT", "SECURITY DEPOSIT"
  - Roman numerals:    "III. TERM"

Returns DetectedSection objects with char-accurate start/end offsets.
The page_num field is set to 1 by default; callers that have page_spans
can call `detect_sections_with_pages(text, page_spans)` for accurate values.
"""

import re
from pydantic import BaseModel


class DetectedSection(BaseModel):
    clause_id: str | None    # Normalized clause identifier, e.g. "1.3" or "Section 4"
    title: str               # Human-readable heading text
    start_char: int          # Inclusive offset of the heading start in the full text
    end_char: int            # Exclusive offset of the heading end (heading only, not body)
    page_num: int            # 1-indexed page (approximate; set by caller if available)


# ─── Compiled patterns, ordered from most-specific to least-specific ─────────
_PATTERNS: list[re.Pattern[str]] = [
    # "Section 3.2 RENT PAYMENT" / "Clause 4.1 Termination"
    re.compile(
        r"(?:^|\n)(?P<clause>(?:Section|Clause|Article|Part)\s+\d+(?:\.\d+)*\.?)"
        r"\s+(?P<title>[A-Z][A-Z0-9 ,;:\-]{2,60})(?=\s|$)",
        re.MULTILINE,
    ),
    # Plain numbered: "1. TERM", "3.2. Security Deposit", "10.1 Governing Law"
    re.compile(
        r"(?:^|\n)(?P<clause>\d+(?:\.\d+)*\.?)\s+"
        r"(?P<title>[A-Z][A-Za-z0-9 ,;:\-]{2,60})(?=\s|$)",
        re.MULTILINE,
    ),
    # Roman numeral: "IV. PETS AND ANIMALS"
    re.compile(
        r"(?:^|\n)(?P<clause>[IVXLCDMivxlcdm]{1,6}\.)\s+"
        r"(?P<title>[A-Z][A-Z0-9 ,;:\-]{2,60})(?=\s|$)",
        re.MULTILINE,
    ),
    # All-caps standalone heading on its own line (≥ 3 words, no trailing colon)
    re.compile(
        r"(?:^|\n)(?P<title>(?:[A-Z]{2,}[ \t]){2,}[A-Z]{2,})(?=\s|$)",
        re.MULTILINE,
    ),
]


def detect_sections(text: str, page_num: int = 1) -> list[DetectedSection]:
    """Detect numbered clauses and major contract headings within text.

    Args:
        text: The full raw text of the document or a single page.
        page_num: Page number to assign to all detected sections (default 1).
                  For multi-page accuracy, use detect_sections_with_pages instead.

    Returns:
        List of DetectedSection sorted by start_char, deduplicated.
    """
    seen_starts: set[int] = set()
    sections: list[DetectedSection] = []

    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start = match.start()
            # Offset past the leading newline character if present
            if text[start : start + 1] == "\n":
                start += 1

            if start in seen_starts:
                continue
            seen_starts.add(start)

            clause = match.groupdict().get("clause")
            raw_title = match.groupdict().get("title", "")
            title = raw_title.strip() if raw_title else ""

            if not title:
                continue

            sections.append(
                DetectedSection(
                    clause_id=clause.strip().rstrip(".") if clause else None,
                    title=title,
                    start_char=start,
                    end_char=match.end(),
                    page_num=page_num,
                )
            )

    sections.sort(key=lambda s: s.start_char)
    return sections


def detect_sections_with_pages(
    text: str,
    page_spans: dict[int, tuple[int, int]],
) -> list[DetectedSection]:
    """Detect sections and assign accurate page numbers using page_spans offsets."""
    raw_sections = detect_sections(text)
    for section in raw_sections:
        for page_num, (span_start, span_end) in page_spans.items():
            if span_start <= section.start_char < span_end:
                object.__setattr__(section, "page_num", page_num)
                break
    return raw_sections

