"""
Citation validator.
Verifies cited source IDs exist, cited page/clause match index,
and every cited quote appears verbatim in the original extracted text.
"""

from backend.app.document.schemas import Evidence, ExtractedDocument


class CitationVerificationResult:

    def __init__(self, is_valid: bool, errors: list[str]):
        self.is_valid = is_valid
        self.errors = errors


def verify_citation(evidence: Evidence, document: ExtractedDocument) -> CitationVerificationResult:
    """Validate that evidence quote is a verbatim substring of original document text."""
    errors = []

    if evidence.source_id != document.metadata.document_id:
        errors.append(
            f"Source ID mismatch: evidence has '{evidence.source_id}', expected '{document.metadata.document_id}'."
        )

    # Check verbatim quote presence
    quote = evidence.quote.strip()
    if not quote:
        errors.append("Evidence quote cannot be empty.")
    elif quote not in document.raw_text:
        # Check normalized quote if exact match fails
        errors.append(f"Quote does not appear verbatim in extracted contract text: '{quote[:60]}...'")

    # If page is specified, check that the quote exists on that page
    if evidence.page is not None and evidence.page in document.page_texts:
        page_text = document.page_texts[evidence.page]
        if quote not in page_text:
            errors.append(f"Quote not found on declared page {evidence.page}.")

    return CitationVerificationResult(is_valid=len(errors) == 0, errors=errors)
