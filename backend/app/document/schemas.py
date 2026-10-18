from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class SourceType(str, Enum):
    CONTRACT = "contract"
    LEGISLATION = "legislation"
    COURT_JUDGMENT = "court_judgment"


class Chunk(BaseModel):
    source_id: str                         # stable ID: "{document_id}::chunk-{index:04d}"
    document_id: str
    source_type: SourceType
    page_start: int
    page_end: int
    clause_id: str | None = None           # detected clause label (e.g. "1.3" or "TERM")
    parent_clause_id: str | None = None    # nearest ancestor clause for sub-chunks
    text: str
    span_start: int                        # inclusive offset into original raw_text
    span_end: int                          # exclusive offset into original raw_text
    chunk_index: int = 0                   # sequential index within the document
    index_version: str = "v1"             # bumped on re-index; used by retrieval cache


class Evidence(BaseModel):
    source_id: str
    source_type: Literal["contract", "legislation", "court_judgment"]
    page: int | None = None
    section: str | None = None
    quote: str
    # Retrieval diagnostic metadata (populated by retrieve_evidence)
    retrieval_rank: int | None = None      # 1-indexed rank in the result list
    rrf_score: float | None = None         # Reciprocal Rank Fusion score
    retrieval_method: str | None = None    # "hybrid", "vector", or "bm25"
    index_version: str | None = None       # matches Chunk.index_version


class EvidenceResult(BaseModel):
    """Public interface returned by retrieve_evidence() — consumed by Member 2's agent."""

    document_id: str
    question: str
    passages: list["Evidence"] = Field(default_factory=list)
    is_sufficient: bool = False
    reason: str | None = None              # why evidence was insufficient, if applicable
    index_version: str = "v1"             # version of the Qdrant index used


class Finding(BaseModel):
    title: str
    category: Literal[
        "obligation",
        "right",
        "deadline",
        "penalty",
        "termination",
        "ambiguity",
        "missing_information",
        "potential_legal_concern",
    ]
    explicit_text: str | None = None
    plain_language_explanation: str
    interpretation_or_concern: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_finding_rules(self) -> "Finding":
        """Deterministic safety and consistency rules for every Finding:

        1. An explicit statement must be supported by evidence.
        2. Missing information category must have no fabricated explicit text.
        3. A legal concern must never contain definitive claims of illegality or unenforceability.
        """
        # Rule 2: Missing information must not claim an existing explicit clause
        if self.category == "missing_information":
            if self.explicit_text is not None and len(self.explicit_text.strip()) > 0:
                raise ValueError(
                    "A missing information finding cannot describe an explicit clause that does not exist."
                )

        # Rule 1: Explicit statements require backing evidence
        if self.explicit_text is not None and len(self.explicit_text.strip()) > 0:
            if not self.evidence:
                raise ValueError(
                    f"Explicit statement '{self.title}' must be supported by at least one evidence item."
                )

        # Rule 3: Legal concerns cannot state definitive illegality
        if self.category == "potential_legal_concern":
            disallowed_definitive_terms = [
                "is illegal",
                "is unlawful",
                "is void",
                "is completely unenforceable",
                "is strictly prohibited and invalid",
                "violates the law definitively",
            ]
            combined_text = (
                f"{self.title} {self.plain_language_explanation} {self.interpretation_or_concern or ''}".lower()
            )
            for term in disallowed_definitive_terms:
                if term in combined_text:
                    raise ValueError(
                        f"Legal concern cannot make a definitive finding of illegality: detected '{term}'. "
                        "Must state as a potential concern with recommendation to consult a qualified legal professional."
                    )

        return self


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_type: Literal["pdf", "txt", "json"]
    file_size_bytes: int
    page_count: int
    char_count: int
    source_type: SourceType = SourceType.CONTRACT


class ExtractedDocument(BaseModel):
    metadata: DocumentMetadata
    raw_text: str
    page_texts: dict[int, str]          # 1-indexed page number -> text on that page
    page_spans: dict[int, tuple[int, int]]  # 1-indexed page -> (span_start, span_end)
    index_version: str = "v1"          # incremented on re-ingestion; used for cache keys


class QueryRequest(BaseModel):
    session_id: str
    question: str


class QueryResponse(BaseModel):
    answer: str
    status: Literal["answered_with_evidence", "insufficient_evidence", "escalation_required"]
    findings: list[Finding] = Field(default_factory=list)
    evidence_used: list[Evidence] = Field(default_factory=list)
    disclaimer: str = (
        "This analysis is for informational triage only and does not constitute legal advice. "
        "Questions regarding enforceability or legal rights should be escalated to a qualified attorney."
    )
