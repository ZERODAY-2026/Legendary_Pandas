"""Core Pydantic data schemas for Private Legal Triage.

Enforces deterministic rules on findings, source separation,
exact text span grounding, and escalation constraints.
"""

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Annotated, Optional
from pydantic import BaseModel, Field, model_validator


class SourceType(str, Enum):
    """Immutable source type assigned at ingestion and never inferred by the model."""
    CONTRACT = "contract"
    LEGISLATION = "legislation"
    COURT_JUDGMENT = "court_judgment"


class FindingType(str, Enum):
    """Classification of findings produced during triage."""
    EXPLICIT_STATEMENT = "explicit_statement"
    INTERPRETATION = "interpretation"
    MISSING_CLAUSE = "missing_clause"
    POTENTIAL_CONCERN = "potential_concern"


class TriageStatus(str, Enum):
    """Status of the evidence-grounded response."""
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ESCALATION_REQUIRED = "escalation_required"


# Banned phrases that constitute definitive judicial findings of illegality
DEFINITIVE_ILLEGALITY_PATTERNS = [
    re.compile(r"\b(is|are)\s+illegal\b", re.IGNORECASE),
    re.compile(r"\b(is|are)\s+unlawful\b", re.IGNORECASE),
    re.compile(r"\b(is|are)\s+strictly\s+void\b", re.IGNORECASE),
    re.compile(r"\b(is|are)\s+completely\s+unenforceable\b", re.IGNORECASE),
    re.compile(r"\b(violates|violating)\s+the\s+law\s+definitively\b", re.IGNORECASE),
    re.compile(r"\bconstitutes\s+an\s+illegal\s+act\b", re.IGNORECASE),
]


class ExtractedPage(BaseModel):
    """Represents a single page extracted locally from a document."""
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(..., description="Normalized text content of the page")
    raw_text: str = Field(..., description="Original verbatim text prior to normalization")
    char_offset: int = Field(..., ge=0, description="Cumulative character offset in full document")


class ExtractedDocument(BaseModel):
    """Complete document model preserved entirely in local memory/storage."""
    document_id: str = Field(..., description="Unique deterministic or UUID identifier")
    filename: str = Field(..., description="Original sanitized filename")
    file_type: str = Field(..., description="Extension (.pdf or .txt)")
    source_type: SourceType = Field(
        default=SourceType.CONTRACT,
        description="Assigned source type, strictly immutable"
    )
    file_size_bytes: int = Field(..., gt=0)
    pages: Annotated[list[ExtractedPage], Field(min_length=1)]
    full_text: str = Field(..., description="Verbatim concatenated original text")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_slice(self, start_char: int, end_char: int) -> str:
        """Returns the exact substring from the full document text."""
        if start_char < 0 or end_char > len(self.full_text) or start_char > end_char:
            raise ValueError(f"Invalid span indices: [{start_char}:{end_char}] for length {len(self.full_text)}")
        return self.full_text[start_char:end_char]


class DocumentSection(BaseModel):
    """Structural section detected in a legal document."""
    section_id: str
    document_id: str
    heading: Optional[str] = None
    clause_number: Optional[str] = None
    page_number: int = Field(..., ge=1)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    text: str


class Chunk(BaseModel):
    """Bounded chunk linked to exact source document span and page."""
    chunk_id: str
    document_id: str
    source_type: SourceType = SourceType.CONTRACT
    page_number: int = Field(..., ge=1)
    page_end: Annotated[Optional[int], Field(ge=1)] = None
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    clause_number: Optional[str] = None
    heading: Optional[str] = None

    def verify_exact_span(self, full_document_text: str) -> bool:
        """Verifies that slicing full_document_text by start_char:end_char matches text exactly."""
        if self.start_char < 0 or self.end_char > len(full_document_text) or self.start_char > self.end_char:
            return False
        return full_document_text[self.start_char:self.end_char] == self.text


class EvidencePassage(BaseModel):
    """Grounding evidence passage with verifiable provenance."""
    evidence_id: str
    document_id: str
    source_type: SourceType
    page_number: int = Field(..., ge=1)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    exact_quote: str = Field(..., min_length=1)
    clause_number: Optional[str] = None
    relevance_score: Optional[float] = None

    def verify_verbatim(self, full_document_text: str) -> bool:
        """Verifies that exact_quote is genuinely present at the specified span."""
        if self.start_char < 0 or self.end_char > len(full_document_text) or self.start_char > self.end_char:
            return False
        return full_document_text[self.start_char:self.end_char] == self.exact_quote


class Finding(BaseModel):
    """Deterministic legal triage finding.
    
    Enforces rules 1-6:
    1. Explicit statements must have verifiable evidence passages.
    2. Interpretations must be explicitly labeled as interpretations.
    3. Missing clauses must not claim the clause exists.
    4. Potential concerns must NOT make definitive findings of illegality.
    5. Evidence source must match expected source type (no mixing statute as contract).
    """
    finding_id: str
    finding_type: FindingType
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    is_interpretation: bool = False
    exists_in_document: bool = True
    evidence: list[EvidencePassage] = Field(default_factory=list)
    definitive_illegality_claim: bool = False
    severity: Optional[str] = Field(default="medium", pattern="^(low|medium|high)$")
    reasoning: Optional[str] = None
    escalation_recommended: bool = False

    @model_validator(mode="after")
    def enforce_deterministic_rules(self) -> "Finding":
        # Rule 1: An explicit statement must be supported by a real passage from the source document
        if self.finding_type == FindingType.EXPLICIT_STATEMENT:
            if not self.evidence:
                raise ValueError("Rule 1 Violation: An explicit statement must be supported by at least one evidence passage.")
            if self.is_interpretation:
                raise ValueError("Rule 1 & 2 Violation: An explicit statement cannot be marked as an interpretation.")

        # Rule 2: An interpretation must be labelled as an interpretation
        if self.finding_type == FindingType.INTERPRETATION:
            if not self.is_interpretation:
                raise ValueError("Rule 2 Violation: Findings of type 'interpretation' must have is_interpretation=True.")

        # Rule 3: A missing clause must not be described as one that exists
        if self.finding_type == FindingType.MISSING_CLAUSE:
            if self.exists_in_document:
                raise ValueError("Rule 3 Violation: A missing clause cannot have exists_in_document=True.")
            if self.evidence:
                raise ValueError("Rule 3 Violation: A missing clause cannot cite existing contract passages as proof of existence.")

        # Rule 4: A legal concern must not be a definitive finding of illegality
        if self.finding_type == FindingType.POTENTIAL_CONCERN:
            if self.definitive_illegality_claim:
                raise ValueError("Rule 4 Violation: A legal concern must not be a definitive finding of illegality.")
            for pattern in DEFINITIVE_ILLEGALITY_PATTERNS:
                if pattern.search(self.title) or pattern.search(self.description):
                    raise ValueError(
                        f"Rule 4 Violation: Finding text contains definitive illegality claim matching '{pattern.pattern}'. "
                        "Must be phrased as a potential concern, risk, or inquiry for legal counsel."
                    )

        # Rule 5: Source separation check on evidence
        for ev in self.evidence:
            if ev.source_type != SourceType.CONTRACT:
                raise ValueError(
                    f"Rule 5 Violation: Uploaded contract evidence cannot cite non-contract source '{ev.source_type}'."
                )

        return self


class OverviewClauseSummary(BaseModel):
    """High-level summary of a key clause group (deposit, notice, term, etc.)."""
    category: str
    clause_number: Optional[str] = None
    summary_text: str
    evidence: list[EvidencePassage] = Field(default_factory=list)
    concerns: list[Finding] = Field(default_factory=list)
    is_missing: bool = False


class DocumentOverview(BaseModel):
    """Comprehensive structured contract overview."""
    document_id: str
    filename: str
    page_count: int
    key_clauses: list[OverviewClauseSummary]
    potential_concerns: list[Finding]
    missing_critical_clauses: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QuestionRequest(BaseModel):
    """User question submission."""
    document_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=2)


class TriageResponse(BaseModel):
    """Triage response complying with evidence-grounded and escalation rules."""
    status: TriageStatus
    answer: str
    evidence: list[EvidencePassage] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    insufficient_evidence_details: Optional[str] = None
    escalation_required: bool = False
    escalation_message: Optional[str] = None

    @model_validator(mode="after")
    def enforce_response_rules(self) -> "TriageResponse":
        # Rule 6: If evidence is insufficient, the answer must state what could not be established
        if self.status == TriageStatus.INSUFFICIENT_EVIDENCE:
            if not self.insufficient_evidence_details:
                raise ValueError(
                    "Rule 6 Violation: When status is 'insufficient_evidence', "
                    "insufficient_evidence_details must explicitly state what could not be established."
                )

        # Escalation consistency
        if self.status == TriageStatus.ESCALATION_REQUIRED:
            if not self.escalation_required or not self.escalation_message:
                raise ValueError(
                    "When status is 'escalation_required', escalation_required must be True "
                    "and an escalation_message to a legal professional must be provided."
                )

        return self
