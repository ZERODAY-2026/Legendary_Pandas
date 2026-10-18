"""Unit tests for core schemas and deterministic rule enforcement."""

import pytest
from pydantic import ValidationError
from backend.app.document.schemas import (
    Chunk,
    EvidencePassage,
    ExtractedDocument,
    ExtractedPage,
    Finding,
    FindingType,
    QuestionRequest,
    SourceType,
    TriageResponse,
    TriageStatus,
)


def test_source_types():
    """Verify the three immutable source types exist."""
    assert SourceType.CONTRACT == "contract"
    assert SourceType.LEGISLATION == "legislation"
    assert SourceType.COURT_JUDGMENT == "court_judgment"


def test_chunk_span_verification():
    """Verify that chunk text strictly matches sliced document text."""
    full_text = "Section 1: The security deposit shall be $2,500 payable on signing."
    # "The security deposit shall be $2,500" starts at index 11 and ends at 47
    start_char = 11
    end_char = 47
    expected_text = full_text[start_char:end_char]
    assert expected_text == "The security deposit shall be $2,500"

    valid_chunk = Chunk(
        chunk_id="chk_001",
        document_id="doc_001",
        source_type=SourceType.CONTRACT,
        page_number=1,
        start_char=start_char,
        end_char=end_char,
        text=expected_text,
    )
    assert valid_chunk.verify_exact_span(full_text) is True

    # Tampered text fails verification
    tampered_chunk = Chunk(
        chunk_id="chk_002",
        document_id="doc_001",
        source_type=SourceType.CONTRACT,
        page_number=1,
        start_char=start_char,
        end_char=end_char,
        text="The security deposit shall be $5,000",
    )
    assert tampered_chunk.verify_exact_span(full_text) is False


def test_rule_1_explicit_statement_requires_evidence():
    """Rule 1: An explicit statement must be supported by a real passage from the source document."""
    # Attempting to create explicit statement without evidence must raise ValidationError
    with pytest.raises(ValidationError, match="Rule 1 Violation"):
        Finding(
            finding_id="f_001",
            finding_type=FindingType.EXPLICIT_STATEMENT,
            title="Deposit is $2,500",
            description="Tenant must pay $2,500 upon lease signing.",
            is_interpretation=False,
            evidence=[],  # Missing evidence
        )

    # Valid explicit statement with evidence passage
    ev = EvidencePassage(
        evidence_id="ev_001",
        document_id="doc_001",
        source_type=SourceType.CONTRACT,
        page_number=1,
        start_char=11,
        end_char=48,
        exact_quote="The security deposit shall be $2,500",
    )
    valid_finding = Finding(
        finding_id="f_001",
        finding_type=FindingType.EXPLICIT_STATEMENT,
        title="Deposit is $2,500",
        description="Tenant must pay $2,500 upon lease signing.",
        is_interpretation=False,
        evidence=[ev],
    )
    assert valid_finding.finding_id == "f_001"


def test_rule_2_interpretation_must_be_labelled():
    """Rule 2: An interpretation must be labelled as an interpretation."""
    with pytest.raises(ValidationError, match="Rule 2 Violation"):
        Finding(
            finding_id="f_002",
            finding_type=FindingType.INTERPRETATION,
            title="Ambiguous repair obligation",
            description="Clause implies tenant is liable for wear-and-tear.",
            is_interpretation=False,  # Not marked as interpretation!
        )

    valid_interpretation = Finding(
        finding_id="f_002",
        finding_type=FindingType.INTERPRETATION,
        title="Ambiguous repair obligation",
        description="Clause implies tenant is liable for wear-and-tear.",
        is_interpretation=True,
    )
    assert valid_interpretation.is_interpretation is True


def test_rule_3_missing_clause_cannot_claim_existence():
    """Rule 3: A missing clause must not be described as one that exists."""
    with pytest.raises(ValidationError, match="Rule 3 Violation"):
        Finding(
            finding_id="f_003",
            finding_type=FindingType.MISSING_CLAUSE,
            title="Notice period missing",
            description="Agreement omits statutory notice period.",
            exists_in_document=True,  # Contradiction!
        )

    valid_missing = Finding(
        finding_id="f_003",
        finding_type=FindingType.MISSING_CLAUSE,
        title="Notice period missing",
        description="Agreement omits statutory notice period.",
        exists_in_document=False,
        evidence=[],
    )
    assert valid_missing.exists_in_document is False


def test_rule_4_legal_concern_cannot_be_definitive_illegality():
    """Rule 4: A legal concern must not be a definitive finding of illegality."""
    # Direct flag violation
    with pytest.raises(ValidationError, match="Rule 4 Violation"):
        Finding(
            finding_id="f_004",
            finding_type=FindingType.POTENTIAL_CONCERN,
            title="Illegal entry clause",
            description="Landlord entry without 24h notice.",
            definitive_illegality_claim=True,
        )

    # Text containing banned phrase "is illegal"
    with pytest.raises(ValidationError, match="Rule 4 Violation"):
        Finding(
            finding_id="f_004b",
            finding_type=FindingType.POTENTIAL_CONCERN,
            title="Landlord access clause is illegal",
            description="The clause violates tenant privacy rights.",
            definitive_illegality_claim=False,
        )

    # Properly framed potential concern (compliant)
    valid_concern = Finding(
        finding_id="f_004c",
        finding_type=FindingType.POTENTIAL_CONCERN,
        title="Potential concern regarding landlord right of entry",
        description="The clause allows immediate landlord entry without advance notice, which may conflict with standard statutory 24-hour notice guidelines. Consultation with a legal professional is advised.",
        definitive_illegality_claim=False,
        escalation_recommended=True,
    )
    assert valid_concern.escalation_recommended is True


def test_rule_5_source_separation_no_mixing():
    """Rule 5: Legislation and judgments must never be used as if they were the uploaded contract."""
    statute_evidence = EvidencePassage(
        evidence_id="ev_statute",
        document_id="doc_statute",
        source_type=SourceType.LEGISLATION,  # Legislation passed as contract finding evidence!
        page_number=1,
        start_char=0,
        end_char=30,
        exact_quote="Section 19: Security deposits",
    )
    with pytest.raises(ValidationError, match="Rule 5 Violation"):
        Finding(
            finding_id="f_005",
            finding_type=FindingType.EXPLICIT_STATEMENT,
            title="Contract deposit requirement",
            description="Deposit rule as stated.",
            is_interpretation=False,
            evidence=[statute_evidence],
        )


def test_rule_6_insufficient_evidence_requires_details():
    """Rule 6: If evidence is insufficient, the answer must state what could not be established."""
    with pytest.raises(ValidationError, match="Rule 6 Violation"):
        TriageResponse(
            status=TriageStatus.INSUFFICIENT_EVIDENCE,
            answer="The document does not state anything about pets.",
            insufficient_evidence_details=None,  # Missing details!
        )

    valid_insufficient = TriageResponse(
        status=TriageStatus.INSUFFICIENT_EVIDENCE,
        answer="The agreement does not provide information regarding pet policies.",
        insufficient_evidence_details="Searched clauses across all 5 pages; no mention of 'pets', 'animals', or 'domestic pets' was established in the contract text.",
    )
    assert valid_insufficient.status == TriageStatus.INSUFFICIENT_EVIDENCE


def test_escalation_enforcement():
    """Questions needing a definitive legal judgment must trigger escalation."""
    with pytest.raises(ValidationError, match="escalation_message to a legal professional must be provided"):
        TriageResponse(
            status=TriageStatus.ESCALATION_REQUIRED,
            answer="Evaluating the enforceability of this clause requires legal counsel.",
            escalation_required=True,
            escalation_message=None,  # Missing required message!
        )

    valid_escalation = TriageResponse(
        status=TriageStatus.ESCALATION_REQUIRED,
        answer="Determining whether this waiver is legally enforceable requires formal legal analysis of local tenancy legislation and judicial precedents.",
        escalation_required=True,
        escalation_message="This system provides document triage only and cannot offer legal judgments or determinations of enforceability. Please consult a qualified tenant attorney or local legal aid clinic.",
    )
    assert valid_escalation.escalation_required is True
