"""
Unit tests for core schemas and deterministic validation rules (Phase 1).
"""

import pytest
from pydantic import ValidationError
from backend.app.config import settings
from backend.app.document.schemas import (
    Chunk,
    Evidence,
    Finding,
    SourceType,
    QueryRequest,
    QueryResponse,
)


def test_config_loopback_and_privacy():
    """Verify strict loopback binding and disabled telemetry."""
    assert settings.HOST == "127.0.0.1"
    assert settings.ENABLE_TELEMETRY is False
    assert "http://127.0.0.1:8000" in settings.ALLOWED_ORIGINS
    assert settings.MAX_FILE_SIZE_BYTES is None


def test_source_types_distinct():
    """Verify the three required source types exist and are distinct."""
    assert SourceType.CONTRACT.value == "contract"
    assert SourceType.LEGISLATION.value == "legislation"
    assert SourceType.COURT_JUDGMENT.value == "court_judgment"


def test_chunk_creation():
    """Verify chunk model stores all required provenance and span offsets."""
    chunk = Chunk(
        source_id="contract_001",
        document_id="doc_123",
        source_type=SourceType.CONTRACT,
        page_start=1,
        page_end=1,
        clause_id="5. SECURITY DEPOSIT",
        text="Tenant shall deposit $1,500.00 as security deposit.",
        span_start=100,
        span_end=151,
    )
    assert chunk.document_id == "doc_123"
    assert chunk.source_type == SourceType.CONTRACT
    assert chunk.span_end - chunk.span_start == len(chunk.text)


def test_valid_finding_with_evidence():
    """Verify a valid Finding passes deterministic validation."""
    evidence = Evidence(
        source_id="doc_123",
        source_type="contract",
        page=1,
        section="5. SECURITY DEPOSIT",
        quote="Tenant shall deposit $1,500.00 as security deposit.",
    )
    finding = Finding(
        title="Security Deposit Obligation",
        category="obligation",
        explicit_text="Tenant shall deposit $1,500.00 as security deposit.",
        plain_language_explanation="The tenant is obligated to pay a $1,500 security deposit upon lease signing.",
        evidence=[evidence],
    )
    assert finding.title == "Security Deposit Obligation"
    assert len(finding.evidence) == 1


def test_rule_1_explicit_statement_requires_evidence():
    """Deterministic Rule 1: An explicit statement must be supported by real passage/evidence."""
    with pytest.raises(ValidationError) as exc_info:
        Finding(
            title="Fabricated Deposit Clause",
            category="obligation",
            explicit_text="Tenant must pay $2,000 immediately.",
            plain_language_explanation="Tenant has to pay $2,000.",
            evidence=[],  # Missing backing evidence
        )
    assert "must be supported by at least one evidence item" in str(exc_info.value)


def test_rule_2_missing_information_cannot_claim_explicit_text():
    """Deterministic Rule 2: A missing clause must not be described as one that exists."""
    with pytest.raises(ValidationError) as exc_info:
        Finding(
            title="Pet Policy Information",
            category="missing_information",
            explicit_text="Pets are permitted with no fee.",  # Contradiction
            plain_language_explanation="Pets policy missing.",
            evidence=[],
        )
    assert "cannot describe an explicit clause that does not exist" in str(exc_info.value)


def test_rule_3_potential_legal_concern_cannot_be_definitive_illegality():
    """Deterministic Rule 3: A legal concern must not be a definitive finding of illegality."""
    with pytest.raises(ValidationError) as exc_info:
        Finding(
            title="Deposit Clause",
            category="potential_legal_concern",
            plain_language_explanation="This penalty is illegal under state law.",
            evidence=[],
        )
    assert "cannot make a definitive finding of illegality" in str(exc_info.value)


def test_query_response_defaults():
    """Verify default query response initialization and disclaimer."""
    resp = QueryResponse(
        answer="The contract does not establish pet regulations.",
        status="insufficient_evidence",
        findings=[],
    )
    assert resp.status == "insufficient_evidence"
    assert "informational triage only" in resp.disclaimer
