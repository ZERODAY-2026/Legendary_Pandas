"""
Unit tests for project structure, sample contracts, and privacy requirements (Phase 1).
"""

from pathlib import Path
from backend.app.config import settings
from backend.app.document.schemas import SourceType


def test_required_docs_exist():
    """Verify all Phase 1 documentation deliverables exist."""
    base_dir = settings.BASE_DIR
    docs_dir = base_dir / "docs"

    assert (docs_dir / "architecture.md").exists()
    assert (docs_dir / "privacy.md").exists()
    assert (docs_dir / "legal-source-policy.md").exists()
    assert (docs_dir / "evaluation.md").exists()
    assert (docs_dir / "model-card.md").exists()


def test_sample_contracts_exist():
    """Verify synthetic rental agreements are present."""
    base_dir = settings.BASE_DIR
    sample_dir = base_dir / "data" / "sample_contracts"

    assert (sample_dir / "standard_residential_lease.txt").exists()
    assert (sample_dir / "commercial_office_lease.txt").exists()
    assert (sample_dir / "adversarial_lease.txt").exists()


def test_privacy_and_threat_model_content():
    """Verify threat model and privacy rules are documented."""
    base_dir = settings.BASE_DIR
    privacy_doc = (base_dir / "docs" / "privacy.md").read_text(encoding="utf-8")

    assert "Zero-Egress" in privacy_doc
    assert "Threat Model" in privacy_doc
    assert "127.0.0.1" in privacy_doc
    assert "Prompt Injection" in privacy_doc


def test_python_and_node_pinned_files():
    """Verify runtime version pinning files exist."""
    base_dir = settings.BASE_DIR
    assert (base_dir / ".python-version").read_text().strip() == "3.11.9"
    assert (base_dir / ".nvmrc").read_text().strip() == "24.17.0"
