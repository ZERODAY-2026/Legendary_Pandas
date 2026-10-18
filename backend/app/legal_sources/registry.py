"""
Legal source registry and strict provenance boundary.
Assigned at ingestion and NEVER inferred by the language model.
MVP scope: Rental agreements (SourceType.CONTRACT) only.
"""

from typing import Literal
from backend.app.document.schemas import SourceType


class LegalSourceRegistry:
    """Maintains registry of distinct source types and disallows conflation.

    In MVP, only 'contract' sources are active, but registry enforces separation.
    """

    ALLOWED_SOURCE_TYPES = {
        SourceType.CONTRACT: "User-uploaded rental contract or lease agreement.",
        SourceType.LEGISLATION: "Statutory laws and tenancy legislation (Reserved for future).",
        SourceType.COURT_JUDGMENT: "Judicial precedent and tenancy court rulings (Reserved for future).",
    }

    @classmethod
    def validate_source_assignment(cls, source_type: str | SourceType) -> SourceType:
        if isinstance(source_type, str):
            try:
                source_type = SourceType(source_type.lower())
            except ValueError:
                raise ValueError(f"Invalid source type '{source_type}'. Must be one of: {list(cls.ALLOWED_SOURCE_TYPES.keys())}")
        return source_type
