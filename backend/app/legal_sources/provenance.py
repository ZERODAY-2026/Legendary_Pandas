"""
Provenance tracking to guarantee that contract text and any external source
are strictly compartmentalized and never cross-pollinated.
"""

from backend.app.document.schemas import Evidence, SourceType


class ProvenanceTracker:
    """Verifies that evidence items maintain strict provenance to their declared source."""

    @staticmethod
    def verify_provenance(evidence: Evidence, expected_source_id: str, expected_source_type: SourceType) -> bool:
        """Ensure evidence item is strictly bound to the target document and source type."""
        if evidence.source_id != expected_source_id:
            return False
        if evidence.source_type != expected_source_type.value:
            return False
        return True
