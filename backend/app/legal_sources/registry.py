"""Source registry maintaining immutable provenance and source-separation boundaries.

Enforces strict separation between uploaded contracts, statutory legislation,
and judicial precedents.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field
from backend.app.document.schemas import SourceType


class SourceMetadata(BaseModel):
    """Metadata recorded at ingestion for provenance tracking."""
    source_id: str
    source_type: SourceType
    title: str
    jurisdiction: Optional[str] = None
    ingestion_timestamp: str
    content_hash: str
    is_active_contract: bool = False


class LegalSourceRegistry:
    """Registry maintaining registered sources.
    
    Ensures that source types are assigned strictly at ingestion
    and can NEVER be modified or inferred by model generation.
    """

    def __init__(self):
        self._sources: Dict[str, SourceMetadata] = {}

    def register_contract(self, source_id: str, title: str, content_hash: str, timestamp: str) -> SourceMetadata:
        """Register an uploaded contract with immutable SourceType.CONTRACT."""
        meta = SourceMetadata(
            source_id=source_id,
            source_type=SourceType.CONTRACT,
            title=title,
            ingestion_timestamp=timestamp,
            content_hash=content_hash,
            is_active_contract=True,
        )
        self._sources[source_id] = meta
        return meta

    def register_legislation(self, source_id: str, title: str, jurisdiction: str, content_hash: str, timestamp: str) -> SourceMetadata:
        """Register a statutory source with immutable SourceType.LEGISLATION."""
        meta = SourceMetadata(
            source_id=source_id,
            source_type=SourceType.LEGISLATION,
            title=title,
            jurisdiction=jurisdiction,
            ingestion_timestamp=timestamp,
            content_hash=content_hash,
            is_active_contract=False,
        )
        self._sources[source_id] = meta
        return meta

    def register_court_judgment(self, source_id: str, title: str, jurisdiction: str, content_hash: str, timestamp: str) -> SourceMetadata:
        """Register a judicial precedent with immutable SourceType.COURT_JUDGMENT."""
        meta = SourceMetadata(
            source_id=source_id,
            source_type=SourceType.COURT_JUDGMENT,
            title=title,
            jurisdiction=jurisdiction,
            ingestion_timestamp=timestamp,
            content_hash=content_hash,
            is_active_contract=False,
        )
        self._sources[source_id] = meta
        return meta

    def get_source(self, source_id: str) -> Optional[SourceMetadata]:
        return self._sources.get(source_id)

    def get_source_type(self, source_id: str) -> Optional[SourceType]:
        meta = self._sources.get(source_id)
        return meta.source_type if meta else None

    def assert_is_contract(self, source_id: str) -> None:
        """Verify that a source is an uploaded contract and not external legal text."""
        source_type = self.get_source_type(source_id)
        if source_type != SourceType.CONTRACT:
            raise ValueError(f"Security/Provenance Error: Source '{source_id}' is '{source_type}', expected 'contract'.")

    def clear(self) -> None:
        """Clear registry for session cleanup."""
        self._sources.clear()


source_registry = LegalSourceRegistry()
