"""
Deterministic workflow orchestrator.
Coordinates: Ingestion -> Segmentation -> Retrieval -> Generation -> Safety Validation.
"""

from pathlib import Path
from backend.app.agent.analyzer import ContractAnalyzer
from backend.app.agent.gemma_runtime import GemmaRuntime
from backend.app.document.chunker import chunk_document
from backend.app.document.schemas import ExtractedDocument, QueryResponse
from backend.app.ingestion.pdf_parser import parse_pdf
from backend.app.ingestion.txt_parser import parse_txt
from backend.app.ingestion.validator import validate_file
from backend.app.retrieval.index import BM25DocumentIndex
from backend.app.retrieval.search import search_chunks


class DocumentTriageWorkflow:
    """Manages the full end-to-end triage pipeline locally."""

    def __init__(self):
        self.runtime = GemmaRuntime()
        self.analyzer = ContractAnalyzer(self.runtime)
        self.active_document: ExtractedDocument | None = None
        self.active_index: BM25DocumentIndex | None = None

    def ingest_document(self, file_path: Path, document_id: str) -> ExtractedDocument:
        with open(file_path, "rb") as f:
            content = f.read()

        file_type = validate_file(file_path, file_path.name, content)
        if file_type == "pdf":
            doc = parse_pdf(file_path, document_id)
        else:
            doc = parse_txt(file_path, document_id)

        chunks = chunk_document(doc)
        self.active_document = doc
        self.active_index = BM25DocumentIndex(chunks)
        return doc

    def query(self, question: str) -> QueryResponse:
        if not self.active_index:
            return QueryResponse(
                answer="No active document loaded. Please upload a rental agreement first.",
                status="insufficient_evidence",
                findings=[],
                evidence_used=[],
            )

        retrieval = search_chunks(self.active_index, question)
        if not retrieval.is_sufficient:
            return QueryResponse(
                answer="The contract does not provide enough information to answer this question.",
                status="insufficient_evidence",
                findings=[],
                evidence_used=[],
            )

        return self.analyzer.analyze_question(question, retrieval.passages)
