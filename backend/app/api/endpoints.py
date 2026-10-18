"""
API Endpoints for local legal triage:
- POST /api/upload
- GET /api/overview/{session_id}
- GET /api/clauses/{session_id}
- POST /api/ask
- DELETE /api/session/{session_id}
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.app.document.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api")


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload rental contract (PDF or TXT) for local in-memory ingestion."""
    return {"message": "Upload endpoint skeleton", "filename": file.filename}


@router.get("/overview/{session_id}")
async def get_overview(session_id: str):
    """Retrieve document metadata and clause summary."""
    return {"session_id": session_id, "status": "active"}


@router.get("/clauses/{session_id}")
async def get_clauses(session_id: str):
    """Explore detected clauses and structured sections."""
    return {"session_id": session_id, "clauses": []}


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask an evidence-grounded question regarding the contract."""
    return QueryResponse(
        answer="Workflow execution pending.",
        status="insufficient_evidence",
        findings=[],
        evidence_used=[],
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Irrecoverably purge local session data and indices."""
    return {"session_id": session_id, "deleted": True}
