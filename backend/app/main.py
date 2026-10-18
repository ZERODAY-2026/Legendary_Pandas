"""
FastAPI Application Entry Point.
Strictly binds to 127.0.0.1 and enforces local-only privacy standards.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings

# Configure privacy-preserving logging
# Rule: Never log document text, user questions, names, or model answers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [private-legal-triage] %(message)s",
)
logger = logging.getLogger("legal_triage")

app = FastAPI(
    title="Private Legal Triage API",
    description="Local-first, evidence-grounded legal document triage system for rental agreements.",
    version="0.1.0",
)

from backend.app.api.endpoints import router as api_router

# Restrictive CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mode": "local_first",
        "cloud_llm_enabled": False,
        "telemetry_enabled": False,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting local triage server on %s:%s", settings.HOST, settings.PORT)
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
