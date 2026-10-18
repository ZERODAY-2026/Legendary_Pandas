"""FastAPI application entrypoint for Private Legal Triage.

Binds strictly to 127.0.0.1 and enforces local-only operation.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings

# Configure strict privacy-preserving logger (Never log content or PII)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("private_legal_triage")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan verifying local-first constraints at startup."""
    logger.info("Initializing Private Legal Triage System...")
    logger.info("Local-first mode active. Network binding: %s:%s", settings.HOST, settings.PORT)
    if settings.HOST != "127.0.0.1":
        raise RuntimeError("Security violation: Server MUST only bind to 127.0.0.1")
    yield
    logger.info("Shutting down Private Legal Triage System.")


app = FastAPI(
    title="Private Legal Triage",
    description="Local-first, evidence-grounded legal document triage system for rental agreements.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

# Restrictive CORS: only localhost Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint indicating local-only status."""
    return {
        "status": "healthy",
        "local_only": settings.LOCAL_ONLY,
        "runtime": "local_inference",
        "version": "0.1.0",
    }
