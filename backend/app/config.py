"""
Configuration settings for local-first legal triage system.
All network bindings are strictly restricted to 127.0.0.1 (loopback).
Cloud calls, remote telemetries, and external model APIs are strictly prohibited.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # Network and Host Security (Strict Loopback Only)
    HOST: str = Field(default_factory=lambda: os.getenv("APP_HOST", "127.0.0.1"))
    PORT: int = Field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    ALLOWED_ORIGINS: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    STORAGE_DIR: Path = BASE_DIR / "backend" / "app" / "storage" / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"

    # Document Ingestion limits (None means unlimited)
    MAX_FILE_SIZE_BYTES: int | None = None
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".json", ".jsonl"}

    # ─── Phase 1: Chunking Configuration ──────────────────────────────────────
    # Suggested: 1600 chars per chunk, 240 chars overlap (evaluate later)
    CHUNK_SIZE_CHARS: int = 1600
    CHUNK_OVERLAP_CHARS: int = 240
    MIN_CHUNK_CHARS: int = 50

    # ─── Phase 2: Embedding Model (local, CPU-safe) ────────────────────────────
    # BAAI/bge-small-en-v1.5: 384-dim, runs fully offline after first download
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # ─── Phase 2: Qdrant Local Storage ────────────────────────────────────────
    QDRANT_PATH: Path = BASE_DIR / "backend" / "app" / "storage" / "qdrant"
    QDRANT_COLLECTION_NAME: str = "legal_chunks"

    # ─── Phase 2: Hybrid Retrieval ────────────────────────────────────────────
    BM25_TOP_K: int = 5
    VECTOR_TOP_K: int = 5
    HYBRID_TOP_K: int = 5
    BM25_MIN_SCORE_THRESHOLD: float = 0.5
    # Reciprocal Rank Fusion constant (k=60 is standard)
    RRF_K: int = 60
    MAX_CONTEXT_TOKENS: int = 2048

    # ─── Phase 3: Caching ─────────────────────────────────────────────────────
    CACHE_DIR: Path = BASE_DIR / "backend" / "app" / "storage" / "cache"
    EMBEDDING_CACHE_DIR: Path = CACHE_DIR / "embeddings"
    RETRIEVAL_CACHE_DIR: Path = CACHE_DIR / "retrieval"
    CACHE_MAX_ENTRIES: int = 1000

    # ─── Local Gemma Model Runtime ────────────────────────────────────────────
    # Target: Gemma 3n E2B / E4B or edge quantized GGUF
    GEMMA_MODEL_NAME: str = "gemma-3-e2b-it"
    GEMMA_MODEL_PATH: Path = MODELS_DIR / "gemma-3-e2b-it-q4_k_m.gguf"
    GEMMA_CONTEXT_WINDOW: int = 4096
    GEMMA_MAX_OUTPUT_TOKENS: int = 1024
    GEMMA_TEMPERATURE: float = 0.0  # Greedy decoding for deterministic evidence extraction
    DEVICE: str = "cpu"  # Or "cuda" if locally available

    # Privacy and Logging
    # Logs MUST NEVER record document text, user questions, names, or model answers
    SANITIZE_LOGS: bool = True
    ENABLE_TELEMETRY: bool = False  # Hard enforced false


settings = Settings()

