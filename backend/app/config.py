"""Configuration module for Private Legal Triage.

Enforces local-first constraints, network isolation (127.0.0.1 only),
and privacy boundaries.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with strict privacy and local-first boundaries."""

    model_config = SettingsConfigDict(
        env_prefix="PLT_",
        env_file=".env",
        extra="ignore"
    )

    # Server binding - MUST ONLY bind to localhost
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False

    # Privacy & Local-Only Guarantees
    LOCAL_ONLY: bool = True
    ALLOW_REMOTE_NETWORK: bool = False
    LOG_DOCUMENT_CONTENT: bool = False
    LOG_USER_QUERIES: bool = False
    LOG_MODEL_ANSWERS: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    SESSIONS_DIR: Path = BASE_DIR / "sessions"

    # File Upload Restrictions (MVP: Rental Agreements PDF and TXT)
    MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")
    ALLOWED_MIME_TYPES: tuple[str, ...] = (
        "application/pdf",
        "text/plain",
    )

    # Retrieval & Evidence Budget
    BM25_TOP_K: int = 5
    BM25_SCORE_THRESHOLD: float = 1.0
    MAX_CONTEXT_CHAR_BUDGET: int = 4000  # Token-conserving context budget for Gemma

    # Model Specification
    DEFAULT_MODEL_NAME: str = "gemma-2-2b-it-Q4_K_M"
    MODEL_CONTEXT_WINDOW: int = 4096


settings = Settings()
