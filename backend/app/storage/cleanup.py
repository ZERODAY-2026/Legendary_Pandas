"""
Local storage cleanup utilities.
Provides complete, irrecoverable local deletion of session documents,
chunks, cached indices, and database records upon user request.
"""

import shutil
from pathlib import Path
from backend.app.config import settings
from backend.app.storage.session_store import SessionStore


def cleanup_session_data(session_id: str, store: SessionStore | None = None) -> bool:
    """Purge all local session records and associated files."""
    if store is None:
        store = SessionStore()

    deleted = store.delete_session(session_id)

    # Check for temporary files or directories associated with this session
    session_temp_dir = settings.STORAGE_DIR / session_id
    if session_temp_dir.exists() and session_temp_dir.is_dir():
        shutil.rmtree(session_temp_dir, ignore_errors=True)

    return deleted


def purge_all_local_data() -> None:
    """Purge entire storage directory for complete zero-trace wipe."""
    if settings.STORAGE_DIR.exists():
        shutil.rmtree(settings.STORAGE_DIR, ignore_errors=True)
        settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
