"""
Local session store using SQLite or filesystem.
Stores user-approved session data strictly on the local device.
Zero remote synchronization, cloud replication, or telemetry.
"""

import sqlite3
from pathlib import Path
from backend.app.config import settings


class SessionStore:
    """Manages local session metadata and ephemeral document records."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (settings.STORAGE_DIR / "sessions.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def register_session(self, session_id: str, document_id: str, filename: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, document_id, filename) VALUES (?, ?, ?)",
                (session_id, document_id, filename),
            )
            conn.commit()

    def delete_session(self, session_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cur.rowcount > 0
