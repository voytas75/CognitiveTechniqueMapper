"""Persistence helpers for feedback entries.

Updates:
    v0.2.0 - 2025-11-09 - Added repository for storing feedback in SQLite.
"""

from __future__ import annotations

from typing import Any, Optional

from .sqlite_client import SQLiteClient


class FeedbackRepository:
    """Provides CRUD operations for feedback records."""

    def __init__(self, sqlite_client: SQLiteClient) -> None:
        self._sqlite = sqlite_client

    def insert(
        self, workflow: str, message: str, rating: Optional[int], *, created_at: str
    ) -> int:
        with self._sqlite.connection as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback (workflow, message, rating, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (workflow, message, rating, created_at),
            )
            return cursor.lastrowid

    def fetch_recent(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._sqlite.connection as conn:
            cursor = conn.execute(
                """
                SELECT workflow, message, rating, created_at
                FROM feedback
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_all(self) -> None:
        with self._sqlite.connection as conn:
            conn.execute("DELETE FROM feedback")
