"""Persistence helpers for user preference signals."""

from __future__ import annotations

from typing import List, Optional

from .sqlite_client import SQLiteClient


class PreferenceRepository:
    """Provides storage and aggregation helpers for preference entries."""

    def __init__(self, sqlite_client: SQLiteClient) -> None:
        self._sqlite = sqlite_client

    def insert(
        self,
        *,
        technique: Optional[str],
        category: Optional[str],
        rating: Optional[int],
        sentiment: str,
        notes: Optional[str],
        created_at: str,
    ) -> int:
        """Persist a preference entry."""

        with self._sqlite.connection as conn:
            cursor = conn.execute(
                """
                INSERT INTO preferences (technique, category, rating, sentiment, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (technique, category, rating, sentiment, notes, created_at),
            )
            return cursor.lastrowid

    def fetch_recent(self, limit: int = 20) -> List[dict]:
        """Return the most recent preference entries."""

        with self._sqlite.connection as conn:
            cursor = conn.execute(
                """
                SELECT technique, category, rating, sentiment, notes, created_at
                FROM preferences
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def fetch_all(self) -> List[dict]:
        """Return all stored preferences."""

        with self._sqlite.connection as conn:
            cursor = conn.execute(
                """
                SELECT technique, category, rating, sentiment, notes, created_at
                FROM preferences
                ORDER BY datetime(created_at) DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_all(self) -> None:
        """Remove all stored preference records."""

        with self._sqlite.connection as conn:
            conn.execute("DELETE FROM preferences")
