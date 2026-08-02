"""Persistence helpers for user preference signals."""

from __future__ import annotations

from typing import List, Optional, TypedDict, cast

from .sqlite_client import SQLiteClient


class PreferenceRecord(TypedDict):
    """Persisted preference signal read from SQLite."""

    technique: str | None
    category: str | None
    rating: int | None
    sentiment: str
    notes: str | None
    created_at: str


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
            row_id = cursor.lastrowid
            if row_id is None:
                raise RuntimeError(
                    "SQLite did not return an identifier for the preference entry"
                )
            return row_id

    def fetch_recent(self, limit: int = 20) -> List[PreferenceRecord]:
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
            return [cast(PreferenceRecord, dict(row)) for row in cursor.fetchall()]

    def fetch_all(self) -> List[PreferenceRecord]:
        """Return all stored preferences."""

        with self._sqlite.connection as conn:
            cursor = conn.execute("""
                SELECT technique, category, rating, sentiment, notes, created_at
                FROM preferences
                ORDER BY datetime(created_at) DESC
                """)
            return [cast(PreferenceRecord, dict(row)) for row in cursor.fetchall()]

    def delete_all(self) -> None:
        """Remove all stored preference records."""

        with self._sqlite.connection as conn:
            conn.execute("DELETE FROM preferences")
