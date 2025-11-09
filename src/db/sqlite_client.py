"""SQLite client utilities.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
    v0.2.0 - 2025-11-09 - Added replace_all helper for dataset refresh workflows.
    v0.3.0 - 2025-05-09 - Added CRUD helpers for interactive technique management.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

TECHNIQUES_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS techniques (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    origin_year INTEGER,
    creator TEXT,
    category TEXT,
    core_principles TEXT
);
"""

FEEDBACK_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    workflow TEXT NOT NULL,
    message TEXT NOT NULL,
    rating INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

PREFERENCES_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY,
    technique TEXT,
    category TEXT,
    rating INTEGER,
    sentiment TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class SQLiteClient:
    """Lightweight wrapper around sqlite3 for the techniques knowledge base."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the SQLite client.

        Args:
            db_path (str | Path): Path to the SQLite database file.
        """

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Return or lazily initialize the SQLite connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(self._db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def initialize_schema(self) -> None:
        """Ensure the base techniques table exists."""
        with self.connection as conn:
            conn.execute(TECHNIQUES_TABLE_SCHEMA)
            conn.execute(FEEDBACK_TABLE_SCHEMA)
            conn.execute(PREFERENCES_TABLE_SCHEMA)

    def insert_technique(
        self,
        name: str,
        description: str,
        origin_year: Optional[int],
        creator: Optional[str],
        category: Optional[str],
        core_principles: Optional[str],
    ) -> int:
        """Insert a new technique record.

        Args:
            name (str): Technique name.
            description (str): Detailed description of the technique.
            origin_year (int | None): Year the technique originated.
            creator (str | None): Creator attribution.
            category (str | None): Category label.
            core_principles (str | None): Core principles overview.

        Returns:
            int: Identifier of the inserted row.
        """

        query = """
        INSERT INTO techniques (name, description, origin_year, creator, category, core_principles)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        with self.connection as conn:
            cursor = conn.execute(
                query,
                (name, description, origin_year, creator, category, core_principles),
            )
            return cursor.lastrowid

    def bulk_insert(self, techniques: Iterable[dict]) -> None:
        """Insert multiple techniques in a single transaction.

        Args:
            techniques (Iterable[dict]): Iterable of technique dictionaries.
        """

        rows = [
            (
                item.get("name"),
                item.get("description"),
                item.get("origin_year"),
                item.get("creator"),
                item.get("category"),
                item.get("core_principles"),
            )
            for item in techniques
        ]
        with self.connection as conn:
            conn.executemany(
                """
                INSERT INTO techniques (name, description, origin_year, creator, category, core_principles)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def replace_all(self, techniques: Iterable[dict]) -> None:
        """Replace all technique rows with the supplied collection."""

        rows = [
            (
                item.get("name"),
                item.get("description"),
                item.get("origin_year"),
                item.get("creator"),
                item.get("category"),
                item.get("core_principles"),
            )
            for item in techniques
        ]

        with self.connection as conn:
            conn.execute("DELETE FROM techniques")
            if rows:
                conn.executemany(
                    """
                    INSERT INTO techniques (name, description, origin_year, creator, category, core_principles)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def fetch_all(self) -> list[sqlite3.Row]:
        """Fetch all technique records from the database.

        Returns:
            list[sqlite3.Row]: Result set containing technique rows.
        """

        with self.connection as conn:
            cursor = conn.execute("SELECT * FROM techniques")
            return cursor.fetchall()

    def fetch_by_name(self, name: str) -> sqlite3.Row | None:
        """Fetch a single technique row by its name.

        Args:
            name (str): Technique name to locate.

        Returns:
            sqlite3.Row | None: Matching row or ``None`` when absent.
        """

        with self.connection as conn:
            cursor = conn.execute(
                "SELECT * FROM techniques WHERE lower(name) = lower(?) LIMIT 1",
                (name,),
            )
            return cursor.fetchone()

    def update_technique(self, name: str, updates: dict[str, Any]) -> int:
        """Update an existing technique with the supplied fields.

        Args:
            name (str): Existing technique name to update.
            updates (dict[str, Any]): Column/value pairs to persist.

        Returns:
            int: Number of rows affected by the update.
        """

        if not updates:
            return 0

        allowed = {
            "name",
            "description",
            "origin_year",
            "creator",
            "category",
            "core_principles",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for column, value in updates.items():
            if column not in allowed:
                continue
            assignments.append(f"{column} = ?")
            values.append(value)

        if not assignments:
            return 0

        values.append(name)
        with self.connection as conn:
            cursor = conn.execute(
                f"UPDATE techniques SET {', '.join(assignments)} WHERE lower(name) = lower(?)",
                values,
            )
            return cursor.rowcount

    def delete_technique(self, name: str) -> int:
        """Delete a technique by name.

        Args:
            name (str): Technique name to remove.

        Returns:
            int: Number of deleted rows.
        """

        with self.connection as conn:
            cursor = conn.execute(
                "DELETE FROM techniques WHERE lower(name) = lower(?)",
                (name,),
            )
            return cursor.rowcount

    def close(self) -> None:
        """Close and discard the active SQLite connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
