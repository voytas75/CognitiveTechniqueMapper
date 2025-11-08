"""SQLite client utilities.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

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

    def fetch_all(self) -> list[sqlite3.Row]:
        """Fetch all technique records from the database.

        Returns:
            list[sqlite3.Row]: Result set containing technique rows.
        """

        with self.connection as conn:
            cursor = conn.execute("SELECT * FROM techniques")
            return cursor.fetchall()

    def close(self) -> None:
        """Close and discard the active SQLite connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
