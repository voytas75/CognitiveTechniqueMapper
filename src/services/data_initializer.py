"""Technique data initialization routines.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
    v0.2.0 - 2025-11-09 - Added dataset refresh capability with embedding rebuild toggle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List

from ..db.sqlite_client import SQLiteClient
from .embedding_gateway import EmbeddingGateway
from .technique_utils import compose_embedding_text

try:
    from ..db.chroma_client import ChromaClient, EmbeddingRecord
except RuntimeError:
    ChromaClient = None  # type: ignore
    EmbeddingRecord = None  # type: ignore


logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path("data/techniques.json")


class TechniqueDataInitializer:
    """Loads technique metadata into SQLite and synchronizes embeddings with Chroma."""

    def __init__(
        self,
        sqlite_client: SQLiteClient,
        embedder: EmbeddingGateway,
        chroma_client: ChromaClient | None = None,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
    ) -> None:
        """Initialize the initializer with its dependencies.

        Args:
            sqlite_client (SQLiteClient): Database client for persistent storage.
            embedder (EmbeddingGateway): Embedding generator for records.
            chroma_client (ChromaClient | None): Optional Chroma client for vector sync.
            dataset_path (Path | str): Source dataset file to load.
        """

        self._sqlite = sqlite_client
        self._embedder = embedder
        self._chroma = chroma_client
        self._dataset_path = Path(dataset_path)

    def initialize(self) -> None:
        """Populate the SQLite database and optionally synchronize embeddings."""
        dataset = self._load_dataset()
        if not dataset:
            return

        seeded = self._seed_sqlite(dataset)

        if seeded and self._chroma and EmbeddingRecord:
            records = self._build_embedding_records(dataset)
            if records:
                self._chroma.upsert_embeddings(records)

    def _seed_sqlite(self, dataset: List[dict]) -> bool:
        """Seed SQLite with the dataset if the techniques table is empty."""

        with self._sqlite.connection as conn:
            cursor = conn.execute("SELECT 1 FROM techniques LIMIT 1")
            has_existing = cursor.fetchone() is not None

        if has_existing:
            return False

        if dataset:
            self._sqlite.bulk_insert(dataset)
        return True

    def refresh(self, *, rebuild_embeddings: bool = True) -> None:
        """Reload the dataset and rebuild embeddings if requested."""

        dataset = self._load_dataset()
        self._sqlite.replace_all(dataset)

        if self._chroma and EmbeddingRecord and rebuild_embeddings:
            try:
                existing_ids = self._chroma.list_ids()
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Failed to enumerate existing embeddings: %s", exc)
                existing_ids = []

            if existing_ids:
                try:
                    self._chroma.delete(existing_ids)
                except Exception as exc:  # pragma: no cover - Chroma optional
                    logger.warning("Failed to delete existing embeddings: %s", exc)

            records = self._build_embedding_records(dataset)
            if not records:
                return
            try:
                self._chroma.upsert_embeddings(records)
            except Exception as exc:  # pragma: no cover - Chroma optional
                logger.warning("Failed to upsert embeddings: %s", exc)

    def _load_dataset(self) -> List[dict]:
        """Load the technique dataset from disk.

        Returns:
            list[dict]: Technique entries parsed from the dataset file.

        Raises:
            ValueError: If the dataset file does not contain a list.
        """

        if not self._dataset_path.exists():
            return []
        with self._dataset_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError("Technique dataset is not a list of objects")
            return data

    def _build_embedding_records(
        self, dataset: Iterable[dict]
    ) -> List[EmbeddingRecord]:
        """Build embedding records for Chroma synchronization.

        Args:
            dataset (Iterable[dict]): Technique entries to embed.

        Returns:
            list[EmbeddingRecord]: Embedding records ready for upsert.
        """

        records: List[EmbeddingRecord] = []
        texts: List[str] = []
        metadata_list: List[dict] = []
        identifiers: List[str] = []
        documents: List[str] = []

        for item in dataset:
            identifier = item.get("name")
            if not identifier:
                continue
            text = self._compose_embedding_text(item)
            texts.append(text)
            identifiers.append(identifier)
            metadata_list.append(
                {
                    "name": item.get("name", ""),
                    "category": item.get("category", ""),
                    "creator": item.get("creator", ""),
                    "origin_year": str(item.get("origin_year", "")),
                }
            )
            documents.append(item.get("description", ""))

        if not texts:
            return []

        embeddings = self._embedder.embed_batch(texts)
        for identifier, embedding_vector, metadata, document in zip(
            identifiers, embeddings, metadata_list, documents
        ):
            records.append(
                EmbeddingRecord(
                    identifier=identifier,
                    embedding=embedding_vector,
                    metadata=metadata,
                    document=document,
                )
            )
        return records

    def _compose_embedding_text(self, item: dict) -> str:
        """Compose embedding text for a dataset entry.

        Args:
            item (dict): Technique metadata.

        Returns:
            str: Structured text used for embedding generation.
        """

        return compose_embedding_text(item)
