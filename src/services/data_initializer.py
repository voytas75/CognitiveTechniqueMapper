"""Technique data initialization routines.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
    v0.2.0 - 2025-11-09 - Added dataset refresh capability with embedding rebuild toggle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Protocol, Sequence, cast

from ..db.sqlite_client import SQLiteClient, TechniqueRecord
from .embedding_gateway import EmbeddingGateway
from .technique_utils import compose_embedding_text

if TYPE_CHECKING:
    from ..db.chroma_client import EmbeddingRecord


class _ChromaEmbeddingStore(Protocol):
    """Operations required from the optional Chroma integration."""

    def upsert_embeddings(self, records: Iterable["EmbeddingRecord"], /) -> None:
        """Insert or update embedding records."""
        ...

    def list_ids(self) -> list[str]:
        """Return stored embedding identifiers."""
        ...

    def delete(self, ids: Sequence[str]) -> None:
        """Delete embedding records by identifier."""
        ...


DEFAULT_DATASET_PATH = Path("data/techniques.json")


class CatalogSynchronizationError(RuntimeError):
    """Raised when optional Chroma cannot stay synchronized with SQLite."""


class TechniqueDataInitializer:
    """Loads technique metadata into SQLite and synchronizes embeddings with Chroma."""

    def __init__(
        self,
        sqlite_client: SQLiteClient,
        embedder: EmbeddingGateway,
        chroma_client: _ChromaEmbeddingStore | None = None,
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

        if seeded and self._chroma:
            records = self._build_embedding_records(dataset)
            if records:
                self._chroma.upsert_embeddings(records)

    def _seed_sqlite(self, dataset: List[TechniqueRecord]) -> bool:
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
        """Reload the dataset and synchronize optional embeddings explicitly."""

        dataset = self._load_dataset()
        existing_ids: list[str] = []
        records: List["EmbeddingRecord"] = []
        if self._chroma and rebuild_embeddings:
            try:
                existing_ids = self._chroma.list_ids()
                records = self._build_embedding_records(dataset)
            except Exception as exc:
                raise CatalogSynchronizationError(
                    "Chroma synchronization preflight failed; SQLite was left unchanged."
                ) from exc

        self._sqlite.replace_all(dataset)

        if self._chroma and rebuild_embeddings:
            try:
                if existing_ids:
                    self._chroma.delete(existing_ids)
                if records:
                    self._chroma.upsert_embeddings(records)
            except Exception as exc:
                raise CatalogSynchronizationError(
                    "Chroma synchronization failed after the SQLite refresh."
                ) from exc

    def _load_dataset(self) -> List[TechniqueRecord]:
        """Load the technique dataset from disk.

        Returns:
            list[dict]: Technique entries parsed from the dataset file.

        Raises:
            ValueError: If the dataset file does not contain a list.
        """

        if not self._dataset_path.exists():
            return []
        with self._dataset_path.open("r", encoding="utf-8") as handle:
            raw_data: object = json.load(handle)
        if not isinstance(raw_data, list):
            raise ValueError("Technique dataset is not a list of objects")

        dataset: list[TechniqueRecord] = []
        for raw_entry in cast(list[object], raw_data):
            record = self._parse_technique_record(raw_entry)
            if record is None:
                raise ValueError(
                    "Technique dataset must contain valid technique objects"
                )
            dataset.append(record)
        return dataset

    @staticmethod
    def _parse_technique_record(value: object) -> TechniqueRecord | None:
        """Validate one JSON dataset entry against the persisted technique schema."""

        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            return None
        raw = cast(dict[str, object], value)
        name = raw.get("name")
        description = raw.get("description")
        origin_year = raw.get("origin_year")
        creator = raw.get("creator")
        category = raw.get("category")
        core_principles = raw.get("core_principles")
        if (
            not isinstance(name, str)
            or not isinstance(description, str)
            or (
                origin_year is not None
                and (not isinstance(origin_year, int) or isinstance(origin_year, bool))
            )
            or (creator is not None and not isinstance(creator, str))
            or (category is not None and not isinstance(category, str))
            or (core_principles is not None and not isinstance(core_principles, str))
        ):
            return None
        return {
            "name": name,
            "description": description,
            "origin_year": origin_year,
            "creator": creator,
            "category": category,
            "core_principles": core_principles,
        }

    def _build_embedding_records(
        self, dataset: Iterable[TechniqueRecord]
    ) -> List["EmbeddingRecord"]:
        """Build embedding records for Chroma synchronization.

        Args:
            dataset (Iterable[dict]): Technique entries to embed.

        Returns:
            list[EmbeddingRecord]: Embedding records ready for upsert.
        """

        records: List["EmbeddingRecord"] = []
        texts: List[str] = []
        metadata_list: List[dict[str, str]] = []
        identifiers: List[str] = []
        documents: List[str] = []

        for item in dataset:
            identifier = item["name"]
            if not identifier:
                continue
            text = self._compose_embedding_text(item)
            texts.append(text)
            identifiers.append(identifier)
            metadata_list.append(
                {
                    "name": item["name"],
                    "category": item["category"] or "",
                    "creator": item["creator"] or "",
                    "origin_year": (
                        str(item["origin_year"])
                        if item["origin_year"] is not None
                        else ""
                    ),
                }
            )
            documents.append(item["description"])

        if not texts:
            return []

        embeddings = self._embedder.embed_batch(texts)
        for identifier, embedding_vector, metadata, document in zip(
            identifiers, embeddings, metadata_list, documents
        ):
            records.append(
                self._create_embedding_record(
                    identifier=identifier,
                    embedding=embedding_vector,
                    metadata=metadata,
                    document=document,
                )
            )
        return records

    @staticmethod
    def _create_embedding_record(
        *,
        identifier: str,
        embedding: Sequence[float],
        metadata: dict[str, str],
        document: str,
    ) -> "EmbeddingRecord":
        """Create a Chroma record only when embedding sync is requested."""

        from ..db.chroma_client import EmbeddingRecord

        return EmbeddingRecord(
            identifier=identifier,
            embedding=embedding,
            metadata=metadata,
            document=document,
        )

    def _compose_embedding_text(self, item: TechniqueRecord) -> str:
        """Compose embedding text for a validated dataset entry.

        Args:
            item (TechniqueRecord): Technique metadata.

        Returns:
            str: Structured text used for embedding generation.
        """

        return compose_embedding_text(
            {
                "description": item["description"],
                "core_principles": item["core_principles"] or "",
                "category": item["category"] or "",
            }
        )
