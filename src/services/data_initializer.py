from __future__ import annotations

import json
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
        self._sqlite = sqlite_client
        self._embedder = embedder
        self._chroma = chroma_client
        self._dataset_path = Path(dataset_path)

    def initialize(self) -> None:
        dataset = self._load_dataset()
        if not dataset:
            return

        if not self._sqlite.fetch_all():
            self._sqlite.bulk_insert(dataset)

        if self._chroma and EmbeddingRecord:
            records = self._build_embedding_records(dataset)
            if records:
                self._chroma.upsert_embeddings(records)

    def _load_dataset(self) -> List[dict]:
        if not self._dataset_path.exists():
            return []
        with self._dataset_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError("Technique dataset is not a list of objects")
            return data

    def _build_embedding_records(self, dataset: Iterable[dict]) -> List[EmbeddingRecord]:
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
        return compose_embedding_text(item)
