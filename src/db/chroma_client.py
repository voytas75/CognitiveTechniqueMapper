"""ChromaDB client utilities.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import chromadb  # type: ignore
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
except ImportError as exc:  # pragma: no cover - guidance for missing dependency
    raise RuntimeError(
        "chromadb is required for ChromaClient. Install via `pip install chromadb`."
    ) from exc


@dataclass(slots=True)
class EmbeddingRecord:
    """Embedding record container for ChromaDB operations."""

    identifier: str
    embedding: Sequence[float]
    metadata: dict[str, str] | None = None
    document: str | None = None


class ChromaClient:
    """Wrapper around ChromaDB to manage technique embeddings."""

    def __init__(self, persist_directory: str | Path, collection_name: str) -> None:
        """Initialize the Chroma client instance.

        Args:
            persist_directory (str | Path): Directory for persistent storage.
            collection_name (str): Name of the collection to operate on.
        """

        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None

    @property
    def client(self) -> ClientAPI:
        """Return or instantiate the Chroma persistent client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self._persist_directory))
        return self._client

    @property
    def collection(self) -> Collection:
        """Return or create the target Chroma collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                self._collection_name
            )
        return self._collection

    def upsert_embeddings(self, embeddings: Iterable[EmbeddingRecord]) -> None:
        """Upsert embedding records into the collection.

        Args:
            embeddings (Iterable[EmbeddingRecord]): Records to insert or update.
        """

        ids, vectors, metadatas, documents = [], [], [], []
        for record in embeddings:
            ids.append(record.identifier)
            vectors.append(list(record.embedding))
            metadatas.append(record.metadata or {})
            documents.append(record.document)

        self.collection.upsert(
            ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents
        )

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        """Query similar embeddings from the collection.

        Args:
            query_embeddings (Sequence[Sequence[float]]): Query vectors.
            n_results (int): Maximum number of results to return.
            where (dict | None): Optional filter criteria.

        Returns:
            dict: Query response including IDs, documents, and metadata.
        """

        return self.collection.query(
            query_embeddings=list(query_embeddings),
            n_results=n_results,
            where=where,
        )

    def delete(self, ids: Sequence[str]) -> None:
        """Delete embeddings associated with the provided identifiers.

        Args:
            ids (Sequence[str]): Identifiers to remove from the collection.
        """

        self.collection.delete(ids=list(ids))
