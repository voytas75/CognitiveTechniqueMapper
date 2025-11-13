from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pytest

from src.db.chroma_client import ChromaClient, EmbeddingRecord


class StubCollection:
    def __init__(self, *, fail_once: bool) -> None:
        self.fail_once = fail_once
        self.upsert_calls: list[dict[str, Any]] = []

    def upsert(
        self,
        *,
        ids: Iterable[str],
        embeddings: Iterable[list[float]],
        metadatas: Iterable[dict[str, Any]],
        documents: Iterable[str],
    ) -> None:
        self.upsert_calls.append(
            {
                "ids": list(ids),
                "embeddings": [list(vector) for vector in embeddings],
                "metadatas": list(metadatas),
                "documents": list(documents),
            }
        )
        if self.fail_once:
            self.fail_once = False
            raise Exception("Collection expecting embedding with dimension of 12, got 3072")

    def delete(self, ids: list[str]) -> None:  # pragma: no cover - API compatibility
        pass


class StubClient:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.created = 0
        self._collections = [
            StubCollection(fail_once=True),
            StubCollection(fail_once=False),
        ]

    def get_or_create_collection(self, name: str) -> StubCollection:
        index = min(self.created, len(self._collections) - 1)
        collection = self._collections[index]
        self.created += 1
        return collection

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)


@pytest.fixture()
def chroma_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChromaClient:
    client = ChromaClient(persist_directory=tmp_path, collection_name="test")
    stub_client = StubClient()
    client._client = stub_client  # type: ignore[attr-defined]
    client._collection = None  # type: ignore[attr-defined]
    return client


def test_upsert_resets_collection_on_dimension_mismatch(
    chroma_client: ChromaClient,
) -> None:
    records = [
        EmbeddingRecord(
            identifier="example",
            embedding=[0.1, 0.2, 0.3],
            metadata={"name": "example"},
            document="Example document",
        )
    ]

    chroma_client.upsert_embeddings(records)

    stub_client: StubClient = chroma_client._client  # type: ignore[attr-defined]
    assert stub_client.deleted == ["test"]
    assert stub_client._collections[1].upsert_calls  # second collection received data
