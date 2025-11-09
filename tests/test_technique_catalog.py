from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.db.sqlite_client import SQLiteClient
from src.services.technique_catalog import TechniqueCatalogService


class StubEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


@pytest.fixture()
def catalog(tmp_path: Path) -> TechniqueCatalogService:
    db_path = tmp_path / "techniques.db"
    sqlite_client = SQLiteClient(db_path)
    sqlite_client.initialize_schema()
    dataset_path = tmp_path / "techniques.json"
    service = TechniqueCatalogService(
        sqlite_client=sqlite_client,
        embedder=StubEmbedder(),
        dataset_path=dataset_path,
        chroma_client=None,
    )
    yield service
    sqlite_client.close()


def _read_dataset(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def test_add_persists_to_db_and_dataset(catalog: TechniqueCatalogService) -> None:
    entry = catalog.add(
        {
            "name": "Test Technique",
            "description": "Helpful description.",
            "origin_year": 1999,
            "creator": "Jane Doe",
            "category": "Decision",
            "core_principles": "Principle A",
        }
    )

    assert entry["name"] == "Test Technique"
    rows = catalog.sqlite_client.fetch_all()
    assert len(rows) == 1
    dataset_entries = _read_dataset(catalog.dataset_path)
    assert dataset_entries[0]["name"] == "Test Technique"


def test_add_rejects_duplicates(catalog: TechniqueCatalogService) -> None:
    catalog.add({"name": "Duplicate", "description": "Initial."})
    with pytest.raises(ValueError):
        catalog.add({"name": "Duplicate", "description": "Again."})


def test_update_modifies_dataset_and_db(catalog: TechniqueCatalogService) -> None:
    catalog.add({"name": "Original", "description": "Desc"})
    updated = catalog.update(
        "Original",
        {"name": "Renamed", "description": "Updated", "category": "Focus"},
    )

    assert updated["name"] == "Renamed"
    assert updated["category"] == "Focus"

    assert catalog.sqlite_client.fetch_by_name("Renamed") is not None
    dataset_entries = _read_dataset(catalog.dataset_path)
    assert dataset_entries[0]["name"] == "Renamed"
    assert dataset_entries[0]["description"] == "Updated"


def test_remove_deletes_from_all_sources(catalog: TechniqueCatalogService) -> None:
    catalog.add({"name": "Temporary", "description": "To delete"})
    catalog.remove("Temporary")

    assert catalog.sqlite_client.fetch_all() == []
    assert _read_dataset(catalog.dataset_path) == []
