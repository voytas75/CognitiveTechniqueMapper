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

    def embed_batch(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


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


def test_export_writes_dataset(catalog: TechniqueCatalogService, tmp_path: Path) -> None:
    catalog.add({"name": "Exported", "description": "Catalog entry"})
    output_path = tmp_path / "out.json"

    path, count = catalog.export_to_file(output_path)

    assert path == output_path
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert count == 1
    assert exported[0]["name"] == "Exported"


def test_import_replace_overwrites_catalog(catalog: TechniqueCatalogService, tmp_path: Path) -> None:
    catalog.add({"name": "Old", "description": "To replace"})
    import_file = tmp_path / "replace.json"
    import_file.write_text(
        json.dumps(
            [
                {
                    "name": "New Technique",
                    "description": "Imported description",
                    "category": "Strategy",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = catalog.import_from_file(import_file, mode="replace", rebuild_embeddings=False)

    assert summary["total"] == 1
    names = [item["name"] for item in catalog.list()]
    assert names == ["New Technique"]


def test_import_append_merges_catalog(catalog: TechniqueCatalogService, tmp_path: Path) -> None:
    catalog.add({"name": "Existing", "description": "Original"})
    import_file = tmp_path / "append.json"
    import_file.write_text(
        json.dumps(
            [
                {
                    "name": "Existing",
                    "description": "Updated",
                    "category": "Updated Category",
                },
                {
                    "name": "Additional",
                    "description": "Another entry",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = catalog.import_from_file(import_file, mode="append", rebuild_embeddings=False)

    assert summary["total"] == 2
    records = {item["name"]: item for item in catalog.list()}
    assert records["Existing"]["description"] == "Updated"
    assert records["Existing"]["category"] == "Updated Category"
    assert records["Additional"]["description"] == "Another entry"
