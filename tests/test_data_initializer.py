import json
from pathlib import Path
from typing import Any

import pytest

from src.db.sqlite_client import SQLiteClient
from src.services.config_service import ConfigService
from src.services.embedding_gateway import EmbeddingGateway
from src.services.data_initializer import TechniqueDataInitializer


def _write_config(dir_path: Path) -> None:
    (dir_path / "settings.yaml").write_text("app: {name: test}\n", encoding="utf-8")
    (dir_path / "database.yaml").write_text(
        "database: {sqlite_path: ':memory:'}\n", encoding="utf-8"
    )
    (dir_path / "models.yaml").write_text(
        """
workflows: {detect_technique: {model: dummy}}
defaults: {provider: openai}
embeddings: {model: dummy-embedding, provider: openai}
""".strip(),
        encoding="utf-8",
    )
    (dir_path / "providers.yaml").write_text(
        "providers: {openai: {api_base: 'http://localhost', api_key_env: 'OPENAI_API_KEY'}}\n",
        encoding="utf-8",
    )


def test_data_initializer_populates_sqlite(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_config(config_dir)

    dataset = [
        {
            "name": "Test Technique",
            "description": "Example description.",
            "origin_year": 2024,
            "creator": "Tester",
            "category": "Demo",
            "core_principles": "Testing",
        }
    ]
    dataset_path = tmp_path / "techniques.json"
    import json

    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    monkeypatch.setenv("CTM_CONFIG_PATH", str(config_dir))
    config_service = ConfigService(config_path=config_dir)
    embedder = EmbeddingGateway(config_service=config_service, use_fallback=True)

    db_path = tmp_path / "techniques.db"
    sqlite_client = SQLiteClient(db_path)
    sqlite_client.initialize_schema()

    initializer = TechniqueDataInitializer(
        sqlite_client=sqlite_client,
        embedder=embedder,
        chroma_client=None,
        dataset_path=dataset_path,
    )
    initializer.initialize()

    rows = sqlite_client.fetch_all()
    assert len(rows) == 1
    assert rows[0]["name"] == "Test Technique"


def test_refresh_replaces_dataset(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_config(config_dir)

    initial_dataset = [
        {
            "name": "Initial Technique",
            "description": "Original description.",
            "origin_year": 2020,
            "creator": "Initial",
            "category": "Demo",
            "core_principles": "Initial principle",
        }
    ]
    updated_dataset = [
        {
            "name": "Updated Technique",
            "description": "Updated description.",
            "origin_year": 2025,
            "creator": "Updater",
            "category": "Demo",
            "core_principles": "Updated principle",
        }
    ]

    dataset_path = tmp_path / "techniques.json"

    import json

    dataset_path.write_text(json.dumps(initial_dataset), encoding="utf-8")

    monkeypatch.setenv("CTM_CONFIG_PATH", str(config_dir))
    config_service = ConfigService(config_path=config_dir)
    embedder = EmbeddingGateway(config_service=config_service, use_fallback=True)

    db_path = tmp_path / "techniques.db"
    sqlite_client = SQLiteClient(db_path)
    sqlite_client.initialize_schema()

    initializer = TechniqueDataInitializer(
        sqlite_client=sqlite_client,
        embedder=embedder,
        chroma_client=None,
        dataset_path=dataset_path,
    )
    initializer.initialize()

    dataset_path.write_text(json.dumps(updated_dataset), encoding="utf-8")
    initializer.refresh(rebuild_embeddings=False)

    rows = sqlite_client.fetch_all()
    assert len(rows) == 1
    assert rows[0]["name"] == "Updated Technique"


def test_refresh_updates_chroma_index(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_config(config_dir)

    dataset = [
        {
            "name": "Technique",
            "description": "Desc",
            "origin_year": 2024,
            "creator": "Author",
            "category": "Category",
            "core_principles": "Principle",
        }
    ]
    dataset_path = tmp_path / "techniques.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    monkeypatch.setenv("CTM_CONFIG_PATH", str(config_dir))

    class StubEmbedder(EmbeddingGateway):
        def __init__(self) -> None:
            pass

        def embed_batch(self, texts: Any) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

    class StubChroma:
        def __init__(self) -> None:
            self.upserts: list[Any] = []
            self.deleted: list[Any] = []

        def upsert_embeddings(self, records: Any) -> None:
            self.upserts.extend(records)

        def delete(self, ids: Any) -> None:
            self.deleted.extend(ids)

    sqlite_client = SQLiteClient(tmp_path / "techniques.db")
    sqlite_client.initialize_schema()

    chroma = StubChroma()

    initializer = TechniqueDataInitializer(
        sqlite_client=sqlite_client,
        embedder=StubEmbedder(),
        chroma_client=chroma,
        dataset_path=dataset_path,
    )

    initializer.initialize()
    initializer.refresh(rebuild_embeddings=True)

    assert chroma.upserts


def test_initialize_skips_reembedding_when_dataset_already_seeded(tmp_path: Path) -> None:
    dataset = [
        {
            "name": "Technique",
            "description": "Desc",
            "origin_year": 2024,
            "creator": "Author",
            "category": "Category",
            "core_principles": "Principle",
        }
    ]
    dataset_path = tmp_path / "techniques.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    class StubEmbedder:
        def embed_batch(self, texts: Any) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

    class StubChroma:
        def __init__(self) -> None:
            self.upserts: list[Any] = []

        def upsert_embeddings(self, records: Any) -> None:
            self.upserts.extend(records)

    sqlite_client = SQLiteClient(tmp_path / "techniques.db")
    sqlite_client.initialize_schema()

    chroma = StubChroma()

    initializer = TechniqueDataInitializer(
        sqlite_client=sqlite_client,
        embedder=StubEmbedder(),  # type: ignore[arg-type]
        chroma_client=chroma,  # type: ignore[arg-type]
        dataset_path=dataset_path,
    )

    initializer.initialize()
    assert len(chroma.upserts) == 1

    initializer.initialize()
    assert len(chroma.upserts) == 1


def test_load_dataset_validates_structure(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text(json.dumps({"name": "Invalid"}), encoding="utf-8")

    initializer = TechniqueDataInitializer(
        sqlite_client=SQLiteClient(tmp_path / "test.db"),
        embedder=None,  # type: ignore[arg-type]
        chroma_client=None,
        dataset_path=dataset_path,
    )

    initializer._sqlite.initialize_schema()

    with pytest.raises(ValueError):
        initializer._load_dataset()
