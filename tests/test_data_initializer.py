from pathlib import Path

from src.db.sqlite_client import SQLiteClient
from src.services.config_service import ConfigService
from src.services.embedding_gateway import EmbeddingGateway
from src.services.data_initializer import TechniqueDataInitializer


def _write_config(dir_path: Path) -> None:
    (dir_path / "settings.yaml").write_text("app: {name: test}\n", encoding="utf-8")
    (dir_path / "database.yaml").write_text("database: {sqlite_path: ':memory:'}\n", encoding="utf-8")
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
