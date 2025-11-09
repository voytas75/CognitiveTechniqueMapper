from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.services.config_service import ConfigService
from src.services.embedding_gateway import EmbeddingGateway


def _write_embedding_config(base_dir: Path) -> None:
    (base_dir / "settings.yaml").write_text("app: {name: test}\n", encoding="utf-8")
    (base_dir / "database.yaml").write_text(
        "database: {sqlite_path: ':memory:'}\n", encoding="utf-8"
    )
    (base_dir / "models.yaml").write_text(
        """
workflows: {}
defaults: {provider: azure}
embeddings: {model: UDTEMBED3L, provider: azure}
""".strip(),
        encoding="utf-8",
    )
    (base_dir / "providers.yaml").write_text(
        """
providers:
  azure:
    api_base: "https://azure.example.com"
    api_version: "2024-05-01-preview"
    api_key_env: "AZURE_KEY"
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def embedding_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConfigService:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_embedding_config(config_dir)
    monkeypatch.setenv("CTM_CONFIG_PATH", str(config_dir))
    monkeypatch.setenv("AZURE_KEY", "secret")
    return ConfigService(config_path=config_dir)


def test_embedding_gateway_uses_provider_metadata(
    embedding_config: ConfigService, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: Dict[str, Any] = {}

    def fake_embedding(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in kwargs["input"]]}

    monkeypatch.setattr(
        "src.services.embedding_gateway.litellm_embedding", fake_embedding
    )

    gateway = EmbeddingGateway(config_service=embedding_config, use_fallback=False)
    result = gateway.embed_batch(["some text"])

    assert result == [[0.1, 0.2, 0.3]]
    assert captured["model"] == "UDTEMBED3L"
    assert captured["input"] == ["some text"]
    assert captured["custom_llm_provider"] == "azure"
    assert captured["api_key"] == "secret"
    assert captured["api_base"] == "https://azure.example.com"
    assert captured["api_version"] == "2024-05-01-preview"
