from pathlib import Path

import pytest

import src.core.config_loader as config_loader
from src.core.config_loader import ConfigLoader
from src.services.config_service import ConfigService


def test_config_service_loads(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "settings.yaml").write_text(
        "app: {name: test, version: '0.0.1'}\n", encoding="utf-8"
    )
    (config_dir / "database.yaml").write_text(
        "database: {sqlite_path: ':memory:'}\n", encoding="utf-8"
    )
    (config_dir / "models.yaml").write_text(
        "workflows: {detect_technique: {model: dummy}}\ndefaults: {provider: mock}\n",
        encoding="utf-8",
    )
    (config_dir / "providers.yaml").write_text(
        "providers: {mock: {api_base: 'http://localhost'}}\n", encoding="utf-8"
    )

    monkeypatch.setenv("CTM_CONFIG_PATH", str(config_dir))
    service = ConfigService(config_path=Path(str(config_dir)))

    assert service.app_metadata["name"] == "test"
    workflow_config = service.get_workflow_model_config("detect_technique")
    assert workflow_config.model == "dummy"


def test_config_loader_uses_project_config_outside_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CTM_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = ConfigLoader().load("settings")

    assert settings["app"]["name"] == "Cognitive Technique Mapper"


def test_config_service_expands_provider_env_vars(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "settings.yaml").write_text("app: {name: test}\n", encoding="utf-8")
    (config_dir / "database.yaml").write_text(
        "database: {sqlite_path: ':memory:'}\n", encoding="utf-8"
    )
    (config_dir / "models.yaml").write_text(
        "workflows: {detect_technique: {model: dummy}}\ndefaults: {provider: mock}\n",
        encoding="utf-8",
    )
    (config_dir / "providers.yaml").write_text(
        'providers:\n  mock:\n    api_base: "${BASE_URL}"\n    api_key_env: "MOCK_KEY"\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("BASE_URL", "https://example.com")
    monkeypatch.setenv("MOCK_KEY", "secret")
    monkeypatch.setenv("CTM_CONFIG_PATH", str(config_dir))

    service = ConfigService(config_path=config_dir)
    providers = service.providers

    assert providers["mock"]["api_base"] == "https://example.com"
    assert providers["mock"]["api_key_env"] == "MOCK_KEY"


def test_get_workflow_model_config_requires_model(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "settings.yaml").write_text("app: {}\n", encoding="utf-8")
    (config_dir / "database.yaml").write_text(
        "database: {sqlite_path: ':memory:'}\n", encoding="utf-8"
    )
    (config_dir / "models.yaml").write_text(
        "workflows: {detect_technique: {}}\n", encoding="utf-8"
    )
    (config_dir / "providers.yaml").write_text("providers: {}\n", encoding="utf-8")

    monkeypatch.setenv("CTM_CONFIG_PATH", str(config_dir))
    service = ConfigService(config_path=config_dir)

    with pytest.raises(ValueError):
        service.get_workflow_model_config("detect_technique")


def test_get_embedding_config_requires_model(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "settings.yaml").write_text("app: {}\n", encoding="utf-8")
    (config_dir / "database.yaml").write_text(
        "database: {sqlite_path: ':memory:'}\n", encoding="utf-8"
    )
    (config_dir / "models.yaml").write_text("embeddings: {}\n", encoding="utf-8")
    (config_dir / "providers.yaml").write_text("providers: {}\n", encoding="utf-8")

    monkeypatch.setenv("CTM_CONFIG_PATH", str(config_dir))
    service = ConfigService(config_path=config_dir)

    with pytest.raises(ValueError):
        service.get_embedding_config()


def _write_config_templates(template_dir: Path) -> None:
    template_dir.mkdir()
    templates = {
        "settings.yaml": "app: {name: template}\n",
        "database.yaml": "database: {sqlite_path: ':memory:'}\n",
        "models.yaml": "workflows: {detect_technique: {model: template}}\n",
        "providers.yaml": "providers: {}\n",
    }
    for name, content in templates.items():
        (template_dir / name).write_text(content, encoding="utf-8")


def test_default_config_bootstraps_from_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    template_dir = tmp_path / "config.example"
    _write_config_templates(template_dir)
    monkeypatch.delenv("CTM_CONFIG_PATH", raising=False)
    monkeypatch.setattr(config_loader, "PROJECT_CONFIG_PATH", config_dir)
    monkeypatch.setattr(config_loader, "PROJECT_CONFIG_TEMPLATE_PATH", template_dir)

    settings = ConfigLoader().load("settings")

    assert settings["app"]["name"] == "template"
    for name in ("settings.yaml", "database.yaml", "models.yaml", "providers.yaml"):
        assert (config_dir / name).read_text(encoding="utf-8") == (
            template_dir / name
        ).read_text(encoding="utf-8")


def test_default_config_does_not_bootstrap_an_incomplete_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    template_dir = tmp_path / "config.example"
    config_dir.mkdir()
    _write_config_templates(template_dir)
    (config_dir / "settings.yaml").write_text("app: {name: local}\n", encoding="utf-8")
    monkeypatch.delenv("CTM_CONFIG_PATH", raising=False)
    monkeypatch.setattr(config_loader, "PROJECT_CONFIG_PATH", config_dir)
    monkeypatch.setattr(config_loader, "PROJECT_CONFIG_TEMPLATE_PATH", template_dir)

    with pytest.raises(FileNotFoundError, match="models.yaml"):
        ConfigLoader().load("models")

    assert not (config_dir / "models.yaml").exists()


def test_custom_config_path_does_not_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_config_dir = tmp_path / "custom-config"
    monkeypatch.setenv("CTM_CONFIG_PATH", str(custom_config_dir))

    with pytest.raises(FileNotFoundError, match="Config directory not found"):
        ConfigLoader()

    assert not custom_config_dir.exists()
