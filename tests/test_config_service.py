from pathlib import Path

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


def test_config_service_expands_provider_env_vars(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "settings.yaml").write_text("app: {name: test}\n", encoding="utf-8")
    (config_dir / "database.yaml").write_text(
        "database: {sqlite_path: ':memory:'}\n", encoding="utf-8"
    )
    (config_dir / "models.yaml").write_text(
        "workflows: {detect_technique: {model: dummy}}\n"
        "defaults: {provider: mock}\n",
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
