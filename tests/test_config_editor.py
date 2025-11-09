from pathlib import Path

import yaml

from src.services.config_editor import ConfigEditor


def _write_models_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "workflows": {
                    "detect_technique": {
                        "model": "azure/gpt-5",
                        "temperature": 1.0,
                        "provider": "azure",
                        "max_tokens": 1024,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_providers_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "azure": {
                        "api_base": "https://example.azure.com",
                        "api_version": "2024-05-01-preview",
                        "api_key_env": "AZURE_OPENAI_KEY",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_update_workflow_model(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    models_path = config_dir / "models.yaml"
    _write_models_config(models_path)

    editor = ConfigEditor(config_path=config_dir)
    updated = editor.update_workflow_model(
        "detect_technique",
        model="openai/gpt-4.1",
        temperature=0.4,
        provider="openai",
        max_tokens=2048,
    )

    with models_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    assert data["workflows"]["detect_technique"]["model"] == "openai/gpt-4.1"
    assert updated["temperature"] == 0.4
    assert updated["provider"] == "openai"
    assert updated["max_tokens"] == 2048


def test_update_provider_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    providers_path = config_dir / "providers.yaml"
    _write_providers_config(providers_path)

    editor = ConfigEditor(config_path=config_dir)
    updated = editor.update_provider(
        "azure",
        api_base="https://new.azure.com",
        api_version=None,
        api_key_env="NEW_AZURE_KEY",
        clear_api_version=True,
    )

    with providers_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    assert data["providers"]["azure"]["api_base"] == "https://new.azure.com"
    assert "api_version" not in data["providers"]["azure"]
    assert updated["api_key_env"] == "NEW_AZURE_KEY"
