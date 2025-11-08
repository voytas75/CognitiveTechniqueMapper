from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config_loader import ConfigLoader


@dataclass(slots=True, frozen=True)
class WorkflowModelConfig:
    workflow: str
    model: str
    temperature: float | None = None
    provider: str | None = None
    max_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class EmbeddingModelConfig:
    model: str
    provider: str | None = None


class ConfigService:
    """Loads and exposes configuration for CTM components."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._loader = ConfigLoader(base_path=config_path)
        self._settings = self._loader.load("settings")
        self._database = self._loader.load("database")
        self._models = self._loader.load("models")
        self._providers = self._loader.load("providers")

    @property
    def app_metadata(self) -> dict[str, Any]:
        return self._settings.get("app", {})

    @property
    def logging_config(self) -> dict[str, Any]:
        return self._settings.get("logging", {})

    @property
    def database_config(self) -> dict[str, Any]:
        return self._database.get("database", {})

    @property
    def providers(self) -> dict[str, Any]:
        return self._providers.get("providers", {})

    def get_workflow_model_config(self, workflow: str) -> WorkflowModelConfig:
        workflows = self._models.get("workflows", {})
        defaults = self._models.get("defaults", {})
        data = workflows.get(workflow)

        if not data:
            raise KeyError(f"Workflow config not found for '{workflow}'")

        return WorkflowModelConfig(
            workflow=workflow,
            model=data.get("model"),
            temperature=data.get("temperature", defaults.get("temperature")),
            provider=data.get("provider", defaults.get("provider")),
            max_tokens=data.get("max_tokens", defaults.get("max_tokens")),
        )

    def iter_workflow_configs(self) -> dict[str, WorkflowModelConfig]:
        workflows = self._models.get("workflows", {})
        return {name: self.get_workflow_model_config(name) for name in workflows}

    def get_embedding_config(self) -> EmbeddingModelConfig:
        data = self._models.get("embeddings")
        if not data:
            raise KeyError("Embedding configuration missing in config/models.yaml")
        return EmbeddingModelConfig(model=data.get("model"), provider=data.get("provider"))
