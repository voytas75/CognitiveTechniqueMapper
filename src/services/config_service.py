"""Configuration service for Cognitive Technique Mapper.

Updates:
    v0.1.1 - 2025-11-09 - Remove shared max_tokens default to rely on per-model limits.
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
    v0.2.0 - 2025-11-09 - Added cache invalidation helper for runtime refreshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.config_loader import ConfigLoader


@dataclass(slots=True, frozen=True)
class WorkflowModelConfig:
    """Workflow-specific model parameters."""

    workflow: str
    model: str
    temperature: float | None = None
    provider: str | None = None
    max_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class EmbeddingModelConfig:
    """Embedding model parameters."""

    model: str
    provider: str | None = None


class ConfigService:
    """Loads and exposes configuration for CTM components."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize configuration caches.

        Args:
            config_path (Path | None): Optional override for the configuration directory.
        """

        self._loader = ConfigLoader(base_path=config_path)
        self._settings = self._loader.load("settings")
        self._database = self._loader.load("database")
        self._models = self._loader.load("models")
        self._providers = self._loader.load("providers")

    @property
    def app_metadata(self) -> dict[str, Any]:
        """Return general application metadata."""
        return self._settings.get("app", {})

    @property
    def logging_config(self) -> dict[str, Any]:
        """Return logging configuration settings."""
        return self._settings.get("logging", {})

    @property
    def database_config(self) -> dict[str, Any]:
        """Return database configuration values."""
        return self._database.get("database", {})

    @property
    def providers(self) -> dict[str, Any]:
        """Return provider configuration registry."""
        return self._providers.get("providers", {})

    def get_workflow_model_config(self, workflow: str) -> WorkflowModelConfig:
        """Return configuration for the requested workflow.

        Args:
            workflow (str): Name of the workflow to retrieve.

        Returns:
            WorkflowModelConfig: Workflow-specific model settings.

        Raises:
            KeyError: If the workflow configuration is missing.
        """

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
            max_tokens=data.get("max_tokens"),
        )

    def iter_workflow_configs(self) -> dict[str, WorkflowModelConfig]:
        """Return mapping of workflow names to configuration data.

        Returns:
            dict[str, WorkflowModelConfig]: Workflow configurations keyed by name.
        """

        workflows = self._models.get("workflows", {})
        return {name: self.get_workflow_model_config(name) for name in workflows}

    def get_embedding_config(self) -> EmbeddingModelConfig:
        """Return the embedding configuration used for vector generation.

        Returns:
            EmbeddingModelConfig: Embedding model name and provider metadata.

        Raises:
            KeyError: If the embedding configuration is missing.
        """

        data = self._models.get("embeddings")
        if not data:
            raise KeyError("Embedding configuration missing in config/models.yaml")
        return EmbeddingModelConfig(
            model=data.get("model"), provider=data.get("provider")
        )

    @staticmethod
    def clear_cache() -> None:
        """Clear cached configuration to reflect file updates."""

        ConfigLoader.load.cache_clear()
