"""Configuration service for Cognitive Technique Mapper.

Updates:
    v0.1.1 - 2025-11-09 - Remove shared max_tokens default to rely on per-model limits.
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
    v0.2.0 - 2025-11-09 - Added cache invalidation helper for runtime refreshes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from ..core.config_loader import ConfigLoader


def _object(value: object) -> dict[str, object]:
    """Return a string-keyed mapping or an empty mapping for dynamic YAML values."""

    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return {}
    return cast(dict[str, object], value)


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


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
        return _object(self._settings.get("app"))

    @property
    def logging_config(self) -> dict[str, Any]:
        """Return logging configuration settings."""
        return _object(self._settings.get("logging"))

    @property
    def database_config(self) -> dict[str, Any]:
        """Return database configuration values."""
        return _object(self._database.get("database"))

    @property
    def providers(self) -> dict[str, Any]:
        """Return provider configuration registry."""
        provider_section = _object(self._providers.get("providers"))
        return {
            name: self._expand_env_values(config)
            for name, config in provider_section.items()
        }

    def get_workflow_model_config(self, workflow: str) -> WorkflowModelConfig:
        """Return configuration for the requested workflow.

        Args:
            workflow (str): Name of the workflow to retrieve.

        Returns:
            WorkflowModelConfig: Workflow-specific model settings.

        Raises:
            KeyError: If the workflow configuration is missing.
        """

        workflows = _object(self._models.get("workflows"))
        defaults = _object(self._models.get("defaults"))
        raw_data = workflows.get(workflow)
        if not isinstance(raw_data, Mapping) or not all(
            isinstance(key, str) for key in raw_data
        ):
            raise KeyError(f"Workflow config not found for '{workflow}'")
        data = cast(dict[str, object], raw_data)

        model = data.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"Workflow config for '{workflow}' requires a non-empty 'model' value."
            )
        normalized_model = model.strip()

        return WorkflowModelConfig(
            workflow=workflow,
            model=normalized_model,
            temperature=_optional_float(
                data.get("temperature", defaults.get("temperature"))
            ),
            provider=_optional_text(data.get("provider", defaults.get("provider"))),
            max_tokens=_optional_int(data.get("max_tokens")),
        )

    def iter_workflow_configs(self) -> dict[str, WorkflowModelConfig]:
        """Return mapping of workflow names to configuration data.

        Returns:
            dict[str, WorkflowModelConfig]: Workflow configurations keyed by name.
        """

        workflows = _object(self._models.get("workflows"))
        return {name: self.get_workflow_model_config(name) for name in workflows}

    def get_embedding_config(self) -> EmbeddingModelConfig:
        """Return the embedding configuration used for vector generation.

        Returns:
            EmbeddingModelConfig: Embedding model name and provider metadata.

        Raises:
            KeyError: If the embedding configuration is missing.
        """

        raw_data = self._models.get("embeddings")
        if not isinstance(raw_data, Mapping) or not all(
            isinstance(key, str) for key in raw_data
        ):
            raise KeyError("Embedding configuration missing in config/models.yaml")
        data = cast(dict[str, object], raw_data)
        defaults = _object(self._models.get("defaults"))
        provider = _optional_text(data.get("provider", defaults.get("provider")))
        model = data.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "Embedding configuration requires a non-empty 'model' value."
            )
        return EmbeddingModelConfig(model=model.strip(), provider=provider)

    @staticmethod
    def clear_cache() -> None:
        """Clear cached configuration to reflect file updates."""

        ConfigLoader.load.cache_clear()

    @staticmethod
    def _expand_env_values(value: Any, *, current_key: str | None = None) -> Any:
        if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
            mapping = cast(Mapping[str, object], value)
            return {
                key: ConfigService._expand_env_values(entry, current_key=key)
                for key, entry in mapping.items()
            }
        if isinstance(value, list):
            return [
                ConfigService._expand_env_values(item, current_key=current_key)
                for item in value
            ]
        if isinstance(value, str) and current_key != "api_key_env":
            expanded = os.path.expandvars(value)
            return expanded
        return value
