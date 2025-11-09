"""Mutable configuration helpers.

Updates:
    v0.2.0 - 2025-11-09 - Added workflow and provider editing utilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def _resolve_config_path(config_path: Path | None) -> Path:
    resolved = config_path or Path(os.environ.get("CTM_CONFIG_PATH", "config"))
    directory = resolved.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Config directory not found: {directory}")
    return directory


@dataclass(slots=True)
class ConfigEditor:
    """Provides mutation operations for YAML configuration files."""

    base_path: Path

    def __init__(self, config_path: Path | None = None) -> None:
        self.base_path = _resolve_config_path(config_path)

    def update_workflow_model(
        self,
        workflow: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        provider: str | None = None,
        max_tokens: int | None = None,
        clear_max_tokens: bool = False,
    ) -> Dict[str, Any]:
        """Update model mapping for the specified workflow."""

        path, data = self._load_yaml("models.yaml")
        workflows = data.setdefault("workflows", {})
        if workflow not in workflows:
            raise KeyError(f"Workflow '{workflow}' not defined in models.yaml")

        entry = dict(workflows[workflow])

        if model is not None:
            entry["model"] = model
        if temperature is not None:
            entry["temperature"] = float(temperature)
        if provider is not None:
            entry["provider"] = provider
        if max_tokens is not None:
            entry["max_tokens"] = int(max_tokens)
        elif clear_max_tokens and "max_tokens" in entry:
            entry.pop("max_tokens")

        workflows[workflow] = entry
        self._write_yaml(path, data)
        return entry

    def update_provider(
        self,
        provider: str,
        *,
        api_base: str | None = None,
        api_version: str | None = None,
        api_key_env: str | None = None,
        clear_api_version: bool = False,
    ) -> Dict[str, Any]:
        """Update provider metadata in providers.yaml."""

        path, data = self._load_yaml("providers.yaml")
        providers = data.setdefault("providers", {})
        if provider not in providers:
            providers[provider] = {}

        entry = dict(providers[provider])

        if api_base is not None:
            entry["api_base"] = api_base
        if api_version is not None:
            entry["api_version"] = api_version
        elif clear_api_version and "api_version" in entry:
            entry.pop("api_version")
        if api_key_env is not None:
            entry["api_key_env"] = api_key_env

        providers[provider] = entry
        self._write_yaml(path, data)
        return entry

    def _load_yaml(self, name: str) -> tuple[Path, Dict[str, Any]]:
        path = self.base_path / name
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return path, data

    def _write_yaml(self, path: Path, data: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
