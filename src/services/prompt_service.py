"""Prompt loading utilities.

Updates:
    v0.2.0 - 2025-11-09 - Initial implementation of prompt registry loader.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import yaml


class PromptService:
    """Loads prompt templates defined in the registry file."""

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize the service and read the prompt registry.

        Args:
            base_path (Path | None): Optional override for the prompts directory.

        Raises:
            FileNotFoundError: If the registry file cannot be located.
        """

        resolved_path = base_path or Path(os.environ.get("CTM_PROMPTS_PATH", "prompts"))
        self._base_path = resolved_path.resolve()
        if not self._base_path.exists():
            raise FileNotFoundError(f"Prompt directory not found: {self._base_path}")

        self._registry_path = self._base_path / "registry.yaml"
        if not self._registry_path.exists():
            raise FileNotFoundError(f"Prompt registry missing: {self._registry_path}")

        self._registry = self._load_registry()

    @property
    def registry(self) -> Dict[str, Path]:
        """Return the prompt registry mapping names to file paths."""

        return self._registry.copy()

    def get_prompt(self, name: str) -> str:
        """Return the prompt text associated with the provided name.

        Args:
            name (str): Logical prompt identifier from the registry.

        Returns:
            str: Prompt template content.

        Raises:
            KeyError: If the prompt name does not exist in the registry.
            FileNotFoundError: If the prompt file path cannot be resolved.
        """

        path = self._registry.get(name)
        if not path:
            raise KeyError(f"Prompt '{name}' is not defined in the registry.")
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    def _load_registry(self) -> Dict[str, Path]:
        """Load and normalize the prompt registry mapping."""

        raw_mapping = yaml.safe_load(self._registry_path.read_text(encoding="utf-8"))
        if raw_mapping is None:
            return {}
        if not isinstance(raw_mapping, dict):
            raise ValueError(
                f"Prompt registry must be a mapping, got {type(raw_mapping).__name__}"
            )

        mapping = raw_mapping
        normalized: Dict[str, Path] = {}
        for name, relative in mapping.items():
            if not isinstance(name, str):
                raise ValueError(
                    f"Prompt registry keys must be strings, got {type(name).__name__}"
                )
            if not isinstance(relative, (str, os.PathLike)):
                raise ValueError(
                    "Prompt registry entries must map to string or path values, "
                    f"got {type(relative).__name__} for '{name}'"
                )

            prompt_path = Path(relative)
            if not prompt_path.is_absolute():
                parts = prompt_path.parts
                if parts and parts[0] == self._base_path.name:
                    prompt_path = self._base_path.joinpath(*parts[1:]).resolve()
                else:
                    prompt_path = (self._base_path / prompt_path).resolve()
            normalized[name] = prompt_path
        return normalized
