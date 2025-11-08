from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Loads YAML configuration files from the project's config directory."""

    def __init__(self, base_path: Path | None = None) -> None:
        self._base_path = base_path or Path(os.environ.get("CTM_CONFIG_PATH", "config")).resolve()
        if not self._base_path.exists():
            raise FileNotFoundError(f"Config directory not found: {self._base_path}")

    def _resolve(self, name: str) -> Path:
        candidate = self._base_path / name
        if candidate.suffix != ".yaml":
            candidate = candidate.with_suffix(".yaml")
        if not candidate.exists():
            raise FileNotFoundError(f"Config file not found: {candidate}")
        return candidate

    @functools.lru_cache(maxsize=None)
    def load(self, name: str) -> Dict[str, Any]:
        path = self._resolve(name)
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}


def load_config(name: str, base_path: Path | None = None) -> Dict[str, Any]:
    """Convenience helper for loading a config file without instantiating the loader."""
    loader = ConfigLoader(base_path=base_path)
    return loader.load(name)
