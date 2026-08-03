"""Configuration loader utilities.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings with metadata.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CONFIG_PATH = PROJECT_ROOT / "config"
PROJECT_CONFIG_TEMPLATE_PATH = PROJECT_ROOT / "config.example"
CONFIG_FILENAMES = ("database.yaml", "models.yaml", "providers.yaml", "settings.yaml")

logger = logging.getLogger(__name__)


def _bootstrap_default_config() -> None:
    """Create the default local config directory from tracked templates once."""

    if PROJECT_CONFIG_PATH.exists():
        return

    missing_templates = [
        name
        for name in CONFIG_FILENAMES
        if not (PROJECT_CONFIG_TEMPLATE_PATH / name).is_file()
    ]
    if missing_templates:
        missing = ", ".join(missing_templates)
        raise FileNotFoundError(
            f"Config templates incomplete in {PROJECT_CONFIG_TEMPLATE_PATH}: {missing}"
        )

    staging_path = PROJECT_CONFIG_PATH.with_name(
        f".{PROJECT_CONFIG_PATH.name}.bootstrap-{uuid.uuid4().hex}"
    )
    try:
        staging_path.mkdir()
        for name in CONFIG_FILENAMES:
            shutil.copy2(PROJECT_CONFIG_TEMPLATE_PATH / name, staging_path / name)
        staging_path.replace(PROJECT_CONFIG_PATH)
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    logger.info(
        "initialized local configuration from templates: %s", PROJECT_CONFIG_PATH
    )


class ConfigLoader:
    """Loads YAML configuration files from the project's config directory."""

    def __init__(self, base_path: Path | None = None) -> None:
        """Configure the loader with the base directory location.

        Args:
            base_path (Path | None): Custom configuration directory if provided.

        Raises:
            FileNotFoundError: If the resolved configuration path does not exist.
        """

        configured_path = os.environ.get("CTM_CONFIG_PATH")
        if base_path is not None:
            self._base_path = base_path
        elif configured_path:
            self._base_path = Path(configured_path).resolve()
        else:
            _bootstrap_default_config()
            self._base_path = PROJECT_CONFIG_PATH
        if not self._base_path.exists():
            raise FileNotFoundError(f"Config directory not found: {self._base_path}")

    def _resolve(self, name: str) -> Path:
        """Resolve a configuration name to a concrete YAML file path.

        Args:
            name (str): Logical configuration name (with or without `.yaml`).

        Returns:
            Path: Path to the requested configuration file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """

        candidate = self._base_path / name
        if candidate.suffix != ".yaml":
            candidate = candidate.with_suffix(".yaml")
        if not candidate.exists():
            raise FileNotFoundError(f"Config file not found: {candidate}")
        return candidate

    @functools.lru_cache(maxsize=None)
    def load(self, name: str) -> Dict[str, Any]:
        """Load and cache a configuration file as a dictionary.

        Args:
            name (str): Logical configuration name to load.

        Returns:
            dict[str, Any]: Parsed YAML content from disk.
        """

        path = self._resolve(name)
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}


def load_config(name: str, base_path: Path | None = None) -> Dict[str, Any]:
    """Load a configuration file without explicitly creating a loader.

    Args:
        name (str): Logical configuration name.
        base_path (Path | None): Optional path override for configuration files.

    Returns:
        dict[str, Any]: Parsed configuration data.
    """

    loader = ConfigLoader(base_path=base_path)
    return loader.load(name)
