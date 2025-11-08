"""Config update workflow definitions.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..services.config_service import ConfigService


@dataclass
class ConfigUpdateWorkflow:
    name: str = "config_update"
    config_path: Path | None = None

    def run(self, context: dict) -> dict:
        """Return configuration details suitable for CLI rendering.

        Args:
            context (dict): Unused, maintained for workflow interface compatibility.

        Returns:
            dict: Aggregated configuration data to display.
        """

        service = ConfigService(config_path=self.config_path)
        return {
            "app": service.app_metadata,
            "database": service.database_config,
            "workflows": {
                workflow: config.__dict__
                for workflow, config in service.iter_workflow_configs().items()
            },
        }
