"""Config update workflow definitions.

Updates:
    v0.1.1 - 2025-11-09 - Use dataclass conversion helper for workflow configs.
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
                workflow: asdict(config)
                for workflow, config in service.iter_workflow_configs().items()
            },
        }
