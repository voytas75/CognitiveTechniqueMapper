from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..services.config_service import ConfigService


@dataclass
class ConfigUpdateWorkflow:
    name: str = "config_update"
    config_path: Path | None = None

    def run(self, context: dict) -> dict:
        service = ConfigService(config_path=self.config_path)
        return {
            "app": service.app_metadata,
            "database": service.database_config,
            "workflows": {
                workflow: config.__dict__ for workflow, config in service.iter_workflow_configs().items()
            },
        }
