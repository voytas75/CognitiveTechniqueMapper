from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Workflow(Protocol):
    name: str

    def run(self, context: dict) -> dict:
        ...


@dataclass(slots=True)
class Orchestrator:
    workflows: dict[str, Workflow]

    def execute(self, workflow_name: str, context: dict) -> dict:
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise KeyError(f"Workflow '{workflow_name}' is not registered.")
        return workflow.run(context)

    def register(self, workflow: Workflow) -> None:
        self.workflows[workflow.name] = workflow
