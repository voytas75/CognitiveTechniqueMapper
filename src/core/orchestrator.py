"""Workflow orchestration utilities.

Updates:
    v0.1.0 - 2025-11-09 - Added Google-style docstrings and metadata log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Workflow(Protocol):
    """Protocol describing the workflow contract."""

    name: str

    def run(self, context: dict) -> dict:
        """Execute the workflow and return a response.

        Args:
            context (dict): Input data required by the workflow.

        Returns:
            dict: Workflow-specific result payload.
        """

        ...


@dataclass(slots=True)
class Orchestrator:
    workflows: dict[str, Workflow]

    def execute(self, workflow_name: str, context: dict) -> dict:
        """Run a registered workflow with the supplied context.

        Args:
            workflow_name (str): Name of the workflow to execute.
            context (dict): Arbitrary payload to pass to the workflow.

        Returns:
            dict: Output produced by the workflow.

        Raises:
            KeyError: If the workflow name is unknown.
        """

        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise KeyError(f"Workflow '{workflow_name}' is not registered.")
        return workflow.run(context)

    def register(self, workflow: Workflow) -> None:
        """Register a workflow implementation with the orchestrator.

        Args:
            workflow (Workflow): Workflow instance to make available.
        """

        self.workflows[workflow.name] = workflow
