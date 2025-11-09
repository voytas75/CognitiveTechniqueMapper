"""Workflow orchestration utilities.

Updates:
    v0.1.0 - 2025-11-09 - Added Google-style docstrings and metadata log.
    v0.3.0 - 2025-05-09 - Instrumented workflows with structured duration logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
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

    _logger = logging.getLogger(__name__)

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
        started = perf_counter()
        try:
            result = workflow.run(context)
        except Exception as exc:
            duration_ms = (perf_counter() - started) * 1000
            self._logger.error(
                "workflow_failed",
                extra={
                    "tool": workflow_name,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(exc),
                },
                exc_info=True,
            )
            raise

        duration_ms = (perf_counter() - started) * 1000
        context_keys = sorted(context.keys())
        self._logger.info(
            "workflow_completed",
            extra={
                "tool": workflow_name,
                "duration_ms": round(duration_ms, 2),
                "context_keys": context_keys,
            },
        )
        return result

    def register(self, workflow: Workflow) -> None:
        """Register a workflow implementation with the orchestrator.

        Args:
            workflow (Workflow): Workflow instance to make available.
        """

        self.workflows[workflow.name] = workflow
