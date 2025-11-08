"""Generate plan workflow.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..services.plan_generator import PlanGenerator


@dataclass
class GeneratePlanWorkflow:
    plan_generator: PlanGenerator
    name: str = "summarize_result"

    def run(self, context: dict) -> dict:
        """Run the summarize_result workflow.

        Args:
            context (dict): Payload containing `technique_summary`.

        Returns:
            dict: Plan data produced by the plan generator.

        Raises:
            ValueError: If no technique summary is provided.
        """

        technique_summary = context.get("technique_summary")
        if not technique_summary:
            raise ValueError("Context missing 'technique_summary'.")
        return self.plan_generator.generate(technique_summary)
