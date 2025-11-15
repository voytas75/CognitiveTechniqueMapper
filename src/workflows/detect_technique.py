"""Detect technique workflow.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..services.technique_selector import TechniqueSelector


@dataclass
class DetectTechniqueWorkflow:
    selector: TechniqueSelector
    name: str = "detect_technique"

    def run(self, context: dict) -> dict:
        """Run the detect technique workflow.

        Args:
            context (dict): Context payload containing `problem_description`.

        Returns:
            dict: Recommendation result from the selector.

        Raises:
            ValueError: If the problem description is missing from context.
        """

        problem_description = context.get("problem_description")
        if not problem_description:
            raise ValueError("Context missing 'problem_description'.")
        include_diagnostics = bool(context.get("include_diagnostics"))
        return self.selector.recommend(
            problem_description, include_diagnostics=include_diagnostics
        )
