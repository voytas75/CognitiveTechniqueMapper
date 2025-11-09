"""Workflow wrapper for candidate comparison."""

from __future__ import annotations

from dataclasses import dataclass

from ..services.comparison_service import ComparisonResult, ComparisonService


@dataclass
class CompareCandidatesWorkflow:
    comparison_service: ComparisonService
    name: str = "compare_candidates"

    def run(self, context: dict) -> dict:
        """Run the comparison workflow given the shortlist context."""

        recommendation = context.get("recommendation")
        matches = context.get("matches") or []
        if not recommendation or not isinstance(matches, list):
            raise ValueError("Comparison workflow requires recommendation and matches.")

        result: ComparisonResult = self.comparison_service.compare(
            recommendation,
            matches,
            focus=context.get("focus"),
            preference_summary=context.get("preference_summary"),
        )
        return {"workflow": self.name, "comparison": result.as_dict()}
