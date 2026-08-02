"""Workflow wrapper for technique simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ..services.simulation_service import SimulationResult, SimulationService


def _optional_text(value: object) -> str | None:
    """Return a textual context value or ``None`` for non-text input."""

    return value if isinstance(value, str) else None


@dataclass
class SimulateTechniqueWorkflow:
    simulation_service: SimulationService
    name: str = "simulate_technique"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run the simulation workflow with the provided context."""

        raw_recommendation = context.get("recommendation")
        if (
            not isinstance(raw_recommendation, dict)
            or not raw_recommendation
            or not all(isinstance(key, str) for key in raw_recommendation)
        ):
            raise ValueError("Simulation requires a recommendation payload.")
        recommendation = cast(dict[str, Any], raw_recommendation)

        result: SimulationResult = self.simulation_service.simulate(
            recommendation,
            problem_description=_optional_text(context.get("problem_description")),
            scenario=_optional_text(context.get("scenario")),
            preference_summary=_optional_text(context.get("preference_summary")),
        )
        return {"workflow": self.name, "simulation": result.as_dict()}
