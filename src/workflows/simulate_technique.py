"""Workflow wrapper for technique simulation."""

from __future__ import annotations

from dataclasses import dataclass

from ..services.simulation_service import SimulationResult, SimulationService


@dataclass
class SimulateTechniqueWorkflow:
    simulation_service: SimulationService
    name: str = "simulate_technique"

    def run(self, context: dict) -> dict:
        """Run the simulation workflow with the provided context."""

        recommendation = context.get("recommendation")
        if not recommendation:
            raise ValueError("Simulation requires a recommendation payload.")

        result: SimulationResult = self.simulation_service.simulate(
            recommendation,
            problem_description=context.get("problem_description"),
            scenario=context.get("scenario"),
            preference_summary=context.get("preference_summary"),
        )
        return {"workflow": self.name, "simulation": result.as_dict()}
