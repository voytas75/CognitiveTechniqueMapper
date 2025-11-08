from __future__ import annotations

from dataclasses import dataclass

from src.services.plan_generator import PlanGenerator


@dataclass
class GeneratePlanWorkflow:
    plan_generator: PlanGenerator
    name: str = "summarize_result"

    def run(self, context: dict) -> dict:
        technique_summary = context.get("technique_summary")
        if not technique_summary:
            raise ValueError("Context missing 'technique_summary'.")
        return self.plan_generator.generate(technique_summary)
