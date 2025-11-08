from __future__ import annotations

from dataclasses import dataclass

from ..services.technique_selector import TechniqueSelector


@dataclass
class DetectTechniqueWorkflow:
    selector: TechniqueSelector
    name: str = "detect_technique"

    def run(self, context: dict) -> dict:
        problem_description = context.get("problem_description")
        if not problem_description:
            raise ValueError("Context missing 'problem_description'.")
        return self.selector.recommend(problem_description)
