"""Technique simulation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.llm_gateway import LLMGateway
from .prompt_service import PromptService


@dataclass(slots=True)
class SimulationResult:
    """Structured response for technique simulation."""

    simulation_overview: Optional[str]
    scenario_variations: List[Dict[str, Any]]
    cautions: List[str]
    recommended_follow_up: List[str]
    raw_response: str

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable representation."""

        return {
            "simulation_overview": self.simulation_overview,
            "scenario_variations": self.scenario_variations,
            "cautions": self.cautions,
            "recommended_follow_up": self.recommended_follow_up,
            "raw_response": self.raw_response,
        }


class SimulationService:
    """Creates scenario-based simulations for recommended techniques."""

    def __init__(
        self, llm_gateway: LLMGateway, prompt_service: PromptService
    ) -> None:
        self._llm = llm_gateway
        self._prompts = prompt_service

    def simulate(
        self,
        recommendation: Dict[str, Any],
        *,
        problem_description: Optional[str],
        scenario: Optional[str],
        preference_summary: Optional[str] = None,
    ) -> SimulationResult:
        """Generate a scenario walkthrough for the selected technique."""

        prompt = self._compose_prompt(
            recommendation,
            problem_description=problem_description,
            scenario=scenario,
            preference_summary=preference_summary,
        )
        response = self._invoke_llm(prompt)
        parsed = self._parse_response(response)
        return SimulationResult(
            simulation_overview=self._coerce_string(parsed.get("simulation_overview")),
            scenario_variations=self._coerce_variations(parsed.get("scenario_variations")),
            cautions=self._coerce_list(parsed.get("cautions")),
            recommended_follow_up=self._coerce_list(parsed.get("recommended_follow_up")),
            raw_response=response,
        )

    def _compose_prompt(
        self,
        recommendation: Dict[str, Any],
        *,
        problem_description: Optional[str],
        scenario: Optional[str],
        preference_summary: Optional[str],
    ) -> str:
        template = self._prompts.get_prompt("simulate_technique").strip()
        sections = [template]
        if problem_description:
            sections.extend(["", "Problem description:", problem_description])
        if scenario:
            sections.extend(["", "Scenario focus:", scenario])
        serialized = json.dumps(recommendation, indent=2, ensure_ascii=False)
        sections.extend(["", "Recommendation payload:", serialized])
        if preference_summary:
            sections.extend(["", "Preference insights:", preference_summary])
        return "\n".join(sections)

    def _invoke_llm(self, prompt: str) -> str:
        try:
            return self._llm.invoke(
                "simulate_technique",
                prompt,
                response_format={"type": "json_object"},
            )
        except RuntimeError:
            return self._llm.invoke("simulate_technique", prompt)

    @staticmethod
    def _parse_response(response: str) -> Dict[str, Any]:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("\n", 1)
            cleaned = parts[1] if len(parts) > 1 else ""
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {}

    @staticmethod
    def _coerce_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        return str(value)

    @staticmethod
    def _coerce_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [segment.strip() for segment in value.split("\n") if segment.strip()]
        return []

    @staticmethod
    def _coerce_variations(value: Any) -> List[Dict[str, Any]]:
        variations: List[Dict[str, Any]] = []
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    variations.append(
                        {
                            "name": entry.get("name"),
                            "outcome": entry.get("outcome"),
                            "guidance": entry.get("guidance"),
                        }
                    )
                else:
                    variations.append(
                        {"name": None, "outcome": str(entry), "guidance": None}
                    )
        return variations
