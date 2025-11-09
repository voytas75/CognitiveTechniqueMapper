"""Explanation workflow helpers.

Updates:
    v0.2.0 - 2025-11-09 - Added structured explain_logic prompt handling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from ..core.llm_gateway import LLMGateway
from .prompt_service import PromptService


@dataclass(slots=True)
class ExplanationResult:
    """Structured explanation produced by the explain_logic workflow."""

    overview: str | None
    key_factors: List[str]
    risks: List[str]
    next_steps: List[str]
    raw_response: str

    def as_dict(self) -> Dict[str, Any]:
        """Return the explanation as a JSON-serializable dictionary."""

        return {
            "overview": self.overview,
            "key_factors": self.key_factors,
            "risks": self.risks,
            "next_steps": self.next_steps,
            "raw_response": self.raw_response,
        }


class ExplanationService:
    """Generates structured explanations for technique recommendations."""

    def __init__(self, llm_gateway: LLMGateway, prompt_service: PromptService) -> None:
        self._llm = llm_gateway
        self._prompts = prompt_service

    def explain(
        self,
        recommendation: Dict[str, Any],
        *,
        problem_description: str | None = None,
    ) -> ExplanationResult:
        """Generate a structured explanation for the current recommendation."""

        prompt = self._compose_prompt(recommendation, problem_description)
        response = self._invoke_llm(prompt)
        parsed = self._parse_response(response)
        return ExplanationResult(
            overview=self._coerce_string(parsed.get("overview")),
            key_factors=self._coerce_list(parsed.get("key_factors")),
            risks=self._coerce_list(parsed.get("risks")),
            next_steps=self._coerce_list(parsed.get("next_steps")),
            raw_response=response,
        )

    def _compose_prompt(
        self, recommendation: Dict[str, Any], problem_description: str | None
    ) -> str:
        template = self._prompts.get_prompt("explain_logic").strip()
        sections = [template]
        if problem_description:
            sections.extend(["", "Problem description:", problem_description])
        serialized = json.dumps(recommendation, indent=2, ensure_ascii=False)
        sections.extend(["", "Recommendation payload:", serialized])
        return "\n".join(sections)

    def _invoke_llm(self, prompt: str) -> str:
        try:
            return self._llm.invoke(
                "explain_logic", prompt, response_format={"type": "json_object"}
            )
        except RuntimeError:
            return self._llm.invoke("explain_logic", prompt)

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
    def _coerce_string(value: Any) -> str | None:
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
            parts = [
                segment.strip() for segment in value.split("\n") if segment.strip()
            ]
            return parts
        return []
