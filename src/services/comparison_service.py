"""Candidate comparison helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.llm_gateway import LLMGateway
from .prompt_service import PromptService


@dataclass(slots=True)
class ComparisonResult:
    """Structured comparison between candidate techniques."""

    current_recommendation: Optional[str]
    best_alternative: Optional[str]
    comparison_points: List[Dict[str, Any]]
    decision_guidance: List[str]
    confidence_notes: Optional[str]
    raw_response: str

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable representation."""

        return {
            "current_recommendation": self.current_recommendation,
            "best_alternative": self.best_alternative,
            "comparison_points": self.comparison_points,
            "decision_guidance": self.decision_guidance,
            "confidence_notes": self.confidence_notes,
            "raw_response": self.raw_response,
        }


class ComparisonService:
    """Generates structured comparisons for candidate techniques."""

    def __init__(
        self,
        llm_gateway: LLMGateway,
        prompt_service: PromptService,
    ) -> None:
        self._llm = llm_gateway
        self._prompts = prompt_service

    def compare(
        self,
        recommendation: Dict[str, Any],
        matches: List[Dict[str, Any]],
        *,
        focus: Optional[str] = None,
        preference_summary: Optional[str] = None,
    ) -> ComparisonResult:
        """Compare shortlisted techniques and highlight trade-offs."""

        prompt = self._compose_prompt(
            recommendation,
            matches,
            focus=focus,
            preference_summary=preference_summary,
        )
        response = self._invoke_llm(prompt)
        parsed = self._parse_response(response)
        return ComparisonResult(
            current_recommendation=self._coerce_string(parsed.get("current_recommendation")),
            best_alternative=self._coerce_string(parsed.get("best_alternative")),
            comparison_points=self._coerce_points(parsed.get("comparison_points")),
            decision_guidance=self._coerce_list(parsed.get("decision_guidance")),
            confidence_notes=self._coerce_string(parsed.get("confidence_notes")),
            raw_response=response,
        )

    def _compose_prompt(
        self,
        recommendation: Dict[str, Any],
        matches: List[Dict[str, Any]],
        *,
        focus: Optional[str],
        preference_summary: Optional[str],
    ) -> str:
        template = self._prompts.get_prompt("compare_candidates").strip()
        sections = [template]
        if focus:
            sections.extend(["", "Technique focus override:", focus])
        serialized_recommendation = json.dumps(
            recommendation, indent=2, ensure_ascii=False
        )
        sections.extend(["", "Current recommendation payload:", serialized_recommendation])
        formatted_matches = json.dumps(matches, indent=2, ensure_ascii=False)
        sections.extend(["", "Candidate shortlist:", formatted_matches])
        if preference_summary:
            sections.extend(["", "Preference insights:", preference_summary])
        return "\n".join(sections)

    def _invoke_llm(self, prompt: str) -> str:
        try:
            return self._llm.invoke(
                "compare_candidates",
                prompt,
                response_format={"type": "json_object"},
            )
        except RuntimeError:
            return self._llm.invoke("compare_candidates", prompt)

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
    def _coerce_points(value: Any) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    points.append(
                        {
                            "technique": entry.get("technique"),
                            "strengths": entry.get("strengths"),
                            "risks": entry.get("risks"),
                            "best_for": entry.get("best_for"),
                        }
                    )
                else:
                    points.append(
                        {
                            "technique": None,
                            "strengths": str(entry),
                            "risks": None,
                            "best_for": None,
                        }
                    )
        return points
