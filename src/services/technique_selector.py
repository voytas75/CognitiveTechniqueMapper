"""Technique selection services.

Updates:
    v0.1.0 - 2025-11-09 - Added module docstring and method documentation.
    v0.2.0 - 2025-11-09 - Integrated prompt registry and structured recommendation parsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, cast

from ..core.llm_gateway import LLMGateway
from ..core.preprocessor import ProblemPreprocessor
from ..db.sqlite_client import SQLiteClient
from .embedding_gateway import EmbeddingGateway
from .preference_service import PreferenceService
from .prompt_service import PromptService
from .technique_vector_search import ChromaSearchClient, TechniqueVectorSearch

# TODO: Extract LLM prompt/response construction into a dedicated selector adapter.


@dataclass(slots=True)
class TechniqueRecommendation:
    """Structured recommendation payload returned by the LLM."""

    suggested_technique: str | None
    why_it_fits: str | None
    steps: List[str]
    raw_response: str

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the recommendation."""

        return {
            "suggested_technique": self.suggested_technique,
            "why_it_fits": self.why_it_fits,
            "steps": self.steps,
            "raw_response": self.raw_response,
        }


class TechniqueSelector:
    """Coordinates preprocessing, vector search, and LLM reasoning to pick a technique."""

    def __init__(
        self,
        sqlite_client: SQLiteClient,
        llm_gateway: LLMGateway,
        prompt_service: PromptService,
        preprocessor: ProblemPreprocessor | None = None,
        embedder: EmbeddingGateway | None = None,
        chroma_client: ChromaSearchClient | None = None,
        preference_service: PreferenceService | None = None,
    ) -> None:
        """Initialize dependencies for technique recommendation.

        Args:
            sqlite_client (SQLiteClient): Accessor for the techniques database.
            llm_gateway (LLMGateway): Gateway responsible for workflow prompts.
            prompt_service (PromptService): Loader supplying prompt templates.
            preprocessor (ProblemPreprocessor | None): Text normalizer for user inputs.
            embedder (EmbeddingGateway | None): Embedding provider for vector searches.
            chroma_client (ChromaClient | None): Optional ChromaDB client for semantic search.
        """

        self._llm = llm_gateway
        self._prompts = prompt_service
        self._preprocessor = preprocessor or ProblemPreprocessor()
        self._preferences = preference_service
        self._vector_searcher = TechniqueVectorSearch(
            sqlite_client=sqlite_client,
            embedder=embedder,
            chroma_client=chroma_client,
        )

    def recommend(
        self, problem_description: str, *, include_diagnostics: bool = False
    ) -> Dict[str, Any]:
        """Recommend a technique for a problem description.

        Args:
            problem_description (str): Raw problem statement supplied by the user.
            include_diagnostics (bool): When true, request LLM diagnostics comparing
                the recommendation with runner-up candidates.

        Returns:
            dict[str, Any]: Recommendation payload produced by the workflow.
        """

        cleaned_description = self._preprocessor.normalize(problem_description)
        embedding_vector = self._generate_query_embedding(cleaned_description)
        candidate_matches = self._vector_search(cleaned_description, embedding_vector)
        preference_summary = (
            self._preferences.preference_summary() if self._preferences else ""
        )
        adjusted_matches = self._apply_preference_adjustments(candidate_matches)
        result = self._llm_reason_about_candidates(
            cleaned_description,
            adjusted_matches,
            preference_summary=preference_summary or None,
        )
        if include_diagnostics:
            diagnostics = self._generate_selection_diagnostics(
                cleaned_description,
                adjusted_matches,
                result.get("recommendation"),
                preference_summary or None,
            )
            if diagnostics:
                result["diagnostics"] = diagnostics
        return result

    def _generate_query_embedding(self, normalized_text: str) -> List[float] | None:
        """Generate a query embedding through the vector-search adapter."""
        return self._vector_searcher.generate_query_embedding(normalized_text)

    def _vector_search(
        self, normalized_text: str, query_embedding: List[float] | None
    ) -> List[Dict[str, Any]]:
        """Search candidates through the vector-search adapter."""
        return self._vector_searcher.search(normalized_text, query_embedding)

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Delegate cosine similarity for backwards-compatible test seams."""
        return self._vector_searcher.cosine_similarity(vec_a, vec_b)

    def _llm_reason_about_candidates(
        self,
        normalized_text: str,
        candidates: List[Dict[str, Any]],
        *,
        preference_summary: str | None = None,
    ) -> Dict[str, Any]:
        """Prompt the LLM to select the best candidate from the shortlist.

        Args:
            normalized_text (str): Normalized user problem description.
            candidates (list[dict[str, Any]]): Candidate techniques with metadata.

        Returns:
            dict[str, Any]: Recommendation payload including the suggested technique.
        """

        if not candidates:
            empty = TechniqueRecommendation(
                suggested_technique=None,
                why_it_fits="No techniques found. Please populate the database.",
                steps=[],
                raw_response="",
            )
            return {
                "workflow": "detect_technique",
                "recommendation": empty.as_dict(),
                "matches": [],
                "preference_summary": preference_summary,
            }

        prompt = self._build_prompt(
            normalized_text, candidates, preference_summary=preference_summary
        )
        response = self._invoke_llm(prompt)
        recommendation = self._parse_recommendation(response)
        displayed_candidates = self._exclude_suggested_candidate(
            candidates, recommendation
        )
        return {
            "workflow": "detect_technique",
            "recommendation": recommendation.as_dict() if recommendation else None,
            "matches": displayed_candidates,
            "preference_summary": preference_summary,
        }

    def _exclude_suggested_candidate(
        self,
        candidates: List[Dict[str, Any]],
        recommendation: TechniqueRecommendation | None,
    ) -> List[Dict[str, Any]]:
        """Return five alternatives without duplicating the suggested technique."""
        if recommendation is None or not recommendation.suggested_technique:
            return candidates[:5]

        suggested_name = recommendation.suggested_technique.strip().casefold()
        for index, candidate in enumerate(candidates):
            metadata = self._object(candidate.get("metadata"))
            candidate_name = self._coerce_string(
                metadata.get("name")
                or candidate.get("name")
                or candidate.get("document")
            )
            if candidate_name and candidate_name.casefold() == suggested_name:
                recommendation.suggested_technique = candidate_name
                return [*candidates[:index], *candidates[index + 1 :]][:5]
        return candidates[:5]

    def _generate_selection_diagnostics(
        self,
        normalized_text: str,
        candidates: List[Dict[str, Any]],
        recommendation: Dict[str, Any] | None,
        preference_summary: str | None = None,
    ) -> Dict[str, Any] | None:
        """Produce diagnostic insight explaining the winning technique."""

        if not candidates or not recommendation:
            return None
        prompt = self._build_diagnostics_prompt(
            normalized_text,
            candidates,
            recommendation=recommendation,
            preference_summary=preference_summary,
        )
        try:
            response = self._llm.invoke(
                "diagnose_selection",
                prompt,
                response_format={"type": "json_object"},
            )
        except RuntimeError:
            response = self._llm.invoke("diagnose_selection", prompt)

        parsed = self._parse_json_response(response) or {}
        parsed["raw_response"] = response
        return parsed or None

    def _build_prompt(
        self,
        normalized_text: str,
        candidates: List[Dict[str, Any]],
        *,
        preference_summary: str | None = None,
    ) -> str:
        """Construct the prompt sent to the detect_technique workflow.

        Args:
            normalized_text (str): Normalized problem description.
            candidates (list[dict[str, Any]]): Candidate technique metadata.

        Returns:
            str: Prompt string summarizing the problem and candidate techniques.
        """

        instructions = self._prompts.get_prompt("detect_technique").strip()
        buffer = [instructions, "", "Problem:", normalized_text, "", "Candidates:"]
        for candidate in candidates:
            metadata = self._object(candidate.get("metadata"))
            if not metadata:
                metadata = dict(candidate)

            name = (
                metadata.get("name")
                or candidate.get("name")
                or candidate.get("document", "Unknown technique")
            )
            description = (
                metadata.get("description")
                or candidate.get("description")
                or candidate.get("document", "")
            )
            principles = metadata.get("core_principles") or candidate.get(
                "core_principles", ""
            )
            buffer.append(
                f"- name: {name}\n  description: {description}\n  core_principles: {principles}"
            )

        buffer.append(
            "\nReply strictly in JSON with keys 'suggested_technique', 'why_it_fits', and 'steps' (array)."
        )
        buffer.append(
            "Ensure 'steps' includes concrete, user-facing actions and limit to 5 entries."
        )
        if preference_summary:
            buffer.extend(
                [
                    "",
                    "User preference insights:",
                    preference_summary,
                ]
            )
        return "\n".join(buffer)

    def _build_diagnostics_prompt(
        self,
        normalized_text: str,
        candidates: List[Dict[str, Any]],
        *,
        recommendation: Dict[str, Any],
        preference_summary: str | None = None,
    ) -> str:
        """Construct a prompt describing candidate scores for diagnostics."""

        template = self._prompts.get_prompt("diagnose_selection").strip()
        payload = {
            "problem": normalized_text,
            "recommendation": recommendation,
            "candidates": [self._diagnostics_candidate(entry) for entry in candidates],
            "preference_summary": preference_summary,
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"{template}\n\nDiagnostics payload:\n{serialized}\n"

    def _apply_preference_adjustments(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Incorporate preference-based score adjustments."""

        if not self._preferences or not candidates:
            return candidates

        adjusted: List[Dict[str, Any]] = []
        for entry in candidates:
            candidate = dict(entry)
            metadata = self._object(candidate.get("metadata"))
            adjustment = self._preferences.score_adjustment(metadata)
            base_score = self._coerce_float(candidate.get("score"))
            if base_score is not None:
                candidate["base_score"] = base_score
                candidate["score"] = base_score + adjustment
            else:
                candidate["base_score"] = None
                candidate["score"] = adjustment
            candidate["preference_adjustment"] = adjustment
            adjusted.append(candidate)

        adjusted.sort(
            key=lambda item: self._coerce_float(item.get("score")) or 0.0,
            reverse=True,
        )
        return adjusted

    def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM with JSON response enforcement and fallback."""

        try:
            return self._llm.invoke(
                "detect_technique",
                prompt,
                response_format={"type": "json_object"},
            )
        except RuntimeError:
            return self._llm.invoke("detect_technique", prompt)

    def _parse_recommendation(self, response: str) -> TechniqueRecommendation | None:
        """Parse the LLM response into a structured recommendation."""

        parsed = self._parse_json_response(response)
        if not parsed:
            return None
        steps = self._coerce_steps(parsed.get("steps"))
        return TechniqueRecommendation(
            suggested_technique=self._coerce_string(parsed.get("suggested_technique")),
            why_it_fits=self._coerce_string(parsed.get("why_it_fits")),
            steps=steps,
            raw_response=response,
        )

    @staticmethod
    def _parse_json_response(response: str) -> Dict[str, Any] | None:
        """Attempt to parse the response as JSON, handling markdown fences."""

        cleaned = response.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("\n", 1)
            cleaned = parts[1] if len(parts) > 1 else ""
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _object(value: object) -> dict[str, object]:
        if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
            return cast(dict[str, object], value)
        return {}

    def _diagnostics_candidate(self, entry: Dict[str, Any]) -> dict[str, object]:
        metadata = self._object(entry.get("metadata"))
        return {
            "name": metadata.get("name") or entry.get("id") or entry.get("document"),
            "score": self._coerce_float(entry.get("score")),
            "base_score": self._coerce_float(entry.get("base_score")),
            "preference_adjustment": self._coerce_float(
                entry.get("preference_adjustment")
            ),
            "category": metadata.get("category"),
            "description": metadata.get("description") or entry.get("document"),
        }

    @staticmethod
    def _coerce_string(value: Any) -> str | None:
        """Convert value to string when possible."""

        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return str(value)

    @staticmethod
    def _coerce_steps(value: Any) -> List[str]:
        """Normalize the steps collection into a list of strings."""

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            steps: List[str] = []
            for entry in cast(Sequence[object], value):
                if entry is None:
                    continue
                steps.append(str(entry).strip())
            return steps
        if isinstance(value, str):
            segments = [segment.strip() for segment in value.split("\n") if segment]
            return segments
        return []

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return None

    def clear_embedding_cache(self) -> None:
        """Clear cached embeddings after dataset refresh operations."""
        self._vector_searcher.clear_embedding_cache()
