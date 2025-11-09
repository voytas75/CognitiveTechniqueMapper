"""Technique selection services.

Updates:
    v0.1.0 - 2025-11-09 - Added module docstring and method documentation.
    v0.2.0 - 2025-11-09 - Integrated prompt registry and structured recommendation parsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from ..core.preprocessor import ProblemPreprocessor
from ..core.llm_gateway import LLMGateway
from ..db.sqlite_client import SQLiteClient
from .embedding_gateway import EmbeddingGateway
from .preference_service import PreferenceService
from .technique_utils import compose_embedding_text
from .prompt_service import PromptService

try:
    from ..db.chroma_client import ChromaClient
except RuntimeError:
    ChromaClient = None  # type: ignore


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
        chroma_client: ChromaClient | None = None,
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

        self._sqlite = sqlite_client
        self._chroma = chroma_client
        self._llm = llm_gateway
        self._prompts = prompt_service
        self._preprocessor = preprocessor or ProblemPreprocessor()
        self._embedder = embedder
        self._preferences = preference_service

    def recommend(self, problem_description: str) -> Dict[str, Any]:
        """Recommend a technique for a problem description.

        Args:
            problem_description (str): Raw problem statement supplied by the user.

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
        return self._llm_reason_about_candidates(
            cleaned_description,
            adjusted_matches,
            preference_summary=preference_summary or None,
        )

    def _generate_query_embedding(self, normalized_text: str) -> List[float] | None:
        """Generate an embedding for the normalized text if an embedder is available.

        Args:
            normalized_text (str): Normalized problem description.

        Returns:
            list[float] | None: Embedding vector or `None` when embeddings are disabled.
        """

        if not self._embedder:
            return None
        return self._embedder.embed(normalized_text)

    def _vector_search(
        self, normalized_text: str, query_embedding: List[float] | None
    ) -> List[Dict[str, Any]]:
        """Search for candidate techniques using vector similarity or database fallback.

        Args:
            normalized_text (str): Normalized problem description.
            query_embedding (list[float] | None): Embedding vector for the query.

        Returns:
            list[dict[str, Any]]: Candidate matches containing metadata and scores.
        """

        if self._chroma and query_embedding is not None:
            results = self._chroma.query(
                query_embeddings=[query_embedding], n_results=5
            )
            matches: List[Dict[str, Any]] = []
            ids = results.get("ids", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances") or results.get("scores") or [[]]
            distance_row = distances[0] if distances else []
            for idx, metadata, document, distance in zip(
                ids, metadatas, documents, distance_row
            ):
                similarity = None
                if distance is not None:
                    try:
                        similarity = 1 / (1 + float(distance))
                    except (TypeError, ValueError):
                        similarity = None
                match = {
                    "id": idx,
                    "metadata": metadata,
                    "document": document,
                    "distance": distance,
                    "score": similarity,
                }
                matches.append(match)
            return matches

        stored = [dict(row) for row in self._sqlite.fetch_all()]
        if not stored:
            return []

        if query_embedding is None or not self._embedder:
            return stored[:5]

        scored_matches = []
        for item in stored:
            technique_text = compose_embedding_text(item)
            technique_embedding = self._embedder.embed(technique_text)
            score = self._cosine_similarity(query_embedding, technique_embedding)
            scored_matches.append(
                {
                    "id": item.get("id"),
                    "metadata": item,
                    "document": item.get("description", ""),
                    "score": score,
                }
            )

        scored_matches.sort(key=lambda entry: entry["score"], reverse=True)
        return scored_matches[:5]

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors with safe guards.

        Args:
            vec_a (list[float]): First embedding vector.
            vec_b (list[float]): Second embedding vector.

        Returns:
            float: Cosine similarity score between 0.0 and 1.0.
        """

        if not vec_a or not vec_b:
            return 0.0
        length = min(len(vec_a), len(vec_b))
        dot = sum(vec_a[i] * vec_b[i] for i in range(length))
        norm_a = sum(component * component for component in vec_a[:length]) ** 0.5
        norm_b = sum(component * component for component in vec_b[:length]) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

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
        return {
            "workflow": "detect_technique",
            "recommendation": recommendation.as_dict() if recommendation else None,
            "matches": candidates,
            "preference_summary": preference_summary,
        }

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
            metadata = candidate.get("metadata", {}) or {}
            if not metadata and isinstance(candidate, dict):
                metadata = {key: candidate.get(key) for key in candidate.keys()}

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

    def _apply_preference_adjustments(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Incorporate preference-based score adjustments."""

        if not self._preferences or not candidates:
            return candidates

        adjusted: List[Dict[str, Any]] = []
        for entry in candidates:
            candidate = dict(entry)
            metadata = candidate.get("metadata") or {}
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
            for entry in value:
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
