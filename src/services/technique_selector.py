"""Technique selection services.

Updates:
    v0.1.0 - 2025-11-09 - Added module docstring and method documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.preprocessor import ProblemPreprocessor
from ..core.llm_gateway import LLMGateway
from ..db.sqlite_client import SQLiteClient
from .embedding_gateway import EmbeddingGateway
from .technique_utils import compose_embedding_text

try:
    from ..db.chroma_client import ChromaClient
except RuntimeError:
    ChromaClient = None  # type: ignore


class TechniqueSelector:
    """Coordinates preprocessing, vector search, and LLM reasoning to pick a technique."""

    def __init__(
        self,
        sqlite_client: SQLiteClient,
        llm_gateway: LLMGateway,
        preprocessor: ProblemPreprocessor | None = None,
        embedder: EmbeddingGateway | None = None,
        chroma_client: ChromaClient | None = None,
    ) -> None:
        """Initialize dependencies for technique recommendation.

        Args:
            sqlite_client (SQLiteClient): Accessor for the techniques database.
            llm_gateway (LLMGateway): Gateway responsible for workflow prompts.
            preprocessor (ProblemPreprocessor | None): Text normalizer for user inputs.
            embedder (EmbeddingGateway | None): Embedding provider for vector searches.
            chroma_client (ChromaClient | None): Optional ChromaDB client for semantic search.
        """

        self._sqlite = sqlite_client
        self._chroma = chroma_client
        self._llm = llm_gateway
        self._preprocessor = preprocessor or ProblemPreprocessor()
        self._embedder = embedder

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
        return self._llm_reason_about_candidates(cleaned_description, candidate_matches)

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
        self, normalized_text: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prompt the LLM to select the best candidate from the shortlist.

        Args:
            normalized_text (str): Normalized user problem description.
            candidates (list[dict[str, Any]]): Candidate techniques with metadata.

        Returns:
            dict[str, Any]: Recommendation payload including the suggested technique.
        """

        if not candidates:
            return {
                "workflow": "detect_technique",
                "suggested_technique": None,
                "reasoning": "No techniques found. Please populate the database.",
                "steps": [],
            }

        prompt = self._build_prompt(normalized_text, candidates)
        response = self._llm.invoke("detect_technique", prompt)
        return {
            "workflow": "detect_technique",
            "suggested_technique": response,
            "matches": candidates,
        }

    def _build_prompt(
        self, normalized_text: str, candidates: List[Dict[str, Any]]
    ) -> str:
        """Construct the prompt sent to the detect_technique workflow.

        Args:
            normalized_text (str): Normalized problem description.
            candidates (list[dict[str, Any]]): Candidate technique metadata.

        Returns:
            str: Prompt string summarizing the problem and candidate techniques.
        """

        buffer = ["User problem:", normalized_text, "\nCandidate techniques:"]
        for candidate in candidates:
            metadata = candidate.get("metadata", {}) or {}
            name = metadata.get("name") or candidate.get(
                "document", "Unknown technique"
            )
            description = metadata.get("description") or candidate.get("document", "")
            buffer.append(f"- {name}: {description}")

        buffer.append(
            "\nReturn the best matching technique, explain why it fits, and outline application steps."
        )
        return "\n".join(buffer)
