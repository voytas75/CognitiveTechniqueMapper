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
        self._sqlite = sqlite_client
        self._chroma = chroma_client
        self._llm = llm_gateway
        self._preprocessor = preprocessor or ProblemPreprocessor()
        self._embedder = embedder

    def recommend(self, problem_description: str) -> Dict[str, Any]:
        cleaned_description = self._preprocessor.normalize(problem_description)
        embedding_vector = self._generate_query_embedding(cleaned_description)
        candidate_matches = self._vector_search(cleaned_description, embedding_vector)
        return self._llm_reason_about_candidates(cleaned_description, candidate_matches)

    def _generate_query_embedding(self, normalized_text: str) -> List[float] | None:
        if not self._embedder:
            return None
        return self._embedder.embed(normalized_text)

    def _vector_search(self, normalized_text: str, query_embedding: List[float] | None) -> List[Dict[str, Any]]:
        if self._chroma and query_embedding is not None:
            results = self._chroma.query(query_embeddings=[query_embedding], n_results=5)
            matches: List[Dict[str, Any]] = []
            ids = results.get("ids", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances") or results.get("scores") or [[]]
            distance_row = distances[0] if distances else []
            for idx, metadata, document, distance in zip(ids, metadatas, documents, distance_row):
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

    def _build_prompt(self, normalized_text: str, candidates: List[Dict[str, Any]]) -> str:
        buffer = ["User problem:", normalized_text, "\nCandidate techniques:"]
        for candidate in candidates:
            metadata = candidate.get("metadata", {}) or {}
            name = metadata.get("name") or candidate.get("document", "Unknown technique")
            description = metadata.get("description") or candidate.get("document", "")
            buffer.append(f"- {name}: {description}")

        buffer.append(
            "\nReturn the best matching technique, explain why it fits, and outline application steps."
        )
        return "\n".join(buffer)
