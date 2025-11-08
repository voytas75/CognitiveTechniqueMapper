from __future__ import annotations

from typing import Any, Dict, List

from src.core.preprocessor import ProblemPreprocessor
from src.core.llm_gateway import LLMGateway
from src.db.sqlite_client import SQLiteClient

try:
    from src.db.chroma_client import ChromaClient
except RuntimeError:
    ChromaClient = None  # type: ignore


class TechniqueSelector:
    """Coordinates preprocessing, vector search, and LLM reasoning to pick a technique."""

    def __init__(
        self,
        sqlite_client: SQLiteClient,
        llm_gateway: LLMGateway,
        preprocessor: ProblemPreprocessor | None = None,
        chroma_client: ChromaClient | None = None,
    ) -> None:
        self._sqlite = sqlite_client
        self._chroma = chroma_client
        self._llm = llm_gateway
        self._preprocessor = preprocessor or ProblemPreprocessor()

    def recommend(self, problem_description: str) -> Dict[str, Any]:
        cleaned_description = self._preprocessor.normalize(problem_description)
        candidate_matches = self._vector_search(cleaned_description)
        return self._llm_reason_about_candidates(cleaned_description, candidate_matches)

    def _vector_search(self, normalized_text: str) -> List[Dict[str, Any]]:
        if not self._chroma:
            stored = self._sqlite.fetch_all()
            return [dict(row) for row in stored]

        # TODO: replace placeholder embedding logic with real embeddings once available.
        pseudo_embedding = [[float(len(normalized_text))]]
        results = self._chroma.query(query_embeddings=pseudo_embedding, n_results=5)

        matches: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        for idx, metadata, document in zip(ids, metadatas, documents):
            match = {"id": idx, "metadata": metadata, "document": document}
            matches.append(match)
        return matches

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
