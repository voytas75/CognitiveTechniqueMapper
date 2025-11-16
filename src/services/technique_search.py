"""Technique search service supporting multiple retrieval modes.

Updates:
    v0.3.0 - 2025-11-10 - Added multi-mode search capabilities for CLI usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.core.preprocessor import ProblemPreprocessor
from src.db.sqlite_client import SQLiteClient
from src.services.embedding_gateway import EmbeddingGateway
from src.services.technique_utils import compose_embedding_text

try:  # pragma: no cover - optional dependency path
    from src.db.chroma_client import ChromaClient
except RuntimeError:  # pragma: no cover - optional dependency path
    ChromaClient = None  # type: ignore


class TechniqueSearchMode(StrEnum):
    """Supported search strategies."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FUZZY = "fuzzy"


@dataclass(slots=True)
class TechniqueSearchResult:
    """Container for ranked technique matches."""

    metadata: dict[str, Any]
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Return the technique name for convenience."""

        return (self.metadata.get("name") or "").strip()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the match."""

        return {
            "name": self.name,
            "score": self.score,
            "breakdown": dict(self.breakdown),
            "highlights": list(self.highlights),
            "metadata": dict(self.metadata),
        }


class TechniqueSearchService:
    """Search techniques using semantic, keyword, fuzzy, or hybrid strategies."""

    def __init__(
        self,
        *,
        sqlite_client: SQLiteClient,
        embedder: EmbeddingGateway | None = None,
        chroma_client: ChromaClient | None = None,
        preprocessor: ProblemPreprocessor | None = None,
    ) -> None:
        """Initialize the service with storage and optional vector search dependencies.

        Args:
            sqlite_client (SQLiteClient): Database accessor for techniques.
            embedder (EmbeddingGateway | None): Embedding provider for semantic search.
            chroma_client (ChromaClient | None): Vector database client for similarity.
            preprocessor (ProblemPreprocessor | None): Normalizer for search text.
        """

        self._sqlite = sqlite_client
        self._embedder = embedder
        self._chroma = chroma_client
        self._preprocessor = preprocessor or ProblemPreprocessor()
        self._embedding_cache: dict[str, list[float]] = {}

    def search(
        self,
        text: str,
        *,
        mode: TechniqueSearchMode | str = TechniqueSearchMode.HYBRID,
        limit: int = 5,
    ) -> list[TechniqueSearchResult]:
        """Search for techniques matching the supplied text.

        Args:
            text (str): Freeform text describing the desired technique context.
            mode (TechniqueSearchMode | str): Retrieval strategy to apply.
            limit (int): Maximum number of matches to return.

        Returns:
            list[TechniqueSearchResult]: Ranked technique matches.
        """

        normalized = self._preprocessor.normalize(text)
        if not normalized:
            return []

        resolved_mode = self._coerce_mode(mode)
        if resolved_mode is TechniqueSearchMode.SEMANTIC:
            return self._semantic_rank(normalized, limit)
        if resolved_mode is TechniqueSearchMode.KEYWORD:
            return self._keyword_rank(normalized, limit)
        if resolved_mode is TechniqueSearchMode.FUZZY:
            return self._fuzzy_rank(normalized, limit)
        return self._hybrid_rank(normalized, limit)

    def _semantic_rank(self, normalized_text: str, limit: int) -> list[TechniqueSearchResult]:
        entries = self._fetch_entries()
        if not entries or not self._embedder:
            return self._keyword_rank(normalized_text, limit)

        query_embedding = self._embedder.embed(normalized_text)
        if not isinstance(query_embedding, Sequence):
            return []

        if self._chroma and query_embedding:
            return self._semantic_rank_chroma(query_embedding, limit)
        return self._semantic_rank_local(entries, query_embedding, limit)

    def _semantic_rank_chroma(
        self, query_embedding: Sequence[float], limit: int
    ) -> list[TechniqueSearchResult]:
        results = self._chroma.query(
            query_embeddings=[query_embedding],
            n_results=limit,
        )
        matches: list[TechniqueSearchResult] = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances") or results.get("scores") or [[]]
        distance_row = distances[0] if distances else []
        for identifier, metadata, document, distance in zip(
            ids, metadatas, documents, distance_row
        ):
            metadata = metadata or {"name": identifier}
            if document and not metadata.get("description"):
                metadata["description"] = document
            score = self._distance_to_similarity(distance)
            matches.append(
                TechniqueSearchResult(
                    metadata=metadata,
                    score=score,
                    breakdown={"semantic": score},
                    highlights=[
                        f"Chroma similarity: {score:.3f}" if score else "Chroma match"
                    ],
                )
            )
        return matches

    def _semantic_rank_local(
        self,
        entries: list[dict[str, Any]],
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[TechniqueSearchResult]:
        scored: list[TechniqueSearchResult] = []
        for entry in entries:
            key = self._entry_key(entry)
            technique_text = compose_embedding_text(entry)
            embedding = self._embedding_cache.get(key)
            if embedding is None and self._embedder:
                embedding = self._embedder.embed(technique_text)
                self._embedding_cache[key] = list(embedding)
            if not embedding:
                continue
            score = self._cosine_similarity(query_embedding, embedding)
            scored.append(
                TechniqueSearchResult(
                    metadata=entry,
                    score=score,
                    breakdown={"semantic": score},
                    highlights=[f"Cosine similarity: {score:.3f}"],
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _keyword_rank(self, normalized_text: str, limit: int) -> list[TechniqueSearchResult]:
        terms = [term for term in normalized_text.split(" ") if term]
        if not terms:
            return []

        entries = self._fetch_entries()
        matches: list[TechniqueSearchResult] = []
        for entry in entries:
            fields = {
                "name": (entry.get("name") or "").lower(),
                "category": (entry.get("category") or "").lower(),
                "description": (entry.get("description") or "").lower(),
                "principles": (entry.get("core_principles") or "").lower(),
            }
            matched_terms: set[str] = set()
            field_hits: dict[str, int] = {label: 0 for label in fields}
            for term in terms:
                for label, content in fields.items():
                    if term and term in content:
                        matched_terms.add(term)
                        field_hits[label] += 1
            if not matched_terms:
                continue
            score = len(matched_terms) / len(terms)
            highlights = [
                "Matched terms: " + ", ".join(sorted(matched_terms)),
            ]
            top_fields = [
                f"{label}: {hits} hits"
                for label, hits in field_hits.items()
                if hits > 0
            ]
            if top_fields:
                highlights.append("Field coverage: " + "; ".join(top_fields))
            matches.append(
                TechniqueSearchResult(
                    metadata=entry,
                    score=score,
                    breakdown={"keyword": score},
                    highlights=highlights,
                )
            )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def _fuzzy_rank(self, normalized_text: str, limit: int) -> list[TechniqueSearchResult]:
        from difflib import SequenceMatcher

        entries = self._fetch_entries()
        scored: list[TechniqueSearchResult] = []
        for entry in entries:
            combined = " ".join(
                filter(
                    None,
                    [
                        entry.get("name"),
                        entry.get("description"),
                        entry.get("core_principles"),
                    ],
                )
            ).lower()
            if not combined:
                continue
            ratio = SequenceMatcher(None, normalized_text, combined).ratio()
            if ratio <= 0:
                continue
            scored.append(
                TechniqueSearchResult(
                    metadata=entry,
                    score=ratio,
                    breakdown={"fuzzy": ratio},
                    highlights=[f"Fuzzy ratio: {ratio:.3f}"],
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _hybrid_rank(self, normalized_text: str, limit: int) -> list[TechniqueSearchResult]:
        if not self._embedder and not self._chroma:
            return self._keyword_rank(normalized_text, limit)
        semantic = self._semantic_rank(normalized_text, limit)
        keyword = self._keyword_rank(normalized_text, limit)
        index = {self._entry_key(item.metadata): item for item in semantic}
        keyword_index = {
            key: value
            for key, value in {
                self._entry_key(item.metadata): item for item in keyword
            }.items()
        }
        combined: dict[str, TechniqueSearchResult] = {}
        for key in set(index) | set(keyword_index):
            semantic_result = index.get(key)
            keyword_result = keyword_index.get(key)
            if semantic_result is None and keyword_result is None:
                continue
            metadata = dict(
                (semantic_result or keyword_result).metadata  # type: ignore[union-attr]
            )
            breakdown: dict[str, float] = {}
            highlights: list[str] = []
            if semantic_result:
                breakdown.update(semantic_result.breakdown)
                highlights.extend(semantic_result.highlights)
            if keyword_result:
                for label, value in keyword_result.breakdown.items():
                    breakdown[label] = value
                highlights.extend(keyword_result.highlights)
            score = self._blend_scores(breakdown.values())
            combined[key] = TechniqueSearchResult(
                metadata=metadata,
                score=score,
                breakdown=breakdown,
                highlights=self._deduplicate_highlights(highlights),
            )
        ranked = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _fetch_entries(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._sqlite.fetch_all()]

    def _entry_key(self, entry: Mapping[str, Any]) -> str:
        identifier = entry.get("name") or entry.get("id") or ""
        return str(identifier).lower()

    def _distance_to_similarity(self, distance: Any) -> float:
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0
        return 1 / (1 + value) if value >= 0 else 0.0

    def _cosine_similarity(
        self, reference: Sequence[float], candidate: Sequence[float]
    ) -> float:
        if not reference or not candidate:
            return 0.0
        length = min(len(reference), len(candidate))
        dot = sum(reference[i] * candidate[i] for i in range(length))
        norm_a = sum(value * value for value in reference[:length]) ** 0.5
        norm_b = sum(value * value for value in candidate[:length]) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _blend_scores(self, values: Iterable[float]) -> float:
        scores = [value for value in values if isinstance(value, (int, float))]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _deduplicate_highlights(self, highlights: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in highlights:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def _coerce_mode(self, mode: TechniqueSearchMode | str) -> TechniqueSearchMode:
        if isinstance(mode, TechniqueSearchMode):
            return mode
        normalized = str(mode).lower().strip()
        for candidate in TechniqueSearchMode:
            if candidate.value == normalized:
                return candidate
        return TechniqueSearchMode.HYBRID


__all__ = [
    "TechniqueSearchMode",
    "TechniqueSearchResult",
    "TechniqueSearchService",
]
