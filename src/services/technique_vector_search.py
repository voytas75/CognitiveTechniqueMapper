"""Vector candidate retrieval for technique selection."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple, cast

from ..db.sqlite_client import SQLiteClient
from .embedding_gateway import EmbeddingGateway
from .technique_utils import compose_embedding_text


class ChromaSearchClient(Protocol):
    """Minimal Chroma query surface required for candidate retrieval."""

    def query(
        self, query_embeddings: Sequence[Sequence[float]], n_results: int = 5
    ) -> Mapping[str, Sequence[Sequence[object]]]: ...


class TechniqueVectorSearch:
    """Retrieve and score candidate techniques using vector embeddings."""

    def __init__(
        self,
        sqlite_client: SQLiteClient,
        embedder: EmbeddingGateway | None = None,
        chroma_client: ChromaSearchClient | None = None,
    ) -> None:
        self._sqlite = sqlite_client
        self._embedder = embedder
        self._chroma = chroma_client
        self._embedding_cache: Dict[str, Tuple[str, List[float]]] = {}

    def generate_query_embedding(self, normalized_text: str) -> List[float] | None:
        """Generate an embedding when an embedding gateway is configured."""
        if self._embedder is None:
            return None
        return self._embedder.embed(normalized_text)

    def search(
        self, normalized_text: str, query_embedding: List[float] | None
    ) -> List[Dict[str, Any]]:
        """Return Chroma candidates or score SQLite rows as a fallback."""
        if self._chroma is not None and query_embedding is not None:
            return self._search_chroma(query_embedding)

        stored = [dict(row) for row in self._sqlite.fetch_all()]
        if not stored:
            return []
        if query_embedding is None or self._embedder is None:
            return stored[:5]

        scored_matches: List[Dict[str, Any]] = []
        for item in stored:
            technique_text = compose_embedding_text(item)
            technique_embedding = self._get_cached_embedding(item, technique_text)
            score = self.cosine_similarity(query_embedding, technique_embedding)
            scored_matches.append(
                {
                    "id": item.get("id"),
                    "metadata": item,
                    "document": item.get("description", ""),
                    "score": score,
                }
            )
        scored_matches.sort(
            key=lambda entry: _coerce_float(entry.get("score")) or 0.0,
            reverse=True,
        )
        return scored_matches[:5]

    def _search_chroma(self, query_embedding: Sequence[float]) -> List[Dict[str, Any]]:
        chroma = self._chroma
        if chroma is None:
            return []
        results = chroma.query(query_embeddings=[query_embedding], n_results=5)
        ids = _first_row(results.get("ids"))
        metadatas = _first_row(results.get("metadatas"))
        documents = _first_row(results.get("documents"))
        distances = _first_row(results.get("distances") or results.get("scores"))
        matches: List[Dict[str, Any]] = []
        for identifier, metadata, document, distance in zip(
            ids, metadatas, documents, distances
        ):
            matches.append(
                {
                    "id": identifier,
                    "metadata": metadata,
                    "document": document,
                    "distance": distance,
                    "score": _distance_similarity(distance),
                }
            )
        return matches

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity with zero-vector safeguards."""
        if not vec_a or not vec_b:
            return 0.0
        length = min(len(vec_a), len(vec_b))
        dot = sum(vec_a[index] * vec_b[index] for index in range(length))
        norm_a = sum(value * value for value in vec_a[:length]) ** 0.5
        norm_b = sum(value * value for value in vec_b[:length]) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _get_cached_embedding(
        self, item: Dict[str, Any], technique_text: str
    ) -> List[float]:
        if self._embedder is None:
            raise RuntimeError("Embedding gateway is unavailable for technique cache.")
        key = _cache_key(item)
        cached = self._embedding_cache.get(key)
        if cached is not None and cached[0] == technique_text:
            return cached[1]
        vector = self._embedder.embed(technique_text)
        self._embedding_cache[key] = (technique_text, vector)
        return vector

    def clear_embedding_cache(self) -> None:
        """Clear cached technique embeddings after a dataset refresh."""
        self._embedding_cache.clear()


def _first_row(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return []
    row = value[0]
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        return []
    return cast(Sequence[object], row)


def _distance_similarity(value: object) -> float | None:
    distance = _coerce_float(value)
    return None if distance is None else 1 / (1 + distance)


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _cache_key(item: Mapping[str, object]) -> str:
    identifier = item.get("id")
    return str(identifier) if identifier is not None else str(item.get("name", ""))
