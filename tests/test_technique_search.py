from typing import Any

from src.services.technique_search import (
    TechniqueSearchMode,
    TechniqueSearchService,
)
from src.services.technique_utils import compose_embedding_text

ENTRIES = [
    {
        "id": "1",
        "name": "Decision Matrix",
        "description": "Compare options with weighted criteria.",
        "category": "Decision",
        "core_principles": "Evaluate trade-offs.",
    },
    {
        "id": "2",
        "name": "Brainstorming",
        "description": "Generate ideas without judgment.",
        "category": "Creativity",
        "core_principles": "Expand alternatives.",
    },
]


class StubSQLite:
    def fetch_all(self) -> list[dict[str, Any]]:
        return ENTRIES


class StubPreprocessor:
    def normalize(self, text: str) -> str:
        return text.strip().lower()


class StubEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.0, 1.0] if "brainstorm" in text.lower() else [1.0, 0.0]


class StubChroma:
    def query(self, **_: Any) -> dict[str, list[list[Any]]]:
        return {
            "ids": [["decision-matrix"]],
            "metadatas": [[{"name": "Decision Matrix", "category": "Decision"}]],
            "documents": [["Compare options with weighted criteria."]],
            "distances": [[0.0]],
        }


def build_service(
    *, embedder: StubEmbedder | None = None, chroma: StubChroma | None = None
) -> TechniqueSearchService:
    return TechniqueSearchService(
        sqlite_client=StubSQLite(),  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        chroma_client=chroma,  # type: ignore[arg-type]
        preprocessor=StubPreprocessor(),  # type: ignore[arg-type]
    )


def test_search_returns_no_results_for_nonpositive_limit() -> None:
    service = build_service()

    assert service.search("decision", mode=TechniqueSearchMode.KEYWORD, limit=0) == []
    assert service.search("decision", mode=TechniqueSearchMode.FUZZY, limit=-1) == []


def test_keyword_and_fuzzy_modes_rank_matching_technique() -> None:
    service = build_service()

    keyword = service.search("decision options", mode="keyword")
    fuzzy = service.search("decision matrix", mode=TechniqueSearchMode.FUZZY)
    fallback = service.search("decision options", mode="unknown")

    assert keyword[0].name == "Decision Matrix"
    assert keyword[0].breakdown["keyword"] == 1.0
    assert "Matched terms: decision, options" in keyword[0].highlights
    assert fuzzy[0].name == "Decision Matrix"
    assert fallback[0].name == "Decision Matrix"


def test_local_semantic_mode_reuses_technique_embeddings() -> None:
    embedder = StubEmbedder()
    service = build_service(embedder=embedder)

    first = service.search("decision", mode="semantic", limit=2)
    second = service.search("decision", mode="semantic", limit=2)

    assert first[0].name == "Decision Matrix"
    assert second[0].as_dict()["breakdown"] == {"semantic": 1.0}
    assert all(
        embedder.calls.count(compose_embedding_text(entry)) == 1 for entry in ENTRIES
    )


def test_hybrid_chroma_result_blends_and_deduplicates_evidence() -> None:
    service = build_service(embedder=StubEmbedder(), chroma=StubChroma())

    result = service.search("decision options", mode="hybrid", limit=1)[0]

    assert result.name == "Decision Matrix"
    assert result.breakdown == {"semantic": 1.0, "keyword": 1.0}
    assert result.metadata["description"] == "Compare options with weighted criteria."
    assert result.highlights.count("Matched terms: decision, options") == 1
