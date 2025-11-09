import importlib
import json
import sys
import types
from typing import Any

import pytest


litellm_stub = types.ModuleType("litellm")
litellm_stub.drop_params = True


def _unused_completion(*_: Any, **__: Any) -> dict[str, Any]:  # pragma: no cover
    raise RuntimeError("litellm completion should not be invoked in tests")


litellm_stub.completion = _unused_completion
sys.modules.setdefault("litellm", litellm_stub)


def import_technique_selector() -> type:
    module_name = "src.services.technique_selector"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name).TechniqueSelector


class StubSQLite:
    def fetch_all(self) -> list[dict[str, Any]]:
        return [
            {
                "id": 1,
                "name": "Decisional Balance",
                "description": "Evaluate pros and cons.",
                "category": "Decision Making",
                "core_principles": "Compare options",
            }
        ]


class StubLLM:
    def invoke(self, workflow: str, prompt: str, **_: Any) -> str:
        assert workflow == "detect_technique"
        assert "Decisional Balance" in prompt
        payload = {
            "suggested_technique": "Decisional Balance",
            "why_it_fits": "Matches dilemmas with competing options.",
            "steps": [
                "List each option.",
                "Score pros and cons.",
                "Compare totals.",
            ],
        }
        return json.dumps(payload)


class StubPromptService:
    def get_prompt(self, name: str) -> str:
        assert name == "detect_technique"
        return "Use provided candidates."


def test_technique_selector_returns_structured_recommendation() -> None:
    TechniqueSelector = import_technique_selector()
    selector = TechniqueSelector(
        sqlite_client=StubSQLite(),
        llm_gateway=StubLLM(),
        prompt_service=StubPromptService(),
        preprocessor=None,
        embedder=None,
        chroma_client=None,
    )

    result = selector.recommend("I need to compare choices.")

    recommendation = result["recommendation"]
    assert recommendation["suggested_technique"] == "Decisional Balance"
    assert "Matches dilemmas" in recommendation["why_it_fits"]
    assert recommendation["steps"] == [
        "List each option.",
        "Score pros and cons.",
        "Compare totals.",
    ]


class StubEmbedder:
    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return [1.0, 0.0, 0.5]


class StubChroma:
    def query(self, *, query_embeddings: list[list[float]], n_results: int) -> dict[str, Any]:
        return {
            "ids": [["technique-1"]],
            "metadatas": [[{"name": "Decisional Balance", "category": "Decision"}]],
            "documents": [["description"]],
            "distances": [[0.5]],
        }


class StubPreferences:
    def preference_summary(self) -> str:
        return "Prefers structure"

    def score_adjustment(self, metadata: dict[str, Any]) -> float:
        return 0.2 if metadata.get("name") == "Decisional Balance" else 0.0


def test_vector_search_uses_chroma(monkeypatch: pytest.MonkeyPatch) -> None:
    TechniqueSelector = import_technique_selector()
    selector = TechniqueSelector(
        sqlite_client=StubSQLite(),
        llm_gateway=StubLLM(),
        prompt_service=StubPromptService(),
        embedder=StubEmbedder(),
        chroma_client=StubChroma(),
    )

    matches = selector._vector_search("normalized", [0.1, 0.2])
    assert matches[0]["score"] == pytest.approx(1 / (1 + 0.5))


def test_vector_search_with_embeddings_scores_results(monkeypatch: pytest.MonkeyPatch) -> None:
    TechniqueSelector = import_technique_selector()

    class DeterministicEmbedder(StubEmbedder):
        def embed(self, text: str) -> list[float]:
            if "pros" in text:
                return [0.5, 0.5]
            return [1.0, 1.0]

    sqlite_stub = StubSQLite()
    selector = TechniqueSelector(
        sqlite_client=sqlite_stub,
        llm_gateway=StubLLM(),
        prompt_service=StubPromptService(),
        embedder=DeterministicEmbedder(),
        chroma_client=None,
    )

    matches = selector._vector_search("normalized", [1.0, 1.0])
    assert matches[0]["score"] is not None


def test_apply_preference_adjustments_reorders_matches() -> None:
    TechniqueSelector = import_technique_selector()
    selector = TechniqueSelector(
        sqlite_client=StubSQLite(),
        llm_gateway=StubLLM(),
        prompt_service=StubPromptService(),
        preference_service=StubPreferences(),
    )

    candidates = [
        {"metadata": {"name": "Other"}, "score": 0.1},
        {"metadata": {"name": "Decisional Balance"}, "score": 0.2},
    ]

    adjusted = selector._apply_preference_adjustments(candidates)
    assert adjusted[0]["metadata"]["name"] == "Decisional Balance"
    assert adjusted[0]["preference_adjustment"] == pytest.approx(0.2)


def test_llm_reasoning_handles_empty_candidates() -> None:
    TechniqueSelector = import_technique_selector()
    selector = TechniqueSelector(
        sqlite_client=StubSQLite(),
        llm_gateway=StubLLM(),
        prompt_service=StubPromptService(),
    )

    result = selector._llm_reason_about_candidates("normalized", [])
    assert result["recommendation"]["suggested_technique"] is None


def test_parse_json_response_handles_markdown() -> None:
    TechniqueSelector = import_technique_selector()
    selector = TechniqueSelector(
        sqlite_client=StubSQLite(),
        llm_gateway=StubLLM(),
        prompt_service=StubPromptService(),
    )

    response = "```json\n{\"suggested_technique\": \"Technique\"}\n```"
    parsed = selector._parse_json_response(response)
    assert parsed["suggested_technique"] == "Technique"
    assert selector._parse_json_response("not json") is None


def test_coerce_helpers() -> None:
    TechniqueSelector = import_technique_selector()
    selector = TechniqueSelector(
        sqlite_client=StubSQLite(),
        llm_gateway=StubLLM(),
        prompt_service=StubPromptService(),
    )

    assert selector._coerce_string(123) == "123"
    assert selector._coerce_steps("a\nb") == ["a", "b"]
    assert selector._coerce_steps(["one", "two"]) == ["one", "two"]
    assert selector._coerce_float("0.5") == 0.5
    assert selector._cosine_similarity([], []) == 0.0
