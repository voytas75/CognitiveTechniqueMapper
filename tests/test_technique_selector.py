import json
import sys
import types
from typing import Any, Dict, List


litellm_stub = types.ModuleType("litellm")
litellm_stub.drop_params = True


def _unused_completion(*_: Any, **__: Any) -> Dict[str, Any]:  # pragma: no cover
    raise RuntimeError("litellm completion should not be invoked in tests")


litellm_stub.completion = _unused_completion
sys.modules.setdefault("litellm", litellm_stub)


from src.services.technique_selector import TechniqueSelector


class StubSQLite:
    def fetch_all(self) -> List[Dict[str, Any]]:
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
