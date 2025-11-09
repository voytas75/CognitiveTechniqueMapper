import importlib
import json
import sys
import types
from typing import Any, List

litellm_stub = types.ModuleType("litellm")
litellm_stub.drop_params = True


def _unused_completion(*_: Any, **__: Any) -> str:  # pragma: no cover
    raise RuntimeError("litellm completion should not be invoked in tests")


litellm_stub.completion = _unused_completion
sys.modules.setdefault("litellm", litellm_stub)


def import_comparison_service() -> type:
    module_name = "src.services.comparison_service"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name).ComparisonService


class StubLLM:
    def __init__(self) -> None:
        self.workflow: str | None = None
        self.prompt: str | None = None
        self.response: str | None = None

    def invoke(self, workflow: str, prompt: str, **_: Any) -> str:
        self.workflow = workflow
        self.prompt = prompt
        if self.response is not None:
            return self.response
        payload = {
            "current_recommendation": "Decisional Balance",
            "best_alternative": "Six Thinking Hats",
            "comparison_points": [
                {
                    "technique": "Decisional Balance",
                    "strengths": "Great for trade-offs",
                    "risks": "May feel analytical",
                    "best_for": "Binary decisions",
                }
            ],
            "decision_guidance": ["Use hats when creativity is needed."],
            "confidence_notes": "High confidence.",
        }
        return json.dumps(payload)


class StubPromptService:
    def get_prompt(self, name: str) -> str:
        assert name == "compare_candidates"
        return "Compare techniques."


def test_comparison_service_returns_structured_payload() -> None:
    ComparisonService = import_comparison_service()
    llm = StubLLM()
    prompts = StubPromptService()
    service = ComparisonService(llm_gateway=llm, prompt_service=prompts)

    recommendation = {"suggested_technique": "Decisional Balance"}
    matches: List[dict[str, Any]] = [
        {"metadata": {"name": "Decisional Balance", "category": "Decision Making"}}
    ]
    result = service.compare(
        recommendation,
        matches,
        focus="Decisional Balance",
        preference_summary="Prefers structured analysis.",
    )

    assert llm.workflow == "compare_candidates"
    assert "structured analysis" in llm.prompt.casefold()
    assert result.current_recommendation == "Decisional Balance"
    assert result.best_alternative == "Six Thinking Hats"
    assert result.comparison_points[0]["technique"] == "Decisional Balance"
    assert result.decision_guidance == ["Use hats when creativity is needed."]


def test_comparison_service_parses_markdown_payload() -> None:
    ComparisonService = import_comparison_service()
    llm = StubLLM()
    prompts = StubPromptService()
    service = ComparisonService(llm_gateway=llm, prompt_service=prompts)

    markdown_response = "```json\n{" "\"best_alternative\": \"Six Thinking Hats\"" "}\n```"
    llm.response = markdown_response

    recommendation = {"suggested_technique": "Decisional Balance"}
    matches: List[dict[str, Any]] = [
        {"metadata": {"name": "Decisional Balance", "category": "Decision Making"}}
    ]
    result = service.compare(recommendation, matches)

    assert result.best_alternative == "Six Thinking Hats"


def test_comparison_service_handles_invalid_json() -> None:
    ComparisonService = import_comparison_service()
    llm = StubLLM()
    prompts = StubPromptService()
    service = ComparisonService(llm_gateway=llm, prompt_service=prompts)

    llm.response = "not-json"
    recommendation = {"suggested_technique": "Decisional Balance"}
    matches: List[dict[str, Any]] = []
    result = service.compare(recommendation, matches)

    assert result.current_recommendation is None
    assert result.best_alternative is None
