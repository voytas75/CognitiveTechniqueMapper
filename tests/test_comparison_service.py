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

from src.services.comparison_service import ComparisonService


class StubLLM:
    def __init__(self) -> None:
        self.workflow: str | None = None
        self.prompt: str | None = None

    def invoke(self, workflow: str, prompt: str, **_: Any) -> str:
        self.workflow = workflow
        self.prompt = prompt
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
