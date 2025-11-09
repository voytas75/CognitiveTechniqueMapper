import json
import sys
import types
from typing import Any, Dict


litellm_stub = types.ModuleType("litellm")
litellm_stub.drop_params = True


def _unused_completion(*_: Any, **__: Any) -> Dict[str, Any]:  # pragma: no cover
    raise RuntimeError("litellm completion should not be invoked in tests")


litellm_stub.completion = _unused_completion
sys.modules.setdefault("litellm", litellm_stub)

from src.services.explanation_service import ExplanationService


class StubLLM:
    def invoke(self, workflow: str, prompt: str, **_: Any) -> str:
        assert workflow == "explain_logic"
        assert "Recommendation payload" in prompt
        payload = {
            "overview": "Technique fits the decision scenario.",
            "key_factors": ["Uses structured comparison."],
            "risks": ["May overlook emotional factors."],
            "next_steps": ["Run through the pros/cons list."],
        }
        return json.dumps(payload)


class StubPrompts:
    def get_prompt(self, name: str) -> str:
        assert name == "explain_logic"
        return "Explain the logic."  # content is irrelevant for the stub


def test_explanation_service_returns_structured_result() -> None:
    service = ExplanationService(  # type: ignore[arg-type]
        llm_gateway=StubLLM(), prompt_service=StubPrompts()
    )

    recommendation = {"suggested_technique": "Decisional Balance"}
    result = service.explain(recommendation=recommendation)

    assert result.overview == "Technique fits the decision scenario."
    assert result.key_factors == ["Uses structured comparison."]
    assert result.risks == ["May overlook emotional factors."]
    assert result.next_steps == ["Run through the pros/cons list."]
