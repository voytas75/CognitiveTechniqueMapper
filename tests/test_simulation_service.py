import json
import sys
import types
from typing import Any

litellm_stub = types.ModuleType("litellm")
litellm_stub.drop_params = True


def _unused_completion(*_: Any, **__: Any) -> str:  # pragma: no cover
    raise RuntimeError("litellm completion should not be invoked in tests")


litellm_stub.completion = _unused_completion
sys.modules.setdefault("litellm", litellm_stub)

from src.services.simulation_service import SimulationService


class StubLLM:
    def __init__(self) -> None:
        self.workflow: str | None = None
        self.prompt: str | None = None

    def invoke(self, workflow: str, prompt: str, **_: Any) -> str:
        self.workflow = workflow
        self.prompt = prompt
        payload = {
            "simulation_overview": "Walkthrough",
            "scenario_variations": [
                {"name": "Best case", "outcome": "Success", "guidance": "Stay on plan"}
            ],
            "cautions": ["Watch timing"],
            "recommended_follow_up": ["Review outcomes"],
        }
        return json.dumps(payload)


class StubPromptService:
    def get_prompt(self, name: str) -> str:
        assert name == "simulate_technique"
        return "Simulate technique."


def test_simulation_service_returns_structured_payload() -> None:
    llm = StubLLM()
    prompts = StubPromptService()
    service = SimulationService(llm_gateway=llm, prompt_service=prompts)

    recommendation = {
        "suggested_technique": "Decisional Balance",
        "why_it_fits": "Balances pros and cons.",
        "steps": ["Step 1"],
    }
    result = service.simulate(
        recommendation,
        problem_description="Need to decide between options.",
        scenario="Switching jobs",
        preference_summary="Prefers decision frameworks.",
    )

    assert llm.workflow == "simulate_technique"
    assert "Switching jobs" in llm.prompt
    assert result.simulation_overview == "Walkthrough"
    assert result.scenario_variations[0]["name"] == "Best case"
    assert result.cautions == ["Watch timing"]
    assert result.recommended_follow_up == ["Review outcomes"]
