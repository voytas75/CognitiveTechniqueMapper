from __future__ import annotations

from typing import Any

import src.services.plan_generator as plan_generator_module


class StubGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def invoke(self, workflow: str, prompt: str, **_: Any) -> str:
        self.calls.append((workflow, prompt))
        return "Step 1\nStep 2"


def test_plan_generator_invokes_llm() -> None:
    generator = plan_generator_module.PlanGenerator(llm_gateway=StubGateway())
    result = generator.generate("Technique summary")

    assert result["workflow"] == "summarize_result"
    assert result["plan"] == "Step 1\nStep 2"
