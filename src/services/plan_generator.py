from __future__ import annotations

from typing import Dict, List

from src.core.llm_gateway import LLMGateway


class PlanGenerator:
    """Uses an LLM workflow to create step-by-step action plans."""

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self._llm = llm_gateway

    def generate(self, technique_summary: str) -> Dict[str, List[str] | str]:
        prompt = (
            "Given the selected cognitive technique, produce a concise implementation plan.\n"
            f"Technique summary:\n{technique_summary}\n"
            "Respond with bullet steps that a user can follow."
        )
        response = self._llm.invoke("summarize_result", prompt)
        return {"workflow": "summarize_result", "plan": response}
