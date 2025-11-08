"""Plan generation utilities.

Updates:
    v0.1.0 - 2025-11-09 - Added module docstring and detailed method documentation.
"""

from __future__ import annotations

from typing import Dict, List

from ..core.llm_gateway import LLMGateway


class PlanGenerator:
    """Uses an LLM workflow to create step-by-step action plans."""

    def __init__(self, llm_gateway: LLMGateway) -> None:
        """Store the LLM gateway dependency.

        Args:
            llm_gateway (LLMGateway): Gateway used to invoke summarize workflow.
        """

        self._llm = llm_gateway

    def generate(self, technique_summary: str) -> Dict[str, List[str] | str]:
        """Generate a plan based on the supplied technique summary.

        Args:
            technique_summary (str): Summary description of the selected technique.

        Returns:
            dict[str, list[str] | str]: Workflow name and generated plan.

        Raises:
            RuntimeError: Propagated if the LLM provider call fails.
        """

        prompt = (
            "Given the selected cognitive technique, produce a concise implementation plan.\n"
            f"Technique summary:\n{technique_summary}\n"
            "Respond with bullet steps that a user can follow."
        )
        response = self._llm.invoke("summarize_result", prompt)
        return {"workflow": "summarize_result", "plan": response}
