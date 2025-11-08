from __future__ import annotations

from typing import Dict, List

from ..core.feedback_manager import FeedbackManager
from ..core.llm_gateway import LLMGateway


class FeedbackService:
    """Handles user feedback and optional LLM-based reflection."""

    def __init__(self, feedback_manager: FeedbackManager, llm_gateway: LLMGateway) -> None:
        self._feedback_manager = feedback_manager
        self._llm = llm_gateway

    def record_feedback(self, workflow: str, message: str, rating: int | None = None) -> None:
        self._feedback_manager.add(workflow=workflow, message=message, rating=rating)

    def summarize_feedback(self) -> Dict[str, str]:
        recent = self._feedback_manager.latest()
        if not recent:
            return {"workflow": "feedback_loop", "summary": "No feedback recorded yet."}

        formatted = "\n".join(
            f"- [{item.workflow}] ({item.rating or 'n/a'}): {item.message}" for item in recent
        )
        prompt = (
            "Summarize the following feedback entries and highlight any improvement actions:\n"
            f"{formatted}"
        )
        summary = self._llm.invoke("feedback_loop", prompt)
        return {"workflow": "feedback_loop", "summary": summary}
