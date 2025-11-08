"""Feedback service orchestration.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings with metadata.
"""

from __future__ import annotations

from typing import Dict, List

from ..core.feedback_manager import FeedbackManager
from ..core.llm_gateway import LLMGateway


class FeedbackService:
    """Handles user feedback and optional LLM-based reflection."""

    def __init__(
        self, feedback_manager: FeedbackManager, llm_gateway: LLMGateway
    ) -> None:
        """Initialize dependencies for feedback processing.

        Args:
            feedback_manager (FeedbackManager): In-memory store for feedback entries.
            llm_gateway (LLMGateway): Gateway for summarization of feedback.
        """

        self._feedback_manager = feedback_manager
        self._llm = llm_gateway

    def record_feedback(
        self, workflow: str, message: str, rating: int | None = None
    ) -> None:
        """Record a feedback entry.

        Args:
            workflow (str): Workflow identifier associated with the feedback.
            message (str): Free-form feedback message.
            rating (int | None): Optional rating in the range 1-5.
        """

        self._feedback_manager.add(workflow=workflow, message=message, rating=rating)

    def summarize_feedback(self) -> Dict[str, str]:
        """Summarize recent feedback entries with the LLM.

        Returns:
            dict[str, str]: Workflow identifier and generated summary text.
        """

        recent = self._feedback_manager.latest()
        if not recent:
            return {"workflow": "feedback_loop", "summary": "No feedback recorded yet."}

        formatted = "\n".join(
            f"- [{item.workflow}] ({item.rating or 'n/a'}): {item.message}"
            for item in recent
        )
        prompt = (
            "Summarize the following feedback entries and highlight any improvement actions:\n"
            f"{formatted}"
        )
        summary = self._llm.invoke("feedback_loop", prompt)
        return {"workflow": "feedback_loop", "summary": summary}
