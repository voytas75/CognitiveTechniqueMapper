"""Feedback workflow dispatcher.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..services.feedback_service import FeedbackService


@dataclass
class FeedbackWorkflow:
    feedback_service: FeedbackService
    name: str = "feedback_loop"

    def run(self, context: dict) -> dict:
        """Dispatch actions to the feedback service.

        Args:
            context (dict): Workflow context containing action and payload.

        Returns:
            dict: Result from recording or summarizing feedback.
        """

        action = context.get("action", "summarize")
        if action == "record":
            self.feedback_service.record_feedback(
                workflow=context.get("workflow", "detect_technique"),
                message=context["message"],
                rating=context.get("rating"),
            )
            return {"status": "ok"}
        return self.feedback_service.summarize_feedback()
