from __future__ import annotations

from dataclasses import dataclass

from src.services.feedback_service import FeedbackService


@dataclass
class FeedbackWorkflow:
    feedback_service: FeedbackService
    name: str = "feedback_loop"

    def run(self, context: dict) -> dict:
        action = context.get("action", "summarize")
        if action == "record":
            self.feedback_service.record_feedback(
                workflow=context.get("workflow", "detect_technique"),
                message=context["message"],
                rating=context.get("rating"),
            )
            return {"status": "ok"}
        return self.feedback_service.summarize_feedback()
