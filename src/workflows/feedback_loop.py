"""Feedback workflow dispatcher.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..services.feedback_service import FeedbackService


def _optional_text(value: object) -> str | None:
    """Return a textual context value or ``None`` for non-text input."""

    return value if isinstance(value, str) else None


def _optional_rating(value: object) -> int | None:
    """Return an integer rating or ``None`` for non-rating input."""

    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass
class FeedbackWorkflow:
    feedback_service: FeedbackService
    name: str = "feedback_loop"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Dispatch actions to the feedback service.

        Args:
            context (dict[str, Any]): Workflow context containing action and payload.

        Returns:
            dict[str, Any]: Result from recording or summarizing feedback.
        """

        action = _optional_text(context.get("action")) or "summarize"
        if action == "record":
            message = _optional_text(context.get("message"))
            if not message:
                raise ValueError("Feedback recording requires a text message.")
            self.feedback_service.record_feedback(
                workflow=_optional_text(context.get("workflow")) or "detect_technique",
                message=message,
                rating=_optional_rating(context.get("rating")),
                technique=_optional_text(context.get("technique")),
                category=_optional_text(context.get("category")),
            )
            return {"status": "ok"}
        return self.feedback_service.summarize_feedback()
