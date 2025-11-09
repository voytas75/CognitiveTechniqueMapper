"""Feedback service orchestration.

Updates:
    v0.1.0 - 2025-11-09 - Added module and method docstrings with metadata.
    v0.2.0 - 2025-11-09 - Persist feedback to SQLite and preload history.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..core.feedback_manager import FeedbackManager
from ..core.llm_gateway import LLMGateway
from ..db.feedback_repository import FeedbackRepository
from .preference_service import PreferenceService


class FeedbackService:
    """Handles user feedback and optional LLM-based reflection."""

    def __init__(
        self,
        feedback_manager: FeedbackManager,
        llm_gateway: LLMGateway,
        repository: FeedbackRepository,
        *,
        history_limit: int = 25,
        preference_service: PreferenceService | None = None,
    ) -> None:
        """Initialize dependencies for feedback processing.

        Args:
            feedback_manager (FeedbackManager): In-memory store for feedback entries.
            llm_gateway (LLMGateway): Gateway for summarization of feedback.
            repository (FeedbackRepository): Persistent storage for feedback records.
            history_limit (int): Number of recent entries to cache in memory.
            preference_service (PreferenceService | None): Optional preference sink.
        """

        self._feedback_manager = feedback_manager
        self._llm = llm_gateway
        self._repository = repository
        self._history_limit = history_limit
        self._preferences = preference_service
        self._bootstrap_manager()

    def record_feedback(
        self,
        workflow: str,
        message: str,
        rating: int | None = None,
        *,
        technique: str | None = None,
        category: str | None = None,
    ) -> None:
        """Record a feedback entry.

        Args:
            workflow (str): Workflow identifier associated with the feedback.
            message (str): Free-form feedback message.
            rating (int | None): Optional rating in the range 1-5.
            technique (str | None): Technique referenced by the feedback.
            category (str | None): Technique category context.
        """

        timestamp = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        self._repository.insert(
            workflow=workflow,
            message=message,
            rating=rating,
            created_at=timestamp,
        )
        self._feedback_manager.add(
            workflow=workflow,
            message=message,
            rating=rating,
            created_at=timestamp,
        )
        if self._preferences:
            self._preferences.record_preference(
                technique=technique,
                category=category,
                rating=rating,
                notes=message,
            )

    def summarize_feedback(self) -> dict[str, str]:
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

    def _bootstrap_manager(self) -> None:
        records = self._repository.fetch_recent(limit=self._history_limit)
        for record in reversed(records):
            self._feedback_manager.add(
                workflow=record.get("workflow", ""),
                message=record.get("message", ""),
                rating=record.get("rating"),
                created_at=record.get("created_at"),
            )
