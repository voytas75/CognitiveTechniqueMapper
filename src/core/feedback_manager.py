"""Feedback capture structures.

Updates:
    v0.1.0 - 2025-11-09 - Added module docstrings and method documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeedbackEntry:
    """Domain object representing a single feedback entry."""

    workflow: str
    message: str
    rating: int | None = None
    created_at: str | None = None


def _feedback_entry_list() -> list["FeedbackEntry"]:
    return []


@dataclass
class FeedbackManager:
    """Captures user feedback for iterative improvement."""

    entries: list[FeedbackEntry] = field(default_factory=_feedback_entry_list)

    def add(
        self,
        workflow: str,
        message: str,
        rating: int | None = None,
        *,
        created_at: str | None = None,
    ) -> None:
        """Store a new feedback entry.

        Args:
            workflow (str): Workflow identifier associated with the feedback.
            message (str): Feedback text supplied by the user.
            rating (int | None): Optional rating between 1 and 5.
        """

        self.entries.append(
            FeedbackEntry(
                workflow=workflow,
                message=message,
                rating=rating,
                created_at=created_at,
            )
        )

    def latest(self, limit: int = 5) -> list[FeedbackEntry]:
        """Return the newest feedback entries in insertion order.

        Args:
            limit (int): Maximum number of entries to return.

        Returns:
            list[FeedbackEntry]: Feedback entries ordered from oldest to newest within the slice.
        """

        return self.entries[-limit:]
