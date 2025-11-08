from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FeedbackEntry:
    workflow: str
    message: str
    rating: int | None = None


@dataclass
class FeedbackManager:
    """Captures user feedback for iterative improvement."""

    entries: List[FeedbackEntry] = field(default_factory=list)

    def add(self, workflow: str, message: str, rating: int | None = None) -> None:
        self.entries.append(FeedbackEntry(workflow=workflow, message=message, rating=rating))

    def latest(self, limit: int = 5) -> List[FeedbackEntry]:
        return self.entries[-limit:]
