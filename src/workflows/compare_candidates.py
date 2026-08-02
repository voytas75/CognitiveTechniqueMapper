"""Workflow wrapper for candidate comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ..services.comparison_service import ComparisonResult, ComparisonService


def _object_payload(value: object) -> dict[str, Any] | None:
    """Return a JSON-object payload with textual keys, if valid."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return cast(dict[str, Any], value)


def _object_list(value: object) -> list[dict[str, Any]] | None:
    """Return a list of JSON-object payloads, if every item is valid."""

    if not isinstance(value, list):
        return None
    payloads: list[dict[str, Any]] = []
    items = cast(list[object], value)
    for item in items:
        payload = _object_payload(item)
        if payload is None:
            return None
        payloads.append(payload)
    return payloads


def _optional_text(value: object) -> str | None:
    """Return a textual context value or ``None`` for non-text input."""

    return value if isinstance(value, str) else None


@dataclass
class CompareCandidatesWorkflow:
    comparison_service: ComparisonService
    name: str = "compare_candidates"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run the comparison workflow given the shortlist context."""

        recommendation = _object_payload(context.get("recommendation"))
        matches = _object_list(context.get("matches") or [])
        if not recommendation or matches is None:
            raise ValueError("Comparison workflow requires recommendation and matches.")

        result: ComparisonResult = self.comparison_service.compare(
            recommendation,
            matches,
            focus=_optional_text(context.get("focus")),
            preference_summary=_optional_text(context.get("preference_summary")),
        )
        return {"workflow": self.name, "comparison": result.as_dict()}
