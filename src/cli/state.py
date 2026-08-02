"""CLI runtime state management."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

if TYPE_CHECKING:
    from src.core.llm_gateway import LLMGateway
    from src.services.explanation_service import ExplanationService
    from src.services.preference_service import PreferenceService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = Path(
    os.environ.get("CTM_STATE_PATH", PROJECT_ROOT / "data" / "state.json")
)


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return {}
    return cast(dict[str, Any], value)


def _optional_object(value: object) -> dict[str, Any] | None:
    result = _object(value)
    return result or None


def _history(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [record for item in cast(list[object], value) if (record := _object(item))]


@dataclass
class AppState:
    """Serializable CLI runtime state."""

    problem_description: Optional[str] = None
    last_recommendation: Optional[dict[str, Any]] = None
    last_explanation: Optional[dict[str, Any]] = None
    last_simulation: Optional[dict[str, Any]] = None
    last_comparison: Optional[dict[str, Any]] = None
    context_history: list[dict[str, Any]] = field(default_factory=list)
    llm_gateway: Optional["LLMGateway"] = field(default=None, repr=False, compare=False)
    explanation_service: Optional["ExplanationService"] = field(
        default=None, repr=False, compare=False
    )
    preference_service: Optional["PreferenceService"] = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> "AppState":
        """Load application state from disk."""

        if path.exists():
            try:
                payload = _object(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                payload = {}
        else:
            payload = {}
        problem_description = payload.get("problem_description")
        return cls(
            problem_description=(
                problem_description if isinstance(problem_description, str) else None
            ),
            last_recommendation=_optional_object(payload.get("last_recommendation")),
            last_explanation=_optional_object(payload.get("last_explanation")),
            last_simulation=_optional_object(payload.get("last_simulation")),
            last_comparison=_optional_object(payload.get("last_comparison")),
            context_history=_history(payload.get("context_history")),
        )

    def save(self, path: Path = STATE_PATH) -> None:
        """Persist application state to disk."""

        payload = {
            "problem_description": self.problem_description,
            "last_recommendation": self.last_recommendation,
            "last_explanation": self.last_explanation,
            "last_simulation": self.last_simulation,
            "last_comparison": self.last_comparison,
            "context_history": self.context_history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["AppState", "PROJECT_ROOT", "STATE_PATH"]
