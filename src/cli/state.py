"""CLI runtime state management."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.llm_gateway import LLMGateway
    from src.services.explanation_service import ExplanationService
    from src.services.preference_service import PreferenceService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = Path(os.environ.get("CTM_STATE_PATH", PROJECT_ROOT / "data" / "state.json"))


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
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}
        return cls(
            problem_description=data.get("problem_description"),
            last_recommendation=data.get("last_recommendation"),
            last_explanation=data.get("last_explanation"),
            last_simulation=data.get("last_simulation"),
            last_comparison=data.get("last_comparison"),
            context_history=data.get("context_history", []),
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
