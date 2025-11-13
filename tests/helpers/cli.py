"""Shared test doubles and utilities for src.cli tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

import src.cli as cli

Handler = Callable[[Dict[str, Any], "RecordingOrchestrator"], Dict[str, Any]]
DefaultHandler = Callable[[str, Dict[str, Any], "RecordingOrchestrator"], Dict[str, Any]]


@dataclass
class PreferenceProfile:
    """Simple preference profile stand-in for tests."""

    summary: str


class StubPreferenceService:
    """Preference service fixture with deterministic behaviour."""

    def __init__(
        self,
        *,
        summary: str = "Prefers structured analysis.",
        adjustment: float = 0.1,
        repository: Any | None = None,
    ) -> None:
        self._summary = summary
        self._adjustment = adjustment
        self.repository = repository
        self.recorded: list[dict[str, Any]] = []
        self.cleared = False

    def preference_summary(self) -> str:
        return self._summary

    def export_profile(self) -> PreferenceProfile:
        return PreferenceProfile(summary=self._summary)

    def record_preference(
        self,
        *,
        technique: str | None,
        category: str | None,
        rating: int | None,
        notes: str,
    ) -> None:
        self.recorded.append(
            {
                "technique": technique,
                "category": category,
                "rating": rating,
                "notes": notes,
            }
        )

    def score_adjustment(self, _: Mapping[str, Any]) -> float:
        return self._adjustment

    def clear(self) -> None:
        self.cleared = True
        self.recorded.clear()


class StubExplanationService:
    """Explanation service that produces deterministic responses."""

    def explain(
        self,
        recommendation: Dict[str, Any],
        *,
        problem_description: str | None = None,
    ) -> cli.ExplanationResult:
        assert recommendation
        assert problem_description
        return cli.ExplanationResult(
            overview="Technique fits the scenario.",
            key_factors=["Clear trade-off analysis"],
            risks=["May ignore intuition"],
            next_steps=["List pros and cons"],
            raw_response='{"ok": true}',
        )


class RecordingOrchestrator:
    """Orchestrator double that records calls and dispatches handlers."""

    def __init__(
        self,
        *,
        handlers: Optional[Mapping[str, Handler]] = None,
        default: Optional[DefaultHandler] = None,
    ) -> None:
        self._handlers: Dict[str, Handler] = dict(handlers or {})
        self._default = default or (lambda _workflow, _context, _self: {"config": {}})
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.data: MutableMapping[str, Any] = {}

    def execute(self, workflow: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((workflow, context))
        handler = self._handlers.get(workflow)
        if handler:
            return handler(context, self)
        return self._default(workflow, context, self)


def mute_console(
    monkeypatch: Any,
    *,
    print_output: bool = True,
    json_output: bool = True,
    panel_output: bool = True,
    log_output: bool = True,
) -> None:
    """Silence Rich console output during tests."""

    if print_output:
        monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)
    if json_output and hasattr(cli.console, "print_json"):
        monkeypatch.setattr(cli.console, "print_json", lambda *args, **kwargs: None)
    if panel_output and hasattr(cli.console, "print_panel"):
        monkeypatch.setattr(cli.console, "print_panel", lambda *args, **kwargs: None, raising=False)  # type: ignore[arg-type]
    if log_output and hasattr(cli.console, "log"):
        monkeypatch.setattr(cli.console, "log", lambda *args, **kwargs: None, raising=False)  # type: ignore[arg-type]


def patch_runtime(monkeypatch: Any, orchestrator: RecordingOrchestrator, state: cli.AppState) -> None:
    """Patch runtime helpers to operate on the supplied orchestrator and state."""

    state.save = lambda path=cli.STATE_PATH: None  # type: ignore[assignment]
    monkeypatch.setattr(cli, "get_runtime", lambda: (orchestrator, state))
    monkeypatch.setattr(cli, "get_state", lambda: state)
    monkeypatch.setattr(cli, "get_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(cli, "set_runtime", lambda runtime: None)


def build_default_handlers() -> Dict[str, Handler]:
    """Return handlers covering the primary CLI workflows for tests."""

    def detect_handler(context: Dict[str, Any], _: RecordingOrchestrator) -> Dict[str, Any]:
        return {
            "recommendation": {
                "suggested_technique": "Decisional Balance",
                "why_it_fits": "Balances pros and cons",
                "steps": ["List options", "Score trade-offs"],
            },
            "matches": [
                {
                    "metadata": {
                        "name": "Decisional Balance",
                        "category": "Decision Making",
                        "description": "Compare pros and cons.",
                    },
                    "score": 0.92,
                }
            ],
            "preference_summary": "Prefers structured analysis.",
        }

    def summarize_handler(context: Dict[str, Any], _: RecordingOrchestrator) -> Dict[str, Any]:
        assert "technique_summary" in context
        return {"plan": {"milestones": ["Gather data", "Evaluate"]}}

    def simulate_handler(context: Dict[str, Any], _: RecordingOrchestrator) -> Dict[str, Any]:
        assert context["recommendation"]
        return {
            "simulation": {
                "simulation_overview": "Simulation overview",
                "scenario_variations": [
                    {"name": "Best case", "outcome": "Success", "guidance": "Stay on plan"}
                ],
                "cautions": ["Time pressure"],
                "recommended_follow_up": ["Review outcomes"],
            }
        }

    def compare_handler(context: Dict[str, Any], _: RecordingOrchestrator) -> Dict[str, Any]:
        assert context["matches"]
        return {
            "comparison": {
                "current_recommendation": "Decisional Balance",
                "best_alternative": "Six Thinking Hats",
                "comparison_points": [
                    {
                        "technique": "Decisional Balance",
                        "strengths": "Structured",
                        "risks": "Slow",
                        "best_for": "Trade-offs",
                    }
                ],
                "decision_guidance": ["Use hats for creativity"],
                "confidence_notes": "High",
            }
        }

    def feedback_handler(context: Dict[str, Any], orchestrator: RecordingOrchestrator) -> Dict[str, Any]:
        if context.get("action") == "record":
            orchestrator.data.setdefault("feedback_records", []).append(context)
            orchestrator.data["feedback_summary"] = {"summary": "Captured"}
            return {"status": "ok"}
        return orchestrator.data.get("feedback_summary", {"summary": ""})

    return {
        "detect_technique": detect_handler,
        "summarize_result": summarize_handler,
        "simulate_technique": simulate_handler,
        "compare_candidates": compare_handler,
        "feedback_loop": feedback_handler,
    }


def make_cli_runtime() -> tuple[RecordingOrchestrator, cli.AppState]:
    """Build a ready-to-use CLI runtime with default handlers."""

    orchestrator = RecordingOrchestrator(
        handlers=build_default_handlers(),
        default=lambda workflow, _context, _self: {"config": {}} if workflow == "config_update" else {},
    )
    state = cli.AppState()
    state.preference_service = StubPreferenceService(repository=None)
    state.explanation_service = StubExplanationService()
    return orchestrator, state


__all__ = [
    "PreferenceProfile",
    "RecordingOrchestrator",
    "StubExplanationService",
    "StubPreferenceService",
    "build_default_handlers",
    "make_cli_runtime",
    "mute_console",
    "patch_runtime",
]
