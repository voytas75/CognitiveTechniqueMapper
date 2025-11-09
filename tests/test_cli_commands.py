from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

import src.cli as cli


@dataclass
class PreferenceProfile:
    summary: str


class StubPreferenceService:
    def __init__(self) -> None:
        self._summary = "Prefers structured analysis."
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

    def clear(self) -> None:
        self.cleared = True


class StubExplanationService:
    def explain(
        self,
        recommendation: dict[str, Any],
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
            raw_response=json.dumps({"ok": True}),
        )


class StubSimulationResult:
    simulation_overview = "Walkthrough"


class StubOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._feedback_summary = {"summary": "Captured"}

    def execute(self, workflow: str, context: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((workflow, context))
        if workflow == "detect_technique":
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
        if workflow == "summarize_result":
            assert "technique_summary" in context
            return {"plan": {"milestones": ["Gather data", "Evaluate"]}}
        if workflow == "simulate_technique":
            assert context["recommendation"]
            return {
                "simulation": {
                    "simulation_overview": "Simulation overview",
                    "scenario_variations": [
                        {
                            "name": "Best case",
                            "outcome": "Success",
                            "guidance": "Stay on plan",
                        }
                    ],
                    "cautions": ["Time pressure"],
                    "recommended_follow_up": ["Review outcomes"],
                }
            }
        if workflow == "compare_candidates":
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
        if workflow == "feedback_loop":
            if context.get("action") == "record":
                return {"status": "ok"}
            return self._feedback_summary
        if workflow == "config_update":
            return {"config": "ok"}
        raise AssertionError(f"Unsupported workflow: {workflow}")


@pytest.fixture()
def patched_runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[StubOrchestrator, cli.AppState, StubPreferenceService]:
    orchestrator = StubOrchestrator()
    state = cli.AppState()
    state.save = lambda path=cli.STATE_PATH: None  # type: ignore[assignment]
    state.preference_service = StubPreferenceService()
    state.explanation_service = StubExplanationService()
    monkeypatch.setattr(cli, "get_runtime", lambda: (orchestrator, state))
    monkeypatch.setattr(cli, "get_state", lambda: state)
    monkeypatch.setattr(cli, "get_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(cli, "set_runtime", lambda runtime: None)
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.console, "print_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.console, "print_panel", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(cli.console, "log", lambda *args, **kwargs: None, raising=False)
    return orchestrator, state, state.preference_service  # type: ignore[return-value]


def test_cli_happy_path_flow(patched_runtime: tuple[StubOrchestrator, cli.AppState, StubPreferenceService]) -> None:
    orchestrator, state, preference_service = patched_runtime

    cli.describe("Need a decision framework", log_level=None)
    assert state.problem_description == "Need a decision framework"
    assert state.context_history[-1]["problem_description"] == "Need a decision framework"

    cli.analyze(show_candidates=True, log_level=None)
    assert state.last_recommendation is not None
    assert orchestrator.calls[0][0] == "detect_technique"

    cli.explain(log_level=None)
    assert state.last_explanation is not None
    assert state.last_explanation["overview"].startswith("Technique fits")

    cli.simulate(scenario=None, log_level=None)
    assert state.last_simulation["simulation_overview"] == "Simulation overview"

    cli.compare(focus=None, limit=1, log_level=None)
    assert state.last_comparison["best_alternative"] == "Six Thinking Hats"

    cli.feedback(
        "Helpful guidance",
        rating=5,
        technique=None,
        category=None,
        log_level=None,
    )
    assert preference_service.recorded[-1]["technique"] == "Decisional Balance"

    cli.history_show(limit=2, raw=True)
    cli.history_show(limit=0, raw=False)
    assert state.context_history

    cli.history_clear(force=True)
    assert not state.context_history

    cli.preferences_summary()
    cli.preferences_export()
    cli.preferences_reset(force=True)
    assert preference_service.cleared
