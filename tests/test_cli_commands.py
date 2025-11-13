from __future__ import annotations

from typing import Any

import pytest

import src.cli as cli
from tests.helpers.cli import (
    RecordingOrchestrator,
    StubExplanationService,
    StubPreferenceService,
    mute_console,
    patch_runtime,
)


@pytest.fixture()
def patched_runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[StubOrchestrator, cli.AppState, StubPreferenceService]:
    def detect_handler(context: dict[str, Any], _: RecordingOrchestrator) -> dict[str, Any]:
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

    def summarize_handler(context: dict[str, Any], _: RecordingOrchestrator) -> dict[str, Any]:
        assert "technique_summary" in context
        return {"plan": {"milestones": ["Gather data", "Evaluate"]}}

    def simulate_handler(context: dict[str, Any], _: RecordingOrchestrator) -> dict[str, Any]:
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

    def compare_handler(context: dict[str, Any], _: RecordingOrchestrator) -> dict[str, Any]:
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

    def feedback_handler(context: dict[str, Any], orchestrator: RecordingOrchestrator) -> dict[str, Any]:
        if context.get("action") == "record":
            orchestrator.data.setdefault("feedback_records", []).append(context)
            orchestrator.data["feedback_summary"] = {"summary": "Captured"}
            return {"status": "ok"}
        return orchestrator.data.get("feedback_summary", {"summary": ""})

    orchestrator = RecordingOrchestrator(
        handlers={
            "detect_technique": detect_handler,
            "summarize_result": summarize_handler,
            "simulate_technique": simulate_handler,
            "compare_candidates": compare_handler,
            "feedback_loop": feedback_handler,
        },
        default=lambda workflow, _context, _self: {"config": {}} if workflow == "config_update" else {},
    )
    state = cli.AppState()
    state.preference_service = StubPreferenceService()
    state.explanation_service = StubExplanationService()
    patch_runtime(monkeypatch, orchestrator, state)
    mute_console(monkeypatch)
    return orchestrator, state, state.preference_service  # type: ignore[return-value]


def test_cli_happy_path_flow(patched_runtime: tuple[RecordingOrchestrator, cli.AppState, StubPreferenceService]) -> None:
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
