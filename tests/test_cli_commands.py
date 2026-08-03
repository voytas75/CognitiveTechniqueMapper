from __future__ import annotations

from typing import Any

import pytest
import typer
from rich.panel import Panel

import src.cli as cli
from tests.helpers.cli import (
    RecordingOrchestrator,
    StubPreferenceService,
    make_cli_runtime,
    mute_console,
    patch_runtime,
)


@pytest.fixture()
def patched_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RecordingOrchestrator, cli.AppState, StubPreferenceService]:
    orchestrator, state = make_cli_runtime()
    patch_runtime(monkeypatch, orchestrator, state)
    mute_console(monkeypatch)
    return orchestrator, state, state.preference_service  # type: ignore[return-value]


def test_cli_happy_path_flow(
    patched_runtime: tuple[RecordingOrchestrator, cli.AppState, StubPreferenceService],
) -> None:
    orchestrator, state, preference_service = patched_runtime

    cli.describe("Need a decision framework", log_level=None)
    assert state.problem_description == "Need a decision framework"
    assert (
        state.context_history[-1]["problem_description"] == "Need a decision framework"
    )

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
    assert len(preference_service.recorded) == 1
    assert preference_service.recorded[-1]["technique"] == "Decisional Balance"

    cli.history_show(limit=2, raw=True)
    cli.history_show(limit=0, raw=False)
    assert state.context_history

    cli.history_clear(force=True)
    assert not state.context_history

    cli.preferences_summary()
    cli.preferences_export()
    cli.preferences_impact(limit=3)
    cli.preferences_reset(force=True)
    assert preference_service.cleared


def test_history_show_renders_human_readable_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = cli.AppState()
    state.context_history.append({"problem_description": "Need a creative path"})

    monkeypatch.setattr(cli, "get_state", lambda: state)

    captured: list[Any] = []

    def capture(renderable: Any, *args: Any, **kwargs: Any) -> None:
        captured.append(renderable)

    monkeypatch.setattr(cli.console, "print", capture)

    cli.history_show(limit=1, raw=False)

    assert captured, "Expected at least one rendered panel"
    panel = captured[0]
    assert isinstance(panel, Panel)
    assert "Need a creative path" in str(panel.renderable)


def test_history_binds_analyze_entry_to_its_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = cli.AppState()
    state.context_history.append(
        {
            "problem_description": "Problem used for analysis",
            "recommendation": {"suggested_technique": "Decisional Balance"},
        }
    )
    monkeypatch.setattr(cli, "get_state", lambda: state)

    captured: list[Any] = []

    def capture(renderable: Any, *args: Any, **kwargs: Any) -> None:
        captured.append(renderable)

    monkeypatch.setattr(cli.console, "print", capture)

    cli.history_show(limit=1, raw=False)

    assert isinstance(captured[0], Panel)
    assert "Analyze" in str(captured[0].title)
    assert "Problem used for analysis" in str(captured[0].renderable)


def test_history_clear_all_removes_every_persisted_session_value(
    patched_runtime: tuple[RecordingOrchestrator, cli.AppState, StubPreferenceService],
) -> None:
    _, state, _ = patched_runtime
    state.problem_description = "Sensitive problem"
    state.last_recommendation = {"raw_response": "Sensitive model response"}
    state.last_explanation = {"overview": "Sensitive explanation"}
    state.last_simulation = {"simulation_overview": "Sensitive simulation"}
    state.last_comparison = {"best_alternative": "Sensitive comparison"}
    state.context_history.append({"problem_description": "Sensitive problem"})

    cli.history_clear(force=True, all_state=True)

    assert state.problem_description is None
    assert state.last_recommendation is None
    assert state.last_explanation is None
    assert state.last_simulation is None
    assert state.last_comparison is None
    assert state.context_history == []


def test_describe_invalidates_artifacts_from_the_previous_problem(
    patched_runtime: tuple[RecordingOrchestrator, cli.AppState, StubPreferenceService],
) -> None:
    _, state, _ = patched_runtime
    state.last_recommendation = {
        "problem_description": "Previous problem",
        "recommendation": {"suggested_technique": "Decisional Balance"},
    }
    state.last_explanation = {"overview": "Previous explanation"}
    state.last_simulation = {"simulation_overview": "Previous simulation"}
    state.last_comparison = {"best_alternative": "Previous comparison"}

    cli.describe("New problem", log_level=None)

    assert state.problem_description == "New problem"
    assert state.last_recommendation is None
    assert state.last_explanation is None
    assert state.last_simulation is None
    assert state.last_comparison is None


def test_analyze_binds_and_renders_the_problem_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, state = make_cli_runtime()
    state.problem_description = "Choose between two implementation approaches."
    patch_runtime(monkeypatch, orchestrator, state)
    captured: list[Any] = []

    def capture(renderable: Any, *args: Any, **kwargs: Any) -> None:
        captured.append(renderable)

    monkeypatch.setattr(cli.console, "print", capture)

    cli.analyze(show_candidates=False, show_diagnostics=False, log_level=None)

    assert state.last_recommendation is not None
    assert (
        state.last_recommendation["problem_description"]
        == "Choose between two implementation approaches."
    )
    assert captured
    assert "Analyzing problem:" in str(captured[0].renderable)
    assert "Choose between two implementation approaches." in str(
        captured[0].renderable
    )


def test_explain_uses_problem_bound_to_the_recommendation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, state = make_cli_runtime()
    state.problem_description = "New unsolved problem"
    state.last_recommendation = {
        "problem_description": "Problem used for analysis",
        "recommendation": {"suggested_technique": "Decisional Balance"},
    }
    captured: dict[str, str | None] = {}

    class CapturingExplanationService:
        def explain(
            self,
            recommendation: dict[str, Any],
            *,
            problem_description: str | None = None,
        ) -> cli.ExplanationResult:
            assert recommendation
            captured["problem_description"] = problem_description
            return cli.ExplanationResult(
                overview="Explanation",
                key_factors=[],
                risks=[],
                next_steps=[],
                raw_response="{}",
            )

    state.explanation_service = CapturingExplanationService()  # type: ignore[assignment]
    patch_runtime(monkeypatch, orchestrator, state)
    mute_console(monkeypatch)

    cli.explain(log_level=None)

    assert captured["problem_description"] == "Problem used for analysis"


def test_explain_rejects_legacy_recommendation_without_problem_provenance(
    patched_runtime: tuple[RecordingOrchestrator, cli.AppState, StubPreferenceService],
) -> None:
    _, state, _ = patched_runtime
    state.problem_description = "Current problem"
    state.last_recommendation = {
        "recommendation": {"suggested_technique": "Decisional Balance"}
    }

    with pytest.raises(typer.BadParameter, match="bound problem description"):
        cli.explain(log_level=None)
