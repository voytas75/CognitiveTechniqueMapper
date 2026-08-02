from __future__ import annotations

from typing import Any

import pytest
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
