from __future__ import annotations

from typing import Any

import pytest

import src.cli as cli
from tests.helpers.cli import RecordingOrchestrator, make_cli_runtime, mute_console, patch_runtime


@pytest.fixture()
def patched_runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[StubOrchestrator, cli.AppState, StubPreferenceService]:
    orchestrator, state = make_cli_runtime()
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
