from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import src.cli as cli
from tests.helpers.cli import make_cli_runtime, mute_console, patch_runtime


@pytest.fixture()
def cli_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[CliRunner, cli.AppState, Any]:
    runner = CliRunner()
    orchestrator, state = make_cli_runtime()
    state.save = lambda path=cli.STATE_PATH: None  # type: ignore[assignment]
    patch_runtime(monkeypatch, orchestrator, state)
    cli.set_runtime((orchestrator, state))
    return runner, state, orchestrator


def test_end_to_end_cli_flow(cli_session: tuple[CliRunner, cli.AppState, Any]) -> None:
    runner, state, orchestrator = cli_session

    result = runner.invoke(cli.app, ["describe", "Need a decision framework"])
    assert result.exit_code == 0
    assert state.problem_description == "Need a decision framework"

    result = runner.invoke(cli.app, ["analyze", "--show-candidates"])
    assert result.exit_code == 0
    assert state.last_recommendation is not None
    assert orchestrator.calls[0][0] == "detect_technique"

    result = runner.invoke(cli.app, ["explain"])
    assert result.exit_code == 0
    assert state.last_explanation is not None

    result = runner.invoke(cli.app, ["simulate"])
    assert result.exit_code == 0
    assert state.last_simulation is not None

    result = runner.invoke(cli.app, ["compare", "--limit", "1"])
    assert result.exit_code == 0
    assert state.last_comparison is not None

    result = runner.invoke(
        cli.app,
        [
            "feedback",
            "Helpful guidance",
            "--rating",
            "5",
        ],
    )
    assert result.exit_code == 0
    assert orchestrator.data["feedback_records"]

    result = runner.invoke(cli.app, ["history", "show", "--limit", "1", "--raw"])
    assert result.exit_code == 0
    assert state.context_history

    result = runner.invoke(cli.app, ["history", "clear", "--force"])
    assert result.exit_code == 0
    assert not state.context_history

    result = runner.invoke(cli.app, ["preferences", "summary"])
    assert result.exit_code == 0

    result = runner.invoke(cli.app, ["settings", "show"])
    assert result.exit_code == 0
