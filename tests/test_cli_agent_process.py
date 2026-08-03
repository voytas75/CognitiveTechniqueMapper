"""Process-level agent contract tests for the CTM CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(
    state_path: Path, *args: str, stdin: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run one isolated CLI process with controlled input streams."""
    environment = os.environ | {"CTM_STATE_PATH": str(state_path)}
    return subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        cwd=PROJECT_ROOT,
        env=environment,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def test_describe_stdin_json_isolated_state_and_clean_streams(tmp_path: Path) -> None:
    """Agent input uses stdin JSON and does not leak across state files."""
    first_state = tmp_path / "first-state.json"
    second_state = tmp_path / "second-state.json"
    payload = json.dumps(
        {"action": "describe", "problem_description": "First isolated problem"}
    )

    first = _run_cli(first_state, "describe", "--stdin-json", stdin=payload)
    second = _run_cli(
        second_state,
        "describe",
        "--stdin-json",
        stdin=payload.replace("First", "Second"),
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert json.loads(first.stdout) == {
        "ok": True,
        "problem_description": "First isolated problem",
    }
    assert json.loads(second.stdout)["problem_description"] == "Second isolated problem"
    assert (
        json.loads(first_state.read_text())["problem_description"]
        == "First isolated problem"
    )
    assert (
        json.loads(second_state.read_text())["problem_description"]
        == "Second isolated problem"
    )


def test_analyze_json_error_uses_stderr_and_nonzero_exit(tmp_path: Path) -> None:
    """A missing agent prerequisite is a structured stderr error, not Rich output."""
    result = _run_cli(tmp_path / "empty-state.json", "analyze", "--json")

    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "ok": False,
        "error": "No problem description found. Use `describe` first.",
    }
