from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import typer

import src.cli as cli
import src.cli.renderers as renderers
from src.cli.utils import infer_category_from_matches


def test_app_state_save_and_load(tmp_path: Path) -> None:
    state = cli.AppState(
        problem_description="Decision needed",
        last_recommendation={"technique": "Decisional Balance"},
    )
    path = tmp_path / "state.json"
    state.save(path)

    loaded = cli.AppState.load(path)

    assert loaded.problem_description == "Decision needed"
    assert loaded.last_recommendation["technique"] == "Decisional Balance"


def test_app_state_save_preserves_existing_state_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    previous = '{"problem_description": "Existing state"}'
    path.write_text(previous, encoding="utf-8")

    def fail_replace(_: Path, __: Path) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replacement failure"):
        cli.AppState(problem_description="New state").save(path)

    assert path.read_text(encoding="utf-8") == previous
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_app_state_load_warns_when_state_file_is_corrupted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not valid JSON", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="src.cli.state"):
        loaded = cli.AppState.load(path)

    assert loaded == cli.AppState()
    assert "Failed to load session state" in caplog.text


def test_apply_log_override_handles_invalid_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []
    monkeypatch.setattr(cli, "set_runtime_level", lambda level: captured.append(level))

    cli._apply_log_override("debug")
    assert captured == ["debug"]

    monkeypatch.setattr(
        cli, "set_runtime_level", lambda level: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(typer.BadParameter):
        cli._apply_log_override("trace")


def test_infer_category_uses_match_id_when_metadata_is_malformed() -> None:
    matches = [
        {"id": "Decisional Balance", "metadata": "invalid", "category": "Decision"}
    ]

    assert infer_category_from_matches(matches, "decisional balance") == "Decision"


def test_compose_plan_summary_formats_sections() -> None:
    recommendation = {
        "suggested_technique": "Decisional Balance",
        "why_it_fits": "Balances pros and cons",
        "steps": ["List options", "Score trade-offs"],
    }
    summary = cli._compose_plan_summary(recommendation)

    assert "Technique: Decisional Balance" in summary
    assert "Suggested steps" in summary


def test_render_helpers_emit_console_output(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[Any] = []
    monkeypatch.setattr(cli.console, "print", lambda value, **_: printed.append(value))

    cli._render_analysis_output(
        recommendation={
            "suggested_technique": "Decisional Balance",
            "why_it_fits": "Balances pros and cons",
            "steps": ["List options", "Score trade-offs"],
        },
        plan={"summary": "Plan"},
        preference_summary="Prefers structure",
        matches=[
            {
                "metadata": {
                    "name": "Decisional Balance",
                    "category": "Decision Making",
                    "description": "Compare pros and cons",
                },
                "score": 0.92,
            }
        ],
    )

    cli._render_explanation_output(
        cli.ExplanationResult(
            overview="Fits",
            key_factors=["Structured"],
            risks=["Slow"],
            next_steps=["List"],
            raw_response=json.dumps({}),
        )
    )

    cli._render_simulation_output(
        {
            "simulation_overview": "Overview",
            "scenario_variations": [
                {"name": "Best", "outcome": "Success", "guidance": "Stay"}
            ],
            "cautions": ["Time"],
            "recommended_follow_up": ["Review"],
        }
    )

    cli._render_comparison_output(
        {
            "current_recommendation": "Decisional Balance",
            "best_alternative": "Six Thinking Hats",
            "comparison_points": [
                {
                    "technique": "Decisional Balance",
                    "strengths": "Structured",
                    "risks": "Slow",
                    "best_for": "Decisions",
                }
            ],
            "decision_guidance": ["Use hats"],
            "confidence_notes": "High",
        }
    )

    assert printed  # ensure console output occurred


def test_render_selection_diagnostics_skips_nonobject_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[Any] = []

    def capture(value: Any, **_: Any) -> None:
        printed.append(value)

    monkeypatch.setattr(renderers.console, "print", capture)

    renderers.render_selection_diagnostics(
        {
            "summary": "Structured review",
            "comparisons": [
                {"technique": "Valid Candidate", "score_reason": "Strong fit"},
                "malformed comparison",
            ],
            "follow_up": ["Clarify goals", 2],
        }
    )

    content = str(printed[0].renderable)
    assert "Valid Candidate" in content
    assert "Clarify goals" in content


def test_render_simulation_skips_malformed_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[Any] = []

    def capture(value: Any, **_: Any) -> None:
        printed.append(value)

    monkeypatch.setattr(renderers.console, "print", capture)
    renderers.render_simulation_output(
        {
            "simulation_overview": "Walkthrough",
            "scenario_variations": [
                {"name": "Best case", "outcome": "Success", "guidance": "Continue"},
                "malformed variation",
            ],
            "cautions": "not a list",
            "recommended_follow_up": "not a list",
        }
    )

    content = str(printed[0].renderable)
    assert "Best case" in content
    assert "Cautions" not in content
    assert "Recommended follow-up" not in content


def test_render_comparison_skips_malformed_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[Any] = []

    def capture(value: Any, **_: Any) -> None:
        printed.append(value)

    monkeypatch.setattr(renderers.console, "print", capture)
    renderers.render_comparison_output(
        {
            "current_recommendation": "Structured review",
            "comparison_points": [
                {"technique": "Valid alternative", "strengths": "Fast"},
                "malformed point",
            ],
            "decision_guidance": "not a list",
        }
    )

    content = str(printed[0].renderable)
    assert "Valid alternative" in content
    assert "Decision guidance" not in content


def test_render_candidate_matches_skips_malformed_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[Any] = []

    def capture(value: Any, **_: Any) -> None:
        printed.append(value)

    monkeypatch.setattr(renderers.console, "print", capture)
    renderers.render_candidate_matches(
        [
            {
                "metadata": {
                    "name": "Valid candidate",
                    "category": "Decision",
                    "description": "Useful summary",
                },
                "score": 0.8,
            },
            {"metadata": "malformed metadata", "id": "Fallback candidate"},
            "malformed match",
        ]
    )

    content = str(printed[0].renderable)
    assert "Valid candidate" in content
    assert "Fallback candidate" in content


def test_render_analysis_ignores_nonlist_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[Any] = []

    def capture(value: Any, **_: Any) -> None:
        printed.append(value)

    monkeypatch.setattr(renderers.console, "print", capture)
    renderers.render_analysis_output(
        recommendation={
            "suggested_technique": "Review",
            "why_it_fits": "Structured",
            "steps": "not a list",
        },
        plan=None,
        problem_description="Example problem",
    )

    content = str(printed[0].renderable)
    assert "How to apply" not in content


def test_render_preference_impacts_skips_malformed_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[Any] = []

    def capture(value: Any, **_: Any) -> None:
        printed.append(value)

    monkeypatch.setattr(renderers.console, "print", capture)
    renderers.render_preference_impacts(
        {
            "categories": [
                {"name": "Decision", "adjustment": 0.2, "count": 2},
                "malformed entry",
            ],
            "techniques": "not a list",
        }
    )

    content = str(printed[0].renderable)
    assert "Decision" in content
    assert "Techniques" not in content


def test_active_preference_summary_and_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubPreferenceService:
        def preference_summary(self) -> str:
            return "Prefers structure"

    state = cli.AppState()
    state.preference_service = StubPreferenceService()
    monkeypatch.setattr(cli, "get_state", lambda: state)

    summary = cli._active_preference_summary()
    assert summary == "Prefers structure"

    matches = [
        {"metadata": {"name": "Decisional Balance", "category": "Decision Making"}}
    ]
    category = cli._infer_category_from_matches(matches, "Decisional Balance")
    assert category == "Decision Making"


def test_prompt_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typer, "prompt", lambda label, default="": " 0.5 ")
    assert cli._prompt_float("Temperature", None) == 0.5

    monkeypatch.setattr(typer, "prompt", lambda label, default="": "42")
    assert cli._prompt_int("Max", None) == 42

    monkeypatch.setattr(typer, "prompt", lambda label, default="": "value")
    assert cli._prompt_value("Field", None) == "value"

    monkeypatch.setattr(typer, "prompt", lambda label, default="": "not-a-number")
    with pytest.raises(typer.BadParameter):
        cli._prompt_float("Temperature", None)
