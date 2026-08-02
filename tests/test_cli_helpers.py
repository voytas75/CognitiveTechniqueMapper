from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer

import src.cli as cli
import src.cli.renderers as renderers


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
