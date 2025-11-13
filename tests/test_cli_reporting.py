from __future__ import annotations

from src.cli.reporting import build_report_payload, render_report_markdown
from tests.helpers.cli import make_cli_runtime


def test_build_report_payload_includes_state() -> None:
    _orchestrator, state = make_cli_runtime()
    state.problem_description = "Need a decision framework"
<<<<<<< HEAD
    state.last_recommendation = {
        "recommendation": {"suggested_technique": "Decisional Balance"},
        "plan": {"milestones": ["Step1"]},
    }
=======
    state.last_recommendation = {"recommendation": "Decisional Balance"}
>>>>>>> a5ff1106c426316af52b3c650e2afc274453d6fc
    state.last_explanation = {"overview": "Explanation"}
    state.last_simulation = {"simulation_overview": "Sim overview"}
    state.last_comparison = {"current_recommendation": "Decisional Balance"}

    payload = build_report_payload(state)

    assert payload["problem_description"] == "Need a decision framework"
<<<<<<< HEAD
    assert payload["recommendation"]["suggested_technique"] == "Decisional Balance"
    assert payload["recommendation"]["plan"] == {"milestones": ["Step1"]}
=======
    assert payload["recommendation"]["recommendation"] == "Decisional Balance"
>>>>>>> a5ff1106c426316af52b3c650e2afc274453d6fc


def test_render_report_markdown_contains_sections() -> None:
    _orchestrator, state = make_cli_runtime()
    state.problem_description = "Need a decision framework"
    state.last_recommendation = {
        "suggested_technique": "Decisional Balance",
        "why_it_fits": "Balances pros and cons",
        "steps": ["List options"],
        "plan": {"milestones": ["Step1"]},
    }
    state.last_explanation = {
        "overview": "Fits the context",
        "key_factors": ["Structured"],
        "risks": ["Time"],
        "next_steps": ["Follow up"],
    }
    state.last_simulation = {
        "simulation_overview": "Overview",
        "scenario_variations": [{"name": "Best", "outcome": "Success", "guidance": "Stay"}],
        "cautions": ["Time pressure"],
        "recommended_follow_up": ["Review"],
    }
    state.last_comparison = {
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
        "decision_guidance": ["Use hats"],
        "confidence_notes": "High",
    }

    payload = build_report_payload(state)
    markdown = render_report_markdown(payload)

    assert "# Cognitive Technique Report" in markdown
    assert "## Recommendation" in markdown
    assert "## Explanation" in markdown
    assert "## Simulation" in markdown
    assert "## Comparison" in markdown
