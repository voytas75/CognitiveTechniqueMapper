"""Primary CLI commands for the Cognitive Technique Mapper."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import typer
from rich.panel import Panel

from src.cli.io import console
from src.cli.renderers import (
    render_analysis_output,
    render_comparison_output,
    render_explanation_output,
    render_simulation_output,
)
from src.cli.reporting import build_report_payload, render_report_markdown
from src.cli.utils import (
    active_preference_summary,
    apply_log_override,
    infer_category_from_matches,
)
from src.services.interactive_question_flow import create_interactive_flow

logger = logging.getLogger(__name__)


def _cli() -> Any:
    return sys.modules["src.cli"]


# TODO: Split command groups into separate modules by workflow responsibility.
def _object(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return cast(dict[str, Any], value)
    return {}


def _bound_problem_description(state: object) -> str:
    """Return the problem description captured with the latest analysis."""

    recommendation = _object(getattr(state, "last_recommendation", None))
    if not recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")
    problem_description = recommendation.get("problem_description")
    if not isinstance(problem_description, str) or not problem_description.strip():
        raise typer.BadParameter(
            "Latest recommendation has no bound problem description. Run `analyze` again."
        )
    return problem_description


def describe(
    problem: str = typer.Argument(..., help="Describe your problem or challenge."),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation (e.g., DEBUG, INFO).",
    ),
) -> None:
    """Store the user's problem description for subsequent workflows."""

    apply_log_override(log_level)

    state = _cli().get_state()
    state.problem_description = problem
    state.last_recommendation = None
    state.last_explanation = None
    state.last_simulation = None
    state.last_comparison = None
    state.context_history.append({"problem_description": problem})
    state.save()
    logger.info("Problem description captured (length=%s)", len(problem))
    console.print(Panel(f"[bold]Problem captured:[/]\n{problem}", title="Describe"))


def analyze(
    show_candidates: bool = typer.Option(
        False,
        "--show-candidates/--hide-candidates",
        help="Display the candidate shortlist with similarity scores.",
    ),
    show_diagnostics: bool = typer.Option(
        False,
        "--show-diagnostics/--hide-diagnostics",
        help="Explain why the selected technique outranked alternatives.",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Trigger the detect_technique workflow."""

    cli_module = _cli()
    state = cli_module.get_state()
    if not state.problem_description:
        raise typer.BadParameter("No problem description found. Use `describe` first.")

    apply_log_override(log_level)

    orchestrator = cli_module.get_orchestrator()
    context = {
        "problem_description": state.problem_description,
        "include_diagnostics": show_diagnostics,
    }
    try:
        result = orchestrator.execute("detect_technique", context)
    except RuntimeError as exc:
        console.print(f"[red]Analyze failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    result["problem_description"] = state.problem_description
    recommendation = _object(result.get("recommendation"))
    plan_output: dict[str, Any] | None = result.get("plan")

    if not plan_output and recommendation:
        plan_summary = cli_module.compose_plan_summary(recommendation)
        if plan_summary:
            try:
                plan_output = cli_module.get_orchestrator().execute(
                    "summarize_result", {"technique_summary": plan_summary}
                )
            except (RuntimeError, ValueError) as exc:  # pragma: no cover - LLM issues
                logger.warning("Plan generation failed: %s", exc)

    if plan_output and plan_output.get("plan"):
        result["plan"] = plan_output.get("plan")

    state.last_recommendation = result
    state.context_history.append(result)
    state.save()
    logger.info("Analysis completed.")
    render_analysis_output(
        recommendation,
        result.get("plan"),
        problem_description=_bound_problem_description(state),
        preference_summary=result.get("preference_summary"),
        matches=result.get("matches") if show_candidates else None,
        diagnostics=result.get("diagnostics") if show_diagnostics else None,
    )


def explain(
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Explain the logic behind the last recommendation via the explain_logic workflow."""

    state = _cli().get_state()
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")
    analysis_problem_description = _bound_problem_description(state)

    apply_log_override(log_level)

    if not state.explanation_service:
        raise typer.BadParameter("Explanation service not initialized.")

    try:
        explanation = state.explanation_service.explain(
            state.last_recommendation or {},
            problem_description=analysis_problem_description,
        )
    except RuntimeError as exc:
        console.print(f"[red]Explain failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    logger.info("Explain workflow executed.")
    state.last_explanation = explanation.as_dict()
    state.context_history.append({"explanation": state.last_explanation})
    state.save()
    render_explanation_output(explanation)


def simulate(
    scenario: Optional[str] = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Optional scenario focus or constraint to explore during simulation.",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Simulate applying the recommended technique across scenario variations."""

    cli_module = _cli()
    state = cli_module.get_state()
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")
    analysis_problem_description = _bound_problem_description(state)

    recommendation = _object(state.last_recommendation.get("recommendation"))
    if not recommendation:
        raise typer.BadParameter("Current recommendation payload is empty.")

    apply_log_override(log_level)
    preference_summary = active_preference_summary()
    context = {
        "recommendation": recommendation,
        "problem_description": analysis_problem_description,
        "scenario": scenario or analysis_problem_description,
        "preference_summary": preference_summary,
    }
    orchestrator = cli_module.get_orchestrator()
    try:
        result = orchestrator.execute("simulate_technique", context)
    except RuntimeError as exc:
        console.print(f"[red]Simulation failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    simulation = _object(result.get("simulation"))
    state.last_simulation = simulation
    state.context_history.append({"simulation": simulation})
    state.save()
    logger.info("Simulation workflow executed.")
    render_simulation_output(simulation)


def compare(
    focus: Optional[str] = typer.Option(
        None,
        "--focus",
        "-f",
        help="Optional technique name to prioritise in the comparison.",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Maximum number of candidates to include from the shortlist (0 = all).",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Compare shortlisted techniques and highlight trade-offs."""

    cli_module = _cli()
    state = cli_module.get_state()
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")
    _bound_problem_description(state)

    recommendation = _object(state.last_recommendation.get("recommendation"))
    matches = state.last_recommendation.get("matches") or []
    if not recommendation or not matches:
        raise typer.BadParameter(
            "Candidate shortlist unavailable. Re-run `analyze` to regenerate matches."
        )

    apply_log_override(log_level)
    shortlist = matches if limit <= 0 else matches[:limit]
    preference_summary = active_preference_summary()
    context = {
        "recommendation": recommendation,
        "matches": shortlist,
        "focus": focus,
        "preference_summary": preference_summary,
    }
    orchestrator = cli_module.get_orchestrator()
    try:
        result = orchestrator.execute("compare_candidates", context)
    except RuntimeError as exc:
        console.print(f"[red]Comparison failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    comparison = _object(result.get("comparison"))
    state.last_comparison = comparison
    state.context_history.append({"comparison": comparison})
    state.save()
    logger.info("Comparison workflow executed.")
    render_comparison_output(comparison)


def refresh(
    rebuild_embeddings: bool = typer.Option(
        True,
        "--rebuild-embeddings/--skip-embeddings",
        help="Recompute and sync embeddings with the vector store.",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Reload the techniques dataset and optionally rebuild embeddings."""

    apply_log_override(log_level)

    initializer, sqlite_client = _cli()._create_initializer()
    try:
        initializer.refresh(rebuild_embeddings=rebuild_embeddings)
    except Exception as exc:  # pragma: no cover - dependent on external services
        console.print(f"[red]Refresh failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        sqlite_client.close()

    _cli()._refresh_runtime()
    console.print(
        Panel("Dataset refreshed with latest configuration.", title="Refresh")
    )


def feedback(
    message: str = typer.Argument(..., help="Feedback message."),
    rating: Optional[int] = typer.Option(None, help="Optional rating 1-5."),
    technique: Optional[str] = typer.Option(
        None,
        "--technique",
        "-t",
        help="Technique the feedback refers to (defaults to last recommendation).",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Technique category the feedback refers to.",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Record user feedback and display the summary of recent entries."""

    cli_module = _cli()
    state = cli_module.get_state()
    orchestrator = cli_module.get_orchestrator()

    apply_log_override(log_level)
    if rating is not None and (rating < 1 or rating > 5):
        raise typer.BadParameter("Rating must be between 1 and 5.")

    if technique is None and state.last_recommendation:
        recommendation = _object(state.last_recommendation.get("recommendation"))
        suggested_technique = recommendation.get("suggested_technique")
        if isinstance(suggested_technique, str):
            technique = suggested_technique
    if category is None and technique:
        category = infer_category_from_matches(
            (
                state.last_recommendation.get("matches")
                if state.last_recommendation
                else []
            ),
            technique,
        )

    context = {
        "action": "record",
        "message": message,
        "rating": rating,
        "workflow": "detect_technique",
        "technique": technique,
        "category": category,
    }
    try:
        orchestrator.execute("feedback_loop", context)
        summary = orchestrator.execute("feedback_loop", {})
    except RuntimeError as exc:
        console.print(f"[red]Feedback failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    logger.info("Feedback recorded (rating=%s)", rating)
    console.print(
        Panel(summary.get("summary", "No summary available."), title="Feedback Summary")
    )


def report(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional file to write the report as Markdown.",
        dir_okay=False,
        resolve_path=True,
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Generate a Markdown report of the latest analysis artifacts."""

    apply_log_override(log_level)

    state = _cli().get_state()
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")
    _bound_problem_description(state)

    payload = build_report_payload(state)
    markdown = render_report_markdown(payload)

    if output:
        output.write_text(markdown, encoding="utf-8")
        console.print(Panel(f"Report saved to {output}", title="Report"))
    else:
        console.print(markdown)


def interactive_flow(
    show_tree: bool = typer.Option(
        False,
        "--show-tree",
        help="Render the decision tree after completing the flow.",
    ),
) -> None:
    """Run the interactive decision-tree question flow.

    Args:
        show_tree (bool): Whether to render the full decision tree after
            completing the flow.
    """

    flow = create_interactive_flow(console, show_visualization=show_tree)
    technique_identifier, technique_label = flow.run()
    if technique_identifier and technique_label:
        console.print(
            Panel(
                f"Technique: {technique_identifier} ({technique_label})",
                title="Interactive Flow Result",
                style="green",
            )
        )


__all__ = [
    "analyze",
    "compare",
    "describe",
    "explain",
    "feedback",
    "interactive_flow",
    "refresh",
    "report",
    "simulate",
]
