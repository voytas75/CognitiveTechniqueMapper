"""Primary CLI commands for the Cognitive Technique Mapper."""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import typer
from rich.panel import Panel

from src.cli.io import console
from src.cli.renderers import (
    render_analysis_output,
    render_comparison_output,
    render_explanation_output,
    render_simulation_output,
)
from src.cli.utils import (
    active_preference_summary,
    apply_log_override,
    infer_category_from_matches,
)

logger = logging.getLogger(__name__)


def _cli() -> Any:
    return sys.modules["src.cli"]


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
    context = {"problem_description": state.problem_description}
    try:
        result = orchestrator.execute("detect_technique", context)
    except RuntimeError as exc:
        console.print(f"[red]Analyze failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    recommendation = result.get("recommendation") or {}
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
        preference_summary=result.get("preference_summary"),
        matches=result.get("matches") if show_candidates else None,
    )


def explain(
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    )
) -> None:
    """Explain the logic behind the last recommendation via the explain_logic workflow."""

    state = _cli().get_state()
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")

    apply_log_override(log_level)

    if not state.explanation_service:
        raise typer.BadParameter("Explanation service not initialized.")

    try:
        explanation = state.explanation_service.explain(
            state.last_recommendation or {},
            problem_description=state.problem_description,
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

    recommendation = state.last_recommendation.get("recommendation") or {}
    if not recommendation:
        raise typer.BadParameter("Current recommendation payload is empty.")

    apply_log_override(log_level)
    preference_summary = active_preference_summary()
    context = {
        "recommendation": recommendation,
        "problem_description": state.problem_description,
        "scenario": scenario or state.problem_description,
        "preference_summary": preference_summary,
    }
    orchestrator = cli_module.get_orchestrator()
    try:
        result = orchestrator.execute("simulate_technique", context)
    except RuntimeError as exc:
        console.print(f"[red]Simulation failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    simulation = result.get("simulation") or {}
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

    recommendation = state.last_recommendation.get("recommendation") or {}
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

    comparison = result.get("comparison") or {}
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
    console.print(Panel("Dataset refreshed with latest configuration.", title="Refresh"))


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
        technique = (
            (state.last_recommendation.get("recommendation") or {}).get(
                "suggested_technique"
            )
        )
    if category is None and technique:
        category = infer_category_from_matches(
            state.last_recommendation.get("matches") if state.last_recommendation else [],
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

    if state.preference_service:
        state.preference_service.record_preference(
            technique=technique,
            category=category,
            rating=rating,
            notes=message,
        )

    logger.info("Feedback recorded (rating=%s)", rating)
    console.print(
        Panel(summary.get("summary", "No summary available."), title="Feedback Summary")
    )


__all__ = [
    "analyze",
    "compare",
    "describe",
    "explain",
    "feedback",
    "refresh",
    "simulate",
]
