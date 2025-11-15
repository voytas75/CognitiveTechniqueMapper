"""Rich renderers for CLI outputs."""

from __future__ import annotations

from typing import Any, Optional

from rich.panel import Panel
from rich.table import Table

from src.cli.io import console
from src.services.explanation_service import ExplanationResult


def render_analysis_output(
    recommendation: dict[str, Any],
    plan: Any,
    *,
    preference_summary: str | None = None,
    matches: Any = None,
    diagnostics: Optional[dict[str, Any]] = None,
) -> None:
    """Display structured recommendation and optional plan to the console."""

    if not recommendation:
        console.print(Panel("No recommendation returned.", title="Suggested Technique"))
        return

    technique = recommendation.get("suggested_technique") or "No recommendation"
    why_it_fits = recommendation.get("why_it_fits") or "No justification provided."
    steps = recommendation.get("steps") or []

    lines = [
        f"[bold]Suggested:[/]\n{technique}",
        f"[bold]Why it fits:[/]\n{why_it_fits}",
    ]

    if preference_summary:
        lines.append(f"[bold]Preference context:[/]\n{preference_summary}")

    if steps:
        lines.append("[bold]How to apply:[/]")
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")

    if plan:
        lines.append("\n[bold]Implementation plan:[/]")
        lines.append(str(plan))

    console.print(Panel("\n".join(lines), title="Suggested Technique"))

    if matches is not None:
        render_candidate_matches(matches)

    if diagnostics:
        render_selection_diagnostics(diagnostics)


def render_candidate_matches(matches: Any) -> None:
    """Render candidate technique matches."""

    if not matches:
        console.print(
            Panel("No candidate techniques were returned.", title="Candidate Matches")
        )
        return

    lines: list[str] = []
    for idx, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        name = metadata.get("name") or match.get("id") or "Unknown"
        score = match.get("score")
        score_display = (
            f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        )
        category = metadata.get("category") or ""
        lines.append(f"{idx}. {name} (score: {score_display})")
        if category:
            lines.append(f"   Category: {category}")
        description = metadata.get("description") or match.get("document") or ""
        if description:
            lines.append(f"   Summary: {description}")
    console.print(Panel("\n".join(lines), title="Candidate Matches"))


def render_selection_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Render LLM-provided diagnostics that contrast candidates."""

    summary = diagnostics.get("summary") or diagnostics.get("overview")
    preference_impact = diagnostics.get("preference_impact")
    comparisons = diagnostics.get("comparisons") or []
    follow_up = diagnostics.get("follow_up") or diagnostics.get("next_questions")

    lines: list[str] = []
    if summary:
        lines.append(f"[bold]Why this technique won:[/]\n{summary}")

    if comparisons:
        lines.append("\n[bold]Runner-up insights:[/]")
        for entry in comparisons:
            technique = entry.get("technique") or entry.get("name") or "Candidate"
            reason = entry.get("score_reason") or entry.get("summary") or ""
            when = entry.get("when_to_choose") or entry.get("best_when") or ""
            caution = entry.get("cautions") or entry.get("risks") or ""
            lines.append(f"- {technique}")
            if reason:
                lines.append(f"  Reason: {reason}")
            if when:
                lines.append(f"  Use when: {when}")
            if caution:
                lines.append(f"  Watch for: {caution}")

    if preference_impact:
        lines.append(f"\n[bold]Preference adjustments:[/]\n{preference_impact}")

    if follow_up:
        lines.append("\n[bold]Follow-up prompts:[/]")
        if isinstance(follow_up, list):
            for idx, item in enumerate(follow_up, start=1):
                lines.append(f"{idx}. {item}")
        else:
            lines.append(str(follow_up))

    content = "\n".join(lines).strip() or "Diagnostics available but empty."
    console.print(Panel(content, title="Selection Diagnostics"))

def render_preference_impacts(impacts: dict[str, Any]) -> None:
    """Display preference-derived score adjustments."""

    techniques = impacts.get("techniques") or []
    categories = impacts.get("categories") or []
    if not techniques and not categories:
        console.print(
            Panel("No preference signals recorded yet.", title="Preference Impacts")
        )
        return

    def _format_entry(entry: dict[str, Any]) -> str:
        name = entry.get("name") or "Unknown"
        adjustment = float(entry.get("adjustment") or 0.0)
        count = int(entry.get("count") or 0)
        average = entry.get("average_rating")
        avg_display = (
            f"{float(average):.2f}" if isinstance(average, (int, float)) else "n/a"
        )
        return f"- {name}: {adjustment:+0.3f} (signals: {count}, avg rating: {avg_display})"

    lines: list[str] = []
    if categories:
        lines.append("[bold]Categories:[/]")
        lines.extend(_format_entry(entry) for entry in categories)

    if techniques:
        if lines:
            lines.append("")
        lines.append("[bold]Techniques:[/]")
        lines.extend(_format_entry(entry) for entry in techniques)

    console.print(Panel("\n".join(lines), title="Preference Impacts"))


def render_explanation_output(result: ExplanationResult) -> None:
    """Render explanation workflow output."""

    lines = []
    if result.overview:
        lines.append(f"[bold]Overview:[/]\n{result.overview}")

    if result.key_factors:
        lines.append("[bold]Key factors:[/]")
        for idx, factor in enumerate(result.key_factors, start=1):
            lines.append(f"{idx}. {factor}")

    if result.risks:
        lines.append("\n[bold]Risks & caveats:[/]")
        for idx, risk in enumerate(result.risks, start=1):
            lines.append(f"{idx}. {risk}")

    if result.next_steps:
        lines.append("\n[bold]Suggested next steps:[/]")
        for idx, step in enumerate(result.next_steps, start=1):
            lines.append(f"{idx}. {step}")

    content = "\n".join(lines) if lines else "No explanation details available."
    console.print(Panel(content, title="Explain Logic"))


def render_simulation_output(simulation: dict[str, Any]) -> None:
    """Render simulation workflow results."""

    if not simulation:
        console.print(Panel("No simulation details available.", title="Simulation"))
        return

    lines: list[str] = []
    overview = simulation.get("simulation_overview")
    if overview:
        lines.append(f"[bold]Simulation overview:[/]\n{overview}")

    variations = simulation.get("scenario_variations") or []
    if variations:
        lines.append("\n[bold]Scenario variations:[/]")
        for entry in variations:
            name = entry.get("name") or "Scenario"
            outcome = entry.get("outcome") or ""
            guidance = entry.get("guidance") or ""
            lines.append(f"- {name}: {outcome}")
            if guidance:
                lines.append(f"  Guidance: {guidance}")

    cautions = simulation.get("cautions") or []
    if cautions:
        lines.append("\n[bold]Cautions:[/]")
        for idx, caution in enumerate(cautions, start=1):
            lines.append(f"{idx}. {caution}")

    follow_up = simulation.get("recommended_follow_up") or []
    if follow_up:
        lines.append("\n[bold]Recommended follow-up:[/]")
        for idx, action in enumerate(follow_up, start=1):
            lines.append(f"{idx}. {action}")

    console.print(Panel("\n".join(lines), title="Simulation"))


def render_comparison_output(comparison: dict[str, Any]) -> None:
    """Render comparison workflow results."""

    if not comparison:
        console.print(Panel("No comparison available.", title="Comparison"))
        return

    lines: list[str] = []
    current = comparison.get("current_recommendation") or "Unknown"
    lines.append(f"[bold]Current recommendation:[/]\n{current}")

    alternative = comparison.get("best_alternative")
    if alternative:
        lines.append(f"\n[bold]Top alternative:[/]\n{alternative}")

    points = comparison.get("comparison_points") or []
    if points:
        lines.append("\n[bold]Comparison points:[/]")
        for point in points:
            technique = point.get("technique") or "Candidate"
            strengths = point.get("strengths") or ""
            risks = point.get("risks") or ""
            best_for = point.get("best_for") or ""
            lines.append(f"- {technique}")
            if strengths:
                lines.append(f"  Strengths: {strengths}")
            if risks:
                lines.append(f"  Risks: {risks}")
            if best_for:
                lines.append(f"  Best for: {best_for}")

    guidance = comparison.get("decision_guidance") or []
    if guidance:
        lines.append("\n[bold]Decision guidance:[/]")
        for idx, tip in enumerate(guidance, start=1):
            lines.append(f"{idx}. {tip}")

    confidence = comparison.get("confidence_notes")
    if confidence:
        lines.append(f"\n[bold]Confidence notes:[/]\n{confidence}")

    console.print(Panel("\n".join(lines), title="Comparison"))


def render_coverage_summary(records: list[dict[str, Any]], *, threshold: int) -> None:
    """Render coverage insights for technique categories."""

    if not records:
        console.print(
            Panel("No technique categories available.", title="Category Coverage")
        )
        return

    table = Table(title=f"Category Coverage (target ≥ {threshold})", show_lines=False)
    table.add_column("Category", style="bold")
    table.add_column("Techniques", justify="right")
    table.add_column("Status")

    show_preferences = any(
        record.get("avg_rating") is not None
        or record.get("negative_ratio") is not None
        for record in records
    )
    if show_preferences:
        table.add_column("Avg Rating", justify="right")
        table.add_column("Negative %", justify="right")

    for record in records:
        category = record.get("category", "Uncategorized")
        count = record.get("count", 0)
        status = record.get("status") or "OK"
        if show_preferences:
            avg_cell = _format_average(record.get("avg_rating"))
            negative_cell = _format_negative_ratio(record.get("negative_ratio"))
            table.add_row(str(category), str(count), status, avg_cell, negative_cell)
        else:
            table.add_row(str(category), str(count), status)

    console.print(table)


def _format_average(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def _format_negative_ratio(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


def render_technique_table(entries: list[dict[str, Any]]) -> None:
    """Render a techniques table."""

    if not entries:
        console.print(Panel("No techniques available.", title="Techniques"))
        return

    table = Table(title="Techniques", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Category")
    table.add_column("Origin Year")
    table.add_column("Creator")
    table.add_column("Description", overflow="fold")

    for entry in entries:
        description = entry.get("description") or ""
        truncated = description if len(description) <= 120 else description[:117] + "..."
        table.add_row(
            entry.get("name") or "",
            entry.get("category") or "-",
            str(entry.get("origin_year") or "-"),
            entry.get("creator") or "-",
            truncated,
        )

    console.print(table)


__all__ = [
    "render_analysis_output",
    "render_candidate_matches",
    "render_comparison_output",
    "render_coverage_summary",
    "render_explanation_output",
    "render_simulation_output",
    "render_technique_table",
]
