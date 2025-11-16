"""History commands for the Cognitive Technique Mapper CLI."""

from __future__ import annotations

import json
import sys
from textwrap import shorten
from typing import Any, Sequence

import typer
from rich.panel import Panel

from src.cli.io import console


def _cli():
    return sys.modules["src.cli"]


def history_show(
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        min=0,
        help="Number of most recent history entries to display (0 = all).",
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Emit raw JSON instead of rendered panels.",
    ),
) -> None:
    """Display recent session history captured by the CLI."""

    state = _cli().get_state()
    entries = state.context_history
    if not entries:
        console.print(Panel("History is empty.", title="History"))
        return

    subset = entries if limit == 0 else entries[-limit:]
    start_index = len(entries) - len(subset)

    if raw:
        console.print_json(data=subset)
        return

    printed_entries = False
    for offset, entry in enumerate(subset, start=1):
        event_number = start_index + offset
        label, summary = _summarize_history_entry(entry)
        console.print(
            Panel(
                summary,
                title=f"Event {event_number} \u2022 {label}",
                expand=False,
            )
        )
        printed_entries = True

    if printed_entries:
        console.print(
            "[dim]Tip: rerun with `history show --raw` to inspect full payloads.[/]"
        )


def history_clear(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Clear without confirmation prompt.",
    )
) -> None:
    """Erase the stored session history."""

    state = _cli().get_state()
    if not state.context_history:
        console.print("[yellow]History is already empty.[/]")
        return

    if not force and not typer.confirm("Clear all history entries?"):
        console.print("[yellow]History unchanged.[/]")
        return

    state.context_history.clear()
    state.save()
    console.print("[green]History cleared.[/]")


__all__ = ["history_clear", "history_show"]


def _summarize_history_entry(entry: dict[str, Any]) -> tuple[str, str]:
    """Return a label and human-readable summary for a history entry."""

    if "problem_description" in entry:
        description = _coerce_string(entry.get("problem_description")) or "(empty)"
        return "Describe", f"[bold]Problem:[/] {description}"

    recommendation = entry.get("recommendation")
    if isinstance(recommendation, dict) and recommendation:
        return _summarize_analysis_entry(entry, recommendation)

    explanation = entry.get("explanation")
    if isinstance(explanation, dict) and explanation:
        return _summarize_explanation_entry(explanation)

    simulation = entry.get("simulation")
    if isinstance(simulation, dict) and simulation:
        return _summarize_simulation_entry(simulation)

    comparison = entry.get("comparison")
    if isinstance(comparison, dict) and comparison:
        return _summarize_comparison_entry(comparison)

    return "Context", json.dumps(entry, ensure_ascii=False, indent=2)


def _summarize_analysis_entry(
    entry: dict[str, Any], recommendation: dict[str, Any]
) -> tuple[str, str]:
    """Summarize an analyze workflow entry."""

    technique = _coerce_string(
        recommendation.get("suggested_technique")
        or recommendation.get("technique")
        or recommendation.get("technique_name")
    )
    reason = _shorten_text(_coerce_string(recommendation.get("why_it_fits")))
    plan = entry.get("plan") if isinstance(entry.get("plan"), dict) else None
    plan_steps = _string_list(plan.get("milestones")) if plan else []
    matches = entry.get("matches") if isinstance(entry.get("matches"), list) else []
    preference_summary = _shorten_text(_coerce_string(entry.get("preference_summary")))

    lines = []
    if technique:
        lines.append(f"[bold]Technique:[/] {technique}")
    if reason:
        lines.append(f"[bold]Why it fits:[/] {reason}")
    if plan_steps:
        lines.append(f"[bold]Plan steps:[/] {_summarize_list(plan_steps)}")
    if matches:
        top_match = matches[0]
        top_name = None
        if isinstance(top_match, dict):
            metadata = top_match.get("metadata")
            if isinstance(metadata, dict):
                top_name = _coerce_string(metadata.get("name"))
        shortlist_line = f"[bold]Shortlist:[/] {len(matches)} candidate(s)"
        if top_name:
            shortlist_line += f" (top: {top_name})"
        lines.append(shortlist_line)
    if preference_summary:
        lines.append(f"[bold]Preferences:[/] {preference_summary}")

    if not lines:
        lines.append("[dim]No summary available.[/]")
    return "Analyze", "\n".join(lines)


def _summarize_explanation_entry(explanation: dict[str, Any]) -> tuple[str, str]:
    """Summarize an explanation workflow entry."""

    overview = _shorten_text(_coerce_string(explanation.get("overview")))
    key_factors = _summarize_list(_string_list(explanation.get("key_factors")))
    risks = _summarize_list(_string_list(explanation.get("risks")))
    next_steps = _summarize_list(_string_list(explanation.get("next_steps")))

    lines: list[str] = []
    if overview:
        lines.append(f"[bold]Overview:[/] {overview}")
    if key_factors:
        lines.append(f"[bold]Key factors:[/] {key_factors}")
    if risks:
        lines.append(f"[bold]Risks:[/] {risks}")
    if next_steps:
        lines.append(f"[bold]Next steps:[/] {next_steps}")
    if not lines:
        lines.append("[dim]No explanation details available.[/]")
    return "Explain", "\n".join(lines)


def _summarize_simulation_entry(simulation: dict[str, Any]) -> tuple[str, str]:
    """Summarize a simulation workflow entry."""

    overview = _shorten_text(_coerce_string(simulation.get("simulation_overview")))
    variations = _format_variations(simulation.get("scenario_variations"))
    cautions = _summarize_list(_string_list(simulation.get("cautions")))
    follow_up = _summarize_list(_string_list(simulation.get("recommended_follow_up")))

    lines: list[str] = []
    if overview:
        lines.append(f"[bold]Overview:[/] {overview}")
    if variations:
        preview = "\n".join(f"• {item}" for item in variations[:2])
        if len(variations) > 2:
            preview += "\n• …"
        lines.append(f"[bold]Variations:[/]\n{preview}")
    if cautions:
        lines.append(f"[bold]Cautions:[/] {cautions}")
    if follow_up:
        lines.append(f"[bold]Follow-up:[/] {follow_up}")
    if not lines:
        lines.append("[dim]No simulation details available.[/]")
    return "Simulate", "\n".join(lines)


def _summarize_comparison_entry(comparison: dict[str, Any]) -> tuple[str, str]:
    """Summarize a comparison workflow entry."""

    current = _coerce_string(comparison.get("current_recommendation"))
    alternative = _coerce_string(comparison.get("best_alternative"))
    guidance = _summarize_list(_string_list(comparison.get("decision_guidance")))
    confidence = _shorten_text(_coerce_string(comparison.get("confidence_notes")))
    points = comparison.get("comparison_points")
    point_count = len(points) if isinstance(points, list) else 0

    lines: list[str] = []
    if current:
        lines.append(f"[bold]Current:[/] {current}")
    if alternative:
        lines.append(f"[bold]Best alternative:[/] {alternative}")
    if point_count:
        lines.append(f"[bold]Comparison points:[/] {point_count}")
    if guidance:
        lines.append(f"[bold]Decision guidance:[/] {guidance}")
    if confidence:
        lines.append(f"[bold]Confidence:[/] {confidence}")
    if not lines:
        lines.append("[dim]No comparison details available.[/]")
    return "Compare", "\n".join(lines)


def _shorten_text(value: str | None, width: int = 120) -> str | None:
    """Shorten long text segments for display."""

    if not value:
        return None
    return shorten(value, width=width, placeholder="…")


def _string_list(value: Any) -> list[str]:
    """Coerce a value into a list of non-empty strings."""

    if isinstance(value, list):
        cleaned: list[str] = []
        for item in value:
            text = None
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = _coerce_string(
                    item.get("name")
                    or item.get("title")
                    or item.get("description")
                    or item.get("technique")
                )
            else:
                text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _summarize_list(items: Sequence[str], limit: int = 3) -> str:
    """Return a comma-separated preview of a list."""

    if not items:
        return ""
    preview = ", ".join(items[:limit])
    if len(items) > limit:
        preview += ", …"
    return preview


def _format_variations(value: Any) -> list[str]:
    """Create readable scenario variation summaries."""

    if not isinstance(value, list):
        return []
    variations: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = _coerce_string(item.get("name")) or "Scenario"
            outcome = _coerce_string(item.get("outcome"))
            guidance = _coerce_string(item.get("guidance"))
            descriptor = outcome or guidance
            if descriptor:
                variations.append(f"{name}: {descriptor}")
            else:
                variations.append(name)
        else:
            text = _coerce_string(item)
            if text:
                variations.append(text)
    return variations


def _coerce_string(value: Any) -> str | None:
    """Convert a value to a trimmed string if possible."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    text = str(value).strip()
    return text or None
