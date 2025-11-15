"""Preference management commands."""

from __future__ import annotations

from dataclasses import asdict
import sys

import typer
from rich.panel import Panel

from src.cli.io import console
from src.cli.renderers import render_preference_impacts


def _cli():
    return sys.modules["src.cli"]


def preferences_summary() -> None:
    """Show a human-readable summary of recorded preferences."""

    state = _cli().get_state()
    if not state.preference_service:
        console.print("[yellow]Preference service unavailable.[/]")
        return

    summary = state.preference_service.preference_summary()
    if not summary:
        console.print(Panel("No preference signals recorded yet.", title="Preferences"))
        return

    console.print(Panel(summary, title="Preference Summary"))


def preferences_export() -> None:
    """Export the full preference profile as JSON."""

    state = _cli().get_state()
    if not state.preference_service:
        console.print("[yellow]Preference service unavailable.[/]")
        return

    profile = state.preference_service.export_profile()
    console.print_json(data=asdict(profile))


def preferences_reset(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Reset preferences without a confirmation prompt.",
    )
) -> None:
    """Remove all stored feedback-based preferences."""

    state = _cli().get_state()
    if not state.preference_service:
        console.print("[yellow]Preference service unavailable.[/]")
        return

    if not force and not typer.confirm("Clear all stored preferences?"):
        console.print("[yellow]Preferences unchanged.[/]")
        return

    state.preference_service.clear()
    console.print("[green]Preferences cleared.[/]")

def preferences_impact(
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Maximum number of entries to display per section.",
    )
) -> None:
    """Show score adjustments derived from stored preferences."""

    state = _cli().get_state()
    if not state.preference_service:
        console.print("[yellow]Preference service unavailable.[/]")
        return

    impacts = state.preference_service.preference_impacts(limit=limit)
    render_preference_impacts(impacts)


__all__ = [
    "preferences_export",
    "preferences_impact",
    "preferences_reset",
    "preferences_summary",
]
