"""History commands for the Cognitive Technique Mapper CLI."""

from __future__ import annotations

import json
import sys

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

    for offset, entry in enumerate(subset, start=1):
        event_number = start_index + offset
        console.print(
            Panel(
                json.dumps(entry, ensure_ascii=False, indent=2),
                title=f"Event {event_number}",
            )
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
