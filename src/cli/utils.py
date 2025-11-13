"""Utility helpers shared across CLI command modules."""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import typer

logger = logging.getLogger(__name__)


def _cli():
    return sys.modules["src.cli"]


def apply_log_override(log_level: Optional[str]) -> None:
    """Override logging level for the current invocation."""

    if not log_level or not isinstance(log_level, str):
        return
    try:
        _cli().set_runtime_level(log_level)
        logger.info("Log level overridden to %s", log_level.upper())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def active_preference_summary() -> Optional[str]:
    """Return the active preference summary when available."""

    state = _cli().get_state()
    if not state.preference_service:
        return None
    summary = state.preference_service.preference_summary()
    return summary or None


def infer_category_from_matches(matches: Any, technique: Optional[str]) -> Optional[str]:
    """Inspect candidate matches to infer the category for a technique."""

    if not technique or not isinstance(matches, list):
        return None
    target = technique.lower()
    for match in matches:
        if not isinstance(match, dict):
            continue
        metadata = match.get("metadata") or {}
        name = metadata.get("name") or match.get("id")
        if isinstance(name, str) and name.lower() == target:
            category = metadata.get("category") or match.get("category")
            if isinstance(category, str) and category.strip():
                return category
    return None


def prompt_value(label: str, current: str | None) -> str | None:
    """Prompt for a string value with an optional default."""

    default_display = current if current is not None else ""
    response = typer.prompt(label, default=default_display)
    return response.strip() or current


def prompt_float(label: str, current: float | None) -> float | None:
    """Prompt for a floating-point value with validation."""

    default_display = "" if current is None else str(current)
    response = typer.prompt(label, default=default_display).strip()
    if not response:
        return current
    try:
        return float(response)
    except ValueError as exc:  # pragma: no cover - input validation path
        raise typer.BadParameter(f"Invalid float for {label}: {response}") from exc


def prompt_int(label: str, current: int | None) -> int | None:
    """Prompt for an integer value with validation."""

    default_display = "" if current is None else str(current)
    response = typer.prompt(label, default=default_display).strip()
    if not response:
        return current
    try:
        return int(response)
    except ValueError as exc:  # pragma: no cover - input validation path
        raise typer.BadParameter(f"Invalid integer for {label}: {response}") from exc


def refresh_runtime_and_preserve_state() -> None:
    """Refresh runtime dependencies while maintaining session state."""

    _cli().refresh_runtime()

__all__ = [
    "active_preference_summary",
    "apply_log_override",
    "infer_category_from_matches",
    "prompt_float",
    "prompt_int",
    "prompt_value",
    "refresh_runtime_and_preserve_state",
]
