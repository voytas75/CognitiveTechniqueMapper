"""Shared console utilities for the Cognitive Technique Mapper CLI."""

from __future__ import annotations

from rich.console import Console

# Single Console instance reused across command modules.
console = Console()

__all__ = ["console"]
