from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_configured = False
_handler: RichHandler | None = None


def configure_logging(config: dict[str, Any] | None = None) -> None:
    """Configure application-wide logging with Rich handler."""
    global _configured, _handler
    if _configured:
        return

    config = config or {}
    level_name = str(config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    console = Console()
    handler = RichHandler(
        rich_tracebacks=True,
        console=console,
        markup=True,
        show_time=False,
    )
    handler.setFormatter(logging.Formatter("%(name)s • %(levelname)s • %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    _handler = handler
    _configured = True


def set_runtime_level(level_name: str) -> None:
    """Adjust logging level at runtime."""
    if not level_name:
        return
    level = getattr(logging, level_name.upper(), None)
    if level is None:
        raise ValueError(f"Invalid log level: {level_name}")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if _handler:
        _handler.setLevel(level)
