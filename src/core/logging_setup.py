from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_configured = False


def configure_logging(config: dict[str, Any] | None = None) -> None:
    """Configure application-wide logging with Rich handler."""
    global _configured
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

    _configured = True
