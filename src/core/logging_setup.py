"""Logging configuration helpers.

Updates:
    v0.1.0 - 2025-11-09 - Added module docstrings and Google-style documentation.
    v0.3.0 - 2025-05-09 - Switched to custom structured JSON logging.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

_configured = False
_handler: logging.Handler | None = None


class JsonFormatter(logging.Formatter):
    """Custom formatter that emits structured JSON log records."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        payload.update(_extract_extra_fields(record))

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(config: dict[str, Any] | None = None) -> None:
    """Configure application-wide logging with structured JSON output.

    Args:
        config (dict[str, Any] | None): Optional logging configuration dictionary.
            Supports a `level` key indicating the minimum log level.
    """

    global _configured, _handler
    if _configured:
        return

    config = config or {}
    level_name = str(config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    _handler = handler
    _configured = True


def _extract_extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key.startswith("_"):
            continue
        if key in {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }:
            continue
        extras[key] = value
    return extras


def set_runtime_level(level_name: str) -> None:
    """Adjust logging level at runtime.

    Args:
        level_name (str): Desired logging level name (e.g., `DEBUG`, `INFO`).

    Raises:
        ValueError: If the level name is not recognized by the logging module.
    """

    if not level_name:
        return
    level = getattr(logging, level_name.upper(), None)
    if level is None:
        raise ValueError(f"Invalid log level: {level_name}")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if _handler:
        _handler.setLevel(level)
