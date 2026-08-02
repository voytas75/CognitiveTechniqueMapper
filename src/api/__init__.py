"""FastAPI application exposing Cognitive Technique Mapper workflows.

This package wraps registered orchestrator workflows in a local loopback REST and
GraphQL service for development use. It is not a production deployment
interface and must remain bound to ``127.0.0.1``.

All public modules are re‑exported via ``src.api`` for convenience::

    from src.api import app  # FastAPI instance

Run locally with::

    uv run --frozen uvicorn src.api:app --reload --host 127.0.0.1

Updates:
    v0.1.0 - 2025-11-16 - Initial public API exposing ``/health`` and
    ``/workflow/{name}`` endpoints plus optional GraphQL overlay.
"""

from __future__ import annotations

from .app import app

__all__ = ["app"]
