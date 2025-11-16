"""FastAPI application exposing Cognitive Technique Mapper workflows.

This package wires the existing orchestrator and workflows into a small REST and
GraphQL service so that other tools can trigger the same reasoning pipelines
without invoking the CLI directly.

All public modules are re‑exported via ``src.api`` for convenience::

    from src.api import app  # FastAPI instance

Run locally with::

    uvicorn src.api:app --reload

Updates:
    v0.1.0 - 2025-11-16 - Initial public API exposing ``/health`` and
    ``/workflow/{name}`` endpoints plus optional GraphQL overlay.
"""

from __future__ import annotations

from .app import app  # noqa: F401  re‑export for ``uvicorn src.api:app``

