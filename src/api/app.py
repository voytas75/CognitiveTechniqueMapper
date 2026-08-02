"""Local-only FastAPI surface for Cognitive Technique Mapper.

This module is a loopback development convenience, not a deployment boundary:
request handlers translate JSON into calls to the registered *Orchestrator*
workflows. It has no authentication or cross-origin contract and must be bound
to ``127.0.0.1``.

The HTTP surface contains only:

1. ``GET /health`` – local liveness endpoint.
2. ``GET /workflows`` – registered orchestrator workflow names.
3. ``POST /workflow/{name}`` – execute one registered workflow with JSON
   context.

CLI-only flows, including ``explain_logic``, are intentionally not HTTP
workflows. Optional GraphQL follows the same local-only restriction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, cast

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.cli.runtime import initialize_runtime

# Initialize the module-level uvicorn application once. Runtime setup performs
# SQLite schema checks and optional Chroma initialization.
_orchestrator, _state = initialize_runtime()

logger = logging.getLogger(__name__)


def create_app(orchestrator: Any | None = None) -> FastAPI:
    """Create the local-only API around an orchestrator instance.

    Args:
        orchestrator: Optional workflow orchestrator override for HTTP tests.

    Returns:
        FastAPI application exposing registered workflows only.
    """

    active_orchestrator = _orchestrator if orchestrator is None else orchestrator
    application = FastAPI(title="Cognitive Technique Mapper API", version="0.1.0")

    async def health() -> Dict[str, str]:
        """Return local liveness state."""

        return {"status": "ok"}

    async def list_workflows() -> List[str]:
        """Return registered HTTP workflow names."""

        return list(active_orchestrator.workflows.keys())

    async def execute_workflow(workflow_name: str, request: Request) -> JSONResponse:  # noqa: D401 – FastAPI handler signature
        """Execute a registered workflow with a JSON context body."""

        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover - framework parsing path
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body must be valid JSON.",
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body must be a JSON object.",
            )
        context: Dict[str, Any] = payload

        logger.info("api_execute_workflow", extra={"workflow": workflow_name})

        try:
            result = active_orchestrator.execute(workflow_name, context)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown workflow '{workflow_name}'",
            ) from None
        except Exception as exc:
            logger.exception("workflow_execution_failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow execution failed.",
            ) from exc

        return JSONResponse(content=result)

    application.add_api_route("/health", health, methods=["GET"], tags=["system"])
    application.add_api_route(
        "/workflows", list_workflows, methods=["GET"], tags=["workflows"]
    )
    application.add_api_route(
        "/workflow/{workflow_name}",
        execute_workflow,
        methods=["POST"],
        response_class=JSONResponse,
        tags=["workflows"],
    )

    # Optional GraphQL overlay ------------------------------------------------
    if _graphql_router is not None:  # noqa: WPS505 – explicit guard
        application.include_router(_graphql_router, prefix="/graphql")

    return application


# Module‑level instance so that ``uvicorn src.api:app`` works out‑of‑the‑box.

# ---------------------------------------------------------------------------
# Conditional GraphQL schema creation (module‑level so Strawberry can resolve
# type annotations properly). We must execute this *before* creating the FastAPI
# instance to avoid the unresolved field type error raised when the Query class
# is nested inside the factory function.
# ---------------------------------------------------------------------------

try:
    import strawberry  # type: ignore
    from strawberry.fastapi import GraphQLRouter

except ImportError:  # pragma: no cover – strawberry optional
    _graphql_router = None
else:
    JSONScalar = strawberry.scalars.JSON  # noqa: N816 – GraphQL scalar alias

    @strawberry.type
    class Query:  # noqa: WPS110 – GraphQL root type has conventional name
        """GraphQL root query type."""

        @strawberry.field
        def workflows(self) -> List[str]:  # noqa: D401 – GraphQL resolver
            """Return available workflow names."""

            return list(_orchestrator.workflows.keys())

        @strawberry.field
        def run_workflow(
            self, name: str, context: strawberry.scalars.JSON
        ) -> strawberry.scalars.JSON:
            """Execute *name* workflow and return its JSON result."""

            raw_context = cast(object, context)
            if not isinstance(raw_context, dict) or not all(
                isinstance(key, str) for key in raw_context
            ):
                raise ValueError("Workflow context must be a JSON object.")
            workflow_context = cast(Dict[str, Any], raw_context)
            try:
                return cast(
                    strawberry.scalars.JSON,
                    _orchestrator.execute(name, workflow_context),
                )
            except KeyError as exc:  # pragma: no cover – mapping error
                raise ValueError(str(exc)) from exc

    _schema = strawberry.Schema(query=Query)
    _graphql_router = GraphQLRouter(_schema)


# Build the FastAPI app after the optional GraphQL router is ready.
app: FastAPI = create_app()
