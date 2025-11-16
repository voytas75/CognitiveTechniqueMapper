"""FastAPI surface for Cognitive Technique Mapper.

The design intentionally keeps the HTTP layer very thin: request handlers simply
translate incoming JSON payloads into calls to the existing *Orchestrator*
instance returned by ``initialize_runtime``.

Only two routes are required for an MVP that unblocks external automation:

1. ``GET /health`` – lightweight liveness endpoint.
2. ``POST /workflow/{name}`` – execute a registered workflow with the JSON
   request body as *context* and return the workflow response.

An additional ``GET /workflows`` route lists available workflow names.

GraphQL support powered by *strawberry* is enabled if the library is
importable.  This is entirely optional and incurs zero runtime overhead when
the dependency is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.cli.runtime import initialize_runtime

# Lazily initialise runtime on first import. This is inexpensive after the
# heavy lifting (SQLite schema checks, Chroma init) is done once.
_orchestrator, _state = initialize_runtime()

logger = logging.getLogger(__name__)


def _create_app() -> FastAPI:  # pragma: no cover – simple factory
    """Factory assembling the FastAPI application."""

    application = FastAPI(title="Cognitive Technique Mapper API", version="0.1.0")

    # Allow typical local dev tools and browser playgrounds.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health", tags=["system"])
    async def health() -> Dict[str, str]:
        """Liveness probe used by orchestrators and load balancers."""

        return {"status": "ok"}

    @application.get("/workflows", tags=["workflows"])
    async def list_workflows() -> List[str]:
        """Return the names of registered workflows."""

        return list(_orchestrator.workflows.keys())

    @application.post("/workflow/{workflow_name}", response_class=JSONResponse, tags=["workflows"])
    async def execute_workflow(
        workflow_name: str, request: Request
    ) -> JSONResponse:  # noqa: D401 – FastAPI handler signature
        """Execute *workflow_name* with JSON body forwarded as *context*."""

        try:
            context: Dict[str, Any] = await request.json()
        except Exception as exc:  # pragma: no cover – FastAPI already validates JSON
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        logger.info("api_execute_workflow", extra={"workflow": workflow_name})

        try:
            result = _orchestrator.execute(workflow_name, context)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown workflow '{workflow_name}'",
            ) from None
        except Exception as exc:
            logger.exception("workflow_execution_failed")
            raise HTTPException(
                status_code=500, detail=f"Workflow error: {exc}"
            ) from exc

        return JSONResponse(content=result)

    # Optional GraphQL overlay ------------------------------------------------
    if _GRAPHQL_ROUTER is not None:  # noqa: WPS505 – explicit guard
        application.include_router(_GRAPHQL_ROUTER, prefix="/graphql")

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
    _GRAPHQL_ROUTER = None
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
        def run_workflow(self, name: str, context: JSONScalar) -> JSONScalar:  # type: ignore  # noqa: E501
            """Execute *name* workflow and return its JSON result."""

            try:
                return _orchestrator.execute(name, context)
            except KeyError as exc:  # pragma: no cover – mapping error
                raise ValueError(str(exc)) from exc

    _schema = strawberry.Schema(query=Query)
    _GRAPHQL_ROUTER = GraphQLRouter(_schema)


# Build the FastAPI app *after* GraphQL router is ready.
app: FastAPI = _create_app()
