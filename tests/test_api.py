"""HTTP contract tests for the local-only CTM API."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.cli import runtime as runtime_module


class _ImportOrchestrator:
    """Minimal default used while importing the module-level ASGI app in tests."""

    workflows: dict[str, object] = {}


with patch.object(
    runtime_module,
    "initialize_runtime",
    return_value=(_ImportOrchestrator(), object()),
):
    from src.api.app import create_app


class StubOrchestrator:
    """Minimal orchestrator double for HTTP contract tests."""

    def __init__(self) -> None:
        self.workflows = {"detect_technique": object()}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, workflow_name: str, context: dict[str, Any]) -> dict[str, Any]:
        """Record and return a deterministic workflow result."""

        if workflow_name not in self.workflows:
            raise KeyError(workflow_name)
        self.calls.append((workflow_name, context))
        return {"workflow": workflow_name, "context": context}


class DetectTechniqueOrchestrator(StubOrchestrator):
    """Workflow double enforcing the real detect-technique input contract."""

    def execute(self, workflow_name: str, context: dict[str, Any]) -> dict[str, Any]:
        if workflow_name == "detect_technique" and not isinstance(
            context.get("problem_description"), str
        ):
            raise ValueError("Context missing 'problem_description'.")
        return super().execute(workflow_name, context)


class FailingOrchestrator(StubOrchestrator):
    """Orchestrator double that models an internal workflow failure."""

    def execute(self, workflow_name: str, context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("provider secret must not be exposed")


def _client(
    orchestrator: StubOrchestrator,
    *,
    client_address: tuple[str, int] = ("127.0.0.1", 50000),
) -> Any:
    """Return an untyped HTTP client until FastAPI's TestClient ships stubs."""

    return TestClient(create_app(orchestrator=orchestrator), client=client_address)


def test_local_api_exposes_registered_workflows_without_cors() -> None:
    """The HTTP surface accepts only orchestrated workflows for local callers."""
    orchestrator = DetectTechniqueOrchestrator()
    client = _client(orchestrator)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/workflows").json() == ["detect_technique"]

    response = client.post(
        "/workflow/detect_technique",
        json={"problem_description": "Prioritize two projects."},
        headers={"Origin": "https://untrusted.example"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "workflow": "detect_technique",
        "context": {"problem_description": "Prioritize two projects."},
    }
    assert response.headers.get("access-control-allow-origin") is None
    assert orchestrator.calls == [
        ("detect_technique", {"problem_description": "Prioritize two projects."})
    ]

    assert client.post("/workflow/explain_logic", json={}).status_code == 404


def test_local_api_rejects_non_loopback_clients_for_rest_and_graphql() -> None:
    """Remote peer addresses must not reach any HTTP or GraphQL API route."""

    client = _client(StubOrchestrator(), client_address=("203.0.113.5", 50000))

    for path in ("/health", "/graphql"):
        response = client.get(path)
        assert response.status_code == 403
        assert response.json() == {"detail": "Loopback clients only."}


def test_local_api_rejects_invalid_workflow_context() -> None:
    """Validation failures become client errors without leaking implementation details."""
    client = _client(DetectTechniqueOrchestrator())

    response = client.post(
        "/workflow/detect_technique", json={"problem": "Prioritize two projects."}
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid workflow context."}


def test_local_api_rejects_non_object_json_context() -> None:
    """Workflow contexts must be JSON objects rather than arbitrary JSON values."""

    orchestrator = StubOrchestrator()
    client = _client(orchestrator)

    response = client.post("/workflow/detect_technique", json=["not", "a context"])

    assert response.status_code == 400
    assert response.json() == {"detail": "Request body must be a JSON object."}
    assert orchestrator.calls == []


def test_local_api_hides_internal_workflow_errors() -> None:
    """Workflow failures remain in server logs rather than client responses."""

    client = _client(FailingOrchestrator())

    response = client.post("/workflow/detect_technique", json={})

    assert response.status_code == 500
    assert response.json() == {"detail": "Workflow execution failed."}
    assert "provider secret" not in response.text
