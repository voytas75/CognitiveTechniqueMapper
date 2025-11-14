from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest
import typer

import src.cli as cli
from tests.helpers.cli import StubPreferenceService


class StubConfigService:
    def __init__(self) -> None:
        self.logging_config = {"level": "INFO"}
        self.database_config = {"sqlite_path": ":memory:"}
        self.providers: Dict[str, Dict[str, Any]] = {}

    def iter_workflow_configs(self) -> Dict[str, Any]:
        return {}


class StubSQLiteClient:
    def __init__(self, path: str) -> None:
        self.path = path
        self.initialized = False

    def initialize_schema(self) -> None:
        self.initialized = True


class StubInitializer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True


class StubOrchestrator:
    def __init__(self, workflows: Dict[str, Any]) -> None:
        self.workflows = workflows
        self.executions: list[tuple[str, dict[str, Any]]] = []

    def execute(self, workflow: str, context: dict[str, Any]) -> dict[str, Any]:
        self.executions.append((workflow, context))
        return {"config": {}}


@dataclass
class StubWorkflow:
    dependency: Any

    def run(self, context: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - unused
        return {"context": context}


def test_initialize_runtime_bootstraps_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    initializer = StubInitializer()
    sqlite_client = StubSQLiteClient(":memory:")

    monkeypatch.setattr(cli, "ConfigService", StubConfigService)
    monkeypatch.setattr(cli, "configure_logging", lambda config: None)
    monkeypatch.setattr(cli, "SQLiteClient", lambda path: sqlite_client)
    monkeypatch.setattr(cli, "ChromaClient", None)
    monkeypatch.setattr(cli, "LLMGateway", lambda config_service: object())
    monkeypatch.setattr(cli, "EmbeddingGateway", lambda config_service: object())
    captured_kwargs: dict[str, Any] = {}

    def _initializer_factory(*args: Any, **kwargs: Any) -> StubInitializer:
        captured_kwargs.update(kwargs)
        return initializer

    monkeypatch.setattr(cli, "TechniqueDataInitializer", _initializer_factory)
    monkeypatch.setattr(cli, "PromptService", lambda: object())
    monkeypatch.setattr(cli, "PreferenceRepository", lambda sqlite_client: object())
    monkeypatch.setattr(cli, "PreferenceService", StubPreferenceService)
    monkeypatch.setattr(cli, "TechniqueSelector", lambda **kwargs: object())
    monkeypatch.setattr(cli, "PlanGenerator", lambda **kwargs: object())
    monkeypatch.setattr(cli, "FeedbackManager", lambda: object())
    monkeypatch.setattr(cli, "FeedbackRepository", lambda sqlite_client: object())
    monkeypatch.setattr(cli, "FeedbackService", lambda **kwargs: object())
    monkeypatch.setattr(cli, "ExplanationService", lambda **kwargs: object())
    monkeypatch.setattr(cli, "SimulationService", lambda **kwargs: object())
    monkeypatch.setattr(cli, "ComparisonService", lambda **kwargs: object())
    monkeypatch.setattr(cli, "DetectTechniqueWorkflow", lambda **kwargs: StubWorkflow(kwargs))
    monkeypatch.setattr(cli, "GeneratePlanWorkflow", lambda **kwargs: StubWorkflow(kwargs))
    monkeypatch.setattr(cli, "FeedbackWorkflow", lambda **kwargs: StubWorkflow(kwargs))
    monkeypatch.setattr(cli, "ConfigUpdateWorkflow", lambda **kwargs: StubWorkflow(kwargs))
    monkeypatch.setattr(cli, "SimulateTechniqueWorkflow", lambda **kwargs: StubWorkflow(kwargs))
    monkeypatch.setattr(cli, "CompareCandidatesWorkflow", lambda **kwargs: StubWorkflow(kwargs))
    monkeypatch.setattr(cli, "Orchestrator", lambda workflows: StubOrchestrator(workflows))
    monkeypatch.setattr(cli.AppState, "load", classmethod(lambda cls: cls()))

    orchestrator, state = cli.initialize_runtime()

    assert sqlite_client.initialized is True
    assert isinstance(orchestrator, StubOrchestrator)
    assert state.preference_service is not None
    assert initializer.initialized is True
    assert captured_kwargs["dataset_path"] == cli.PROJECT_ROOT / "data" / "techniques.json"


def test_settings_update_workflow_requires_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ConfigService", lambda: StubConfigService())
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)

    with pytest.raises(typer.BadParameter):
        cli.settings_update_workflow(
            workflow=None,
            model=None,
            temperature=None,
            provider=None,
            max_tokens=None,
            clear_max_tokens=False,
            interactive=False,
        )


def test_settings_update_provider_requires_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ConfigService", lambda: StubConfigService())
    monkeypatch.setattr(cli.console, "print", lambda *args, **kwargs: None)

    with pytest.raises(typer.BadParameter):
        cli.settings_update_provider(
            provider=None,
            api_base=None,
            api_version=None,
            api_key_env=None,
            clear_api_version=False,
            interactive=False,
        )
