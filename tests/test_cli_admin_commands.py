from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import src.cli as cli
from src.services.config_service import WorkflowModelConfig
from tests.helpers.cli import RecordingOrchestrator, mute_console


class StubConfigEditor:
    def __init__(self) -> None:
        self.workflow_updates: list[dict[str, Any]] = []
        self.provider_updates: list[dict[str, Any]] = []

    def update_workflow_model(self, workflow: str, **kwargs: Any) -> None:
        self.workflow_updates.append({"workflow": workflow, **kwargs})

    def update_provider(self, provider: str, **kwargs: Any) -> None:
        self.provider_updates.append({"provider": provider, **kwargs})


class StubConfigService:
    workflows: Dict[str, WorkflowModelConfig] = {}
    providers_data: Dict[str, Dict[str, Any]] = {}
    clear_cache_called = False

    def __init__(self) -> None:
        self._workflows = self.__class__.workflows
        self.providers = self.__class__.providers_data

    def iter_workflow_configs(self) -> Dict[str, WorkflowModelConfig]:
        return self._workflows

    @staticmethod
    def clear_cache() -> None:
        StubConfigService.clear_cache_called = True


@dataclass
class StubInitializer:
    refresh_called: Optional[bool] = None

    def refresh(self, *, rebuild_embeddings: bool) -> None:
        self.refresh_called = rebuild_embeddings


class StubCatalog:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = [
            {
                "name": "Decisional Balance",
                "category": "Decision Making",
                "origin_year": 1960,
                "creator": "Janis",
                "description": "Compare pros and cons",
            }
        ]
        self.add_payloads: list[dict[str, Any]] = []
        self.update_calls: list[tuple[str, dict[str, Any]]] = []
        self.removed: list[str] = []
        self.export_path = Path("techniques.json")
        self.import_summary = {"count": 1}

    def list(self) -> list[dict[str, Any]]:
        return self.entries

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.add_payloads.append(payload)
        return payload

    def update(self, name: str, updates: dict[str, Any]) -> dict[str, Any]:
        self.update_calls.append((name, updates))
        return {"name": name, **updates}

    def remove(self, name: str) -> None:
        self.removed.append(name)

    def export_to_file(self, file: Path) -> tuple[Path, int]:
        return file, len(self.entries)

    def import_from_file(self, file: Path, *, mode: str, rebuild_embeddings: bool) -> dict[str, Any]:
        return {"file": str(file), "mode": mode, "rebuilt": rebuild_embeddings}


class StubSQLiteClient:
    def close(self) -> None:  # pragma: no cover - trivial
        pass


@pytest.fixture()
def patched_console(monkeypatch: pytest.MonkeyPatch) -> None:
    mute_console(monkeypatch)


def test_settings_update_workflow_interactive(monkeypatch: pytest.MonkeyPatch, patched_console: None) -> None:
    StubConfigService.workflows = {
        "detect_technique": WorkflowModelConfig(
            workflow="detect_technique",
            model="gpt-4.1",
            temperature=0.2,
            provider="openai",
            max_tokens=1024,
        )
    }
    StubConfigService.providers_data = {
        "openai": {"api_base": "https://api.example.com", "api_key_env": "OPENAI_API_KEY"}
    }
    monkeypatch.setattr(cli, "ConfigService", StubConfigService)
    editor = StubConfigEditor()
    monkeypatch.setattr(cli, "ConfigEditor", lambda: editor)
    orchestrator = RecordingOrchestrator()
    monkeypatch.setattr(cli, "get_orchestrator", lambda: orchestrator)
    monkeypatch.setattr(cli, "_refresh_runtime", lambda: None)
    monkeypatch.setattr(cli, "_prompt_value", lambda label, current: current or "detect_technique")
    monkeypatch.setattr(cli, "_prompt_float", lambda label, current: 0.6)
    monkeypatch.setattr(cli, "_prompt_int", lambda label, current: 2048)

    cli.settings_update_workflow(
        workflow=None,
        model=None,
        temperature=None,
        provider=None,
        max_tokens=None,
        clear_max_tokens=False,
        interactive=True,
    )

    assert editor.workflow_updates[0]["workflow"] == "detect_technique"
    assert editor.workflow_updates[0]["temperature"] == 0.6
    assert StubConfigService.clear_cache_called


def test_settings_update_provider(monkeypatch: pytest.MonkeyPatch, patched_console: None) -> None:
    StubConfigService.workflows = {}
    StubConfigService.providers_data = {
        "openai": {
            "api_base": "https://api.example.com",
            "api_version": "v1",
            "api_key_env": "OPENAI_API_KEY",
        }
    }
    StubConfigService.clear_cache_called = False
    monkeypatch.setattr(cli, "ConfigService", StubConfigService)
    editor = StubConfigEditor()
    monkeypatch.setattr(cli, "ConfigEditor", lambda: editor)
    monkeypatch.setattr(cli, "get_orchestrator", lambda: RecordingOrchestrator())
    monkeypatch.setattr(cli, "_refresh_runtime", lambda: None)
    monkeypatch.setattr(cli, "_prompt_value", lambda label, current: current or "openai")

    cli.settings_update_provider(
        provider=None,
        api_base=None,
        api_version=None,
        api_key_env=None,
        clear_api_version=False,
        interactive=True,
    )

    assert editor.provider_updates[0]["provider"] == "openai"
    assert StubConfigService.clear_cache_called


def test_refresh_reinitializes_runtime(monkeypatch: pytest.MonkeyPatch, patched_console: None, tmp_path: Path) -> None:
    initializer = StubInitializer()
    sqlite_client = StubSQLiteClient()
    monkeypatch.setattr(cli, "_create_initializer", lambda: (initializer, sqlite_client))
    refresh_called = {"count": 0}
    monkeypatch.setattr(cli, "_refresh_runtime", lambda: refresh_called.__setitem__("count", refresh_called["count"] + 1))

    cli.refresh(rebuild_embeddings=True, log_level=None)

    assert initializer.refresh_called is True
    assert refresh_called["count"] == 1


def test_techniques_commands(monkeypatch: pytest.MonkeyPatch, patched_console: None, tmp_path: Path) -> None:
    catalog = StubCatalog()
    sqlite_client = StubSQLiteClient()
    monkeypatch.setattr(cli, "_create_catalog_service", lambda: (catalog, sqlite_client))
    monkeypatch.setattr(cli, "_refresh_runtime", lambda: None)

    cli.techniques_list()
    cli.techniques_add(
        name="Force Field Analysis",
        description="Analyze driving and restraining forces",
        origin_year=1951,
        creator="Lewin",
        category="Change Management",
        core_principles="List forces",
    )
    cli.techniques_update(
        name="Decisional Balance",
        new_name="Balanced Decision",
        description="Updated",
    )
    cli.techniques_remove(name="Balanced Decision")


def test_techniques_refresh(monkeypatch: pytest.MonkeyPatch, patched_console: None) -> None:
    initializer = StubInitializer()
    sqlite_client = StubSQLiteClient()
    monkeypatch.setattr(cli, "_create_initializer", lambda: (initializer, sqlite_client))
    refresh_calls = {"count": 0}
    monkeypatch.setattr(
        cli,
        "_refresh_runtime",
        lambda: refresh_calls.__setitem__("count", refresh_calls["count"] + 1),
    )

    cli.techniques_refresh(rebuild_embeddings=False, log_level=None)

    assert initializer.refresh_called is False
    assert refresh_calls["count"] == 1
