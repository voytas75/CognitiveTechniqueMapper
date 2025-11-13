"""Cognitive Technique Mapper CLI package."""

from __future__ import annotations

import logging
from typing import Any, Optional

import typer

from src.cli.commands.core import (
    analyze,
    compare,
    describe,
    explain,
    feedback,
    report,
    refresh,
    simulate,
)
from src.cli.commands.history import history_clear, history_show
from src.cli.commands.preferences import (
    preferences_export,
    preferences_reset,
    preferences_summary,
)
from src.cli.commands.settings import (
    settings_show,
    settings_update_provider,
    settings_update_workflow,
)
from src.cli.commands.techniques import (
    techniques_add,
    techniques_export,
    techniques_import,
    techniques_list,
    techniques_remove,
    techniques_update,
)
from src.cli.io import console
from src.cli.renderers import (
    render_analysis_output,
    render_candidate_matches,
    render_comparison_output,
    render_explanation_output,
    render_simulation_output,
    render_technique_table,
)
from src.cli.runtime import (
    ChromaClient,
    ConfigEditor,
    ConfigService,
    compose_plan_summary,
    create_catalog_service,
    create_initializer,
    get_orchestrator,
    get_runtime,
    get_state,
    initialize_runtime,
    refresh_runtime,
    set_runtime,
    set_runtime_level,
)
from src.cli.state import AppState, PROJECT_ROOT, STATE_PATH
from src.cli.utils import (
    active_preference_summary,
    apply_log_override,
    infer_category_from_matches,
    prompt_float,
    prompt_int,
    prompt_value,
    refresh_runtime_and_preserve_state,
)
from src.core.feedback_manager import FeedbackManager
from src.core.llm_gateway import LLMGateway
from src.core.logging_setup import configure_logging
from src.core.orchestrator import Orchestrator
from src.db.feedback_repository import FeedbackRepository
from src.db.preference_repository import PreferenceRepository
from src.db.sqlite_client import SQLiteClient
from src.services.comparison_service import ComparisonService
from src.services.data_initializer import TechniqueDataInitializer
from src.services.embedding_gateway import EmbeddingGateway
from src.services.explanation_service import ExplanationResult, ExplanationService
from src.services.feedback_service import FeedbackService
from src.services.plan_generator import PlanGenerator
from src.services.preference_service import PreferenceService
from src.services.prompt_service import PromptService
from src.services.simulation_service import SimulationService
from src.services.technique_catalog import TechniqueCatalogService
from src.services.technique_selector import TechniqueSelector
from src.workflows.compare_candidates import CompareCandidatesWorkflow
from src.workflows.config_update import ConfigUpdateWorkflow
from src.workflows.detect_technique import DetectTechniqueWorkflow
from src.workflows.feedback_loop import FeedbackWorkflow
from src.workflows.generate_plan import GeneratePlanWorkflow
from src.workflows.simulate_technique import SimulateTechniqueWorkflow

logger = logging.getLogger(__name__)

# Backwards-compatible helper wrappers ---------------------------------------


def _apply_log_override(log_level: str | None) -> None:
    apply_log_override(log_level)


def _compose_plan_summary(recommendation: dict[str, Any]) -> str:
    return compose_plan_summary(recommendation)


def _render_analysis_output(
    recommendation: dict[str, Any],
    plan: Any,
    *,
    preference_summary: str | None = None,
    matches: Any = None,
) -> None:
    render_analysis_output(
        recommendation,
        plan,
        preference_summary=preference_summary,
        matches=matches,
    )


def _render_explanation_output(result: ExplanationResult) -> None:
    render_explanation_output(result)


def _render_simulation_output(simulation: dict[str, Any]) -> None:
    render_simulation_output(simulation)


def _render_comparison_output(comparison: dict[str, Any]) -> None:
    render_comparison_output(comparison)


def _active_preference_summary() -> str | None:
    return active_preference_summary()


def _infer_category_from_matches(matches: Any, technique: Optional[str]) -> Optional[str]:
    return infer_category_from_matches(matches, technique)


def _prompt_value(label: str, current: str | None) -> str | None:
    return prompt_value(label, current)


def _prompt_float(label: str, current: float | None) -> float | None:
    return prompt_float(label, current)


def _prompt_int(label: str, current: int | None) -> int | None:
    return prompt_int(label, current)


def _refresh_runtime() -> None:
    refresh_runtime_and_preserve_state()


def _create_initializer() -> tuple[TechniqueDataInitializer, SQLiteClient]:
    return create_initializer()


def _create_catalog_service() -> tuple[TechniqueCatalogService, SQLiteClient]:
    return create_catalog_service()


# Typer applications ---------------------------------------------------------

app = typer.Typer(add_completion=False, help="Cognitive Technique Mapper CLI")
settings_app = typer.Typer(
    add_completion=False, help="Inspect and edit application configuration."
)
techniques_app = typer.Typer(
    add_completion=False, help="Manage techniques in the catalog."
)
history_app = typer.Typer(
    add_completion=False, help="Inspect or clear session history."
)
preferences_app = typer.Typer(
    add_completion=False, help="Inspect personalization preferences."
)


@settings_app.callback(invoke_without_command=True)
def _settings_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        settings_show()


# Command registration -------------------------------------------------------

app.command()(describe)
app.command()(analyze)
app.command()(explain)
app.command()(simulate)
app.command()(compare)
app.command()(refresh)
app.command()(feedback)
app.command()(report)

settings_app.command("show")(settings_show)
settings_app.command("update-workflow")(settings_update_workflow)
settings_app.command("update-provider")(settings_update_provider)

history_app.command("show")(history_show)
history_app.command("clear")(history_clear)

preferences_app.command("summary")(preferences_summary)
preferences_app.command("export")(preferences_export)
preferences_app.command("reset")(preferences_reset)

techniques_app.command("list")(techniques_list)
techniques_app.command("add")(techniques_add)
techniques_app.command("update")(techniques_update)
techniques_app.command("remove")(techniques_remove)
techniques_app.command("export")(techniques_export)
techniques_app.command("import")(techniques_import)

app.add_typer(history_app, name="history")
app.add_typer(preferences_app, name="preferences")
app.add_typer(techniques_app, name="techniques")
app.add_typer(settings_app, name="settings")


def main() -> None:
    """CLI entry point."""

    app()


__all__: list[str] = [
    # Typer apps / entrypoints
    "app",
    "settings_app",
    "techniques_app",
    "history_app",
    "preferences_app",
    "main",
    # Console & logging
    "console",
    "logger",
    # State & runtime
    "AppState",
    "STATE_PATH",
    "PROJECT_ROOT",
    "compose_plan_summary",
    "create_catalog_service",
    "create_initializer",
    "get_orchestrator",
    "get_runtime",
    "get_state",
    "initialize_runtime",
    "refresh_runtime",
    "refresh_runtime_and_preserve_state",
    "set_runtime",
    "set_runtime_level",
    "_apply_log_override",
    "_compose_plan_summary",
    "_render_analysis_output",
    "_render_explanation_output",
    "_render_simulation_output",
    "_render_comparison_output",
    "_active_preference_summary",
    "_infer_category_from_matches",
    "_prompt_value",
    "_prompt_float",
    "_prompt_int",
    "_refresh_runtime",
    "_create_initializer",
    "_create_catalog_service",
    # Commands
    "describe",
    "analyze",
    "explain",
    "simulate",
    "compare",
    "refresh",
    "report",
    "feedback",
    "settings_show",
    "settings_update_workflow",
    "settings_update_provider",
    "history_show",
    "history_clear",
    "preferences_summary",
    "preferences_export",
    "preferences_reset",
    "techniques_list",
    "techniques_add",
    "techniques_update",
    "techniques_remove",
    "techniques_export",
    "techniques_import",
    # Renderers
    "render_analysis_output",
    "render_candidate_matches",
    "render_comparison_output",
    "render_explanation_output",
    "render_simulation_output",
    "render_technique_table",
    # Utilities
    "apply_log_override",
    "active_preference_summary",
    "infer_category_from_matches",
    "prompt_float",
    "prompt_int",
    "prompt_value",
    # External classes re-exported for tests/compatibility
    "ConfigService",
    "configure_logging",
    "SQLiteClient",
    "ChromaClient",
    "LLMGateway",
    "EmbeddingGateway",
    "TechniqueDataInitializer",
    "PromptService",
    "PreferenceRepository",
    "PreferenceService",
    "TechniqueSelector",
    "PlanGenerator",
    "FeedbackManager",
    "FeedbackRepository",
    "FeedbackService",
    "ExplanationService",
    "SimulationService",
    "ComparisonService",
    "DetectTechniqueWorkflow",
    "GeneratePlanWorkflow",
    "FeedbackWorkflow",
    "ConfigUpdateWorkflow",
    "SimulateTechniqueWorkflow",
    "CompareCandidatesWorkflow",
    "Orchestrator",
    "ExplanationResult",
    "TechniqueCatalogService",
]
