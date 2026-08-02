"""Runtime wiring for the Cognitive Technique Mapper CLI."""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

from src.cli.service_factories import (
    create_catalog_service,
    create_initializer,
    create_optional_chroma_client,
    create_search_service,
)
from src.cli.state import PROJECT_ROOT, AppState
from src.core.feedback_manager import FeedbackManager
from src.core.llm_gateway import LLMGateway
from src.core.logging_setup import (
    configure_logging,
    set_runtime_level,  # re-export via utils
)
from src.core.orchestrator import Orchestrator
from src.db.feedback_repository import FeedbackRepository
from src.db.preference_repository import PreferenceRepository
from src.db.sqlite_client import SQLiteClient
from src.services.comparison_service import ComparisonService
from src.services.config_editor import ConfigEditor
from src.services.config_service import ConfigService
from src.services.data_initializer import TechniqueDataInitializer
from src.services.embedding_gateway import EmbeddingGateway
from src.services.explanation_service import ExplanationService
from src.services.feedback_service import FeedbackService
from src.services.plan_generator import PlanGenerator
from src.services.preference_service import PreferenceService
from src.services.prompt_service import PromptService
from src.services.simulation_service import SimulationService
from src.services.technique_selector import TechniqueSelector
from src.workflows.compare_candidates import CompareCandidatesWorkflow
from src.workflows.config_update import ConfigUpdateWorkflow
from src.workflows.detect_technique import DetectTechniqueWorkflow
from src.workflows.feedback_loop import FeedbackWorkflow
from src.workflows.generate_plan import GeneratePlanWorkflow
from src.workflows.simulate_technique import SimulateTechniqueWorkflow

try:
    from src.db.chroma_client import ChromaClient as _DEFAULT_CHROMA_CLIENT
except RuntimeError:
    _DEFAULT_CHROMA_CLIENT = None  # type: ignore[assignment]

# Re-exported for compatibility; may be monkeypatched by tests.
ChromaClient = _DEFAULT_CHROMA_CLIENT

logger = logging.getLogger(__name__)

_runtime_cache: tuple[Orchestrator, AppState] | None = None

_DEFAULT_CONFIG_SERVICE = ConfigService
_DEFAULT_SQLITE_CLIENT = SQLiteClient
_DEFAULT_LLM_GATEWAY = LLMGateway
_DEFAULT_EMBEDDING_GATEWAY = EmbeddingGateway
_DEFAULT_INITIALIZER = TechniqueDataInitializer
_DEFAULT_PROMPT_SERVICE = PromptService
_DEFAULT_PREFERENCE_REPOSITORY = PreferenceRepository
_DEFAULT_PREFERENCE_SERVICE = PreferenceService
_DEFAULT_TECHNIQUE_SELECTOR = TechniqueSelector
_DEFAULT_PLAN_GENERATOR = PlanGenerator
_DEFAULT_FEEDBACK_MANAGER = FeedbackManager
_DEFAULT_FEEDBACK_REPOSITORY = FeedbackRepository
_DEFAULT_FEEDBACK_SERVICE = FeedbackService
_DEFAULT_EXPLANATION_SERVICE = ExplanationService
_DEFAULT_SIMULATION_SERVICE = SimulationService
_DEFAULT_COMPARISON_SERVICE = ComparisonService
_DEFAULT_DETECT_WORKFLOW = DetectTechniqueWorkflow
_DEFAULT_PLAN_WORKFLOW = GeneratePlanWorkflow
_DEFAULT_FEEDBACK_WORKFLOW = FeedbackWorkflow
_DEFAULT_CONFIG_UPDATE_WORKFLOW = ConfigUpdateWorkflow
_DEFAULT_SIMULATE_WORKFLOW = SimulateTechniqueWorkflow
_DEFAULT_COMPARE_WORKFLOW = CompareCandidatesWorkflow


def compose_plan_summary(recommendation: dict[str, Any]) -> str:
    """Compose a concise summary for the plan generator from a recommendation."""

    technique = recommendation.get("suggested_technique") or ""
    why_it_fits = recommendation.get("why_it_fits") or ""
    steps = recommendation.get("steps")
    step_items = cast(list[object], steps) if isinstance(steps, list) else []
    joined_steps = "; ".join(str(step) for step in step_items if step)
    segments = [
        f"Technique: {technique}" if technique else "",
        f"Why it fits: {why_it_fits}" if why_it_fits else "",
        f"Suggested steps: {joined_steps}" if joined_steps else "",
    ]
    return "\n".join(segment for segment in segments if segment)


def initialize_runtime() -> tuple[Orchestrator, AppState]:
    """Initialize core services and hydrated state for CLI usage."""

    config_service_cls = _resolve_dependency("ConfigService", _DEFAULT_CONFIG_SERVICE)
    config_service = config_service_cls()
    configure_logging(config_service.logging_config)
    logger.debug("Runtime initialization starting.")
    db_config = config_service.database_config

    sqlite_cls = _resolve_dependency("SQLiteClient", _DEFAULT_SQLITE_CLIENT)
    sqlite_client = sqlite_cls(db_config.get("sqlite_path", "./data/techniques.db"))
    sqlite_client.initialize_schema()

    chroma_client = _initialize_chroma_client(
        db_config.get("chromadb_path", "./embeddings"),
        db_config.get("chromadb_collection", "techniques"),
    )

    llm_gateway_cls = _resolve_dependency("LLMGateway", _DEFAULT_LLM_GATEWAY)
    llm_gateway = llm_gateway_cls(config_service=config_service)
    embedding_gateway_cls = _resolve_dependency(
        "EmbeddingGateway", _DEFAULT_EMBEDDING_GATEWAY
    )
    embedding_gateway = embedding_gateway_cls(config_service=config_service)
    initializer_cls = _resolve_dependency(
        "TechniqueDataInitializer", _DEFAULT_INITIALIZER
    )
    dataset_path = PROJECT_ROOT / "data" / "techniques.json"
    initializer = initializer_cls(
        sqlite_client=sqlite_client,
        embedder=embedding_gateway,
        chroma_client=chroma_client,
        dataset_path=dataset_path,
    )
    initializer.initialize()
    logger.debug("Initialization completed (Chroma enabled=%s).", bool(chroma_client))

    prompt_service_cls = _resolve_dependency("PromptService", _DEFAULT_PROMPT_SERVICE)
    prompt_service = prompt_service_cls()
    preference_repository_cls = _resolve_dependency(
        "PreferenceRepository", _DEFAULT_PREFERENCE_REPOSITORY
    )
    preference_repository = preference_repository_cls(sqlite_client=sqlite_client)
    preference_service_cls = _resolve_dependency(
        "PreferenceService", _DEFAULT_PREFERENCE_SERVICE
    )
    preference_service = preference_service_cls(repository=preference_repository)

    technique_selector_cls = _resolve_dependency(
        "TechniqueSelector", _DEFAULT_TECHNIQUE_SELECTOR
    )
    selector = technique_selector_cls(
        sqlite_client=sqlite_client,
        llm_gateway=llm_gateway,
        prompt_service=prompt_service,
        preprocessor=None,
        embedder=embedding_gateway,
        chroma_client=chroma_client,
        preference_service=preference_service,
    )
    plan_generator_cls = _resolve_dependency("PlanGenerator", _DEFAULT_PLAN_GENERATOR)
    plan_generator = plan_generator_cls(llm_gateway=llm_gateway)
    feedback_manager_cls = _resolve_dependency(
        "FeedbackManager", _DEFAULT_FEEDBACK_MANAGER
    )
    feedback_manager = feedback_manager_cls()
    feedback_repository_cls = _resolve_dependency(
        "FeedbackRepository", _DEFAULT_FEEDBACK_REPOSITORY
    )
    feedback_repository = feedback_repository_cls(sqlite_client=sqlite_client)
    feedback_service_cls = _resolve_dependency(
        "FeedbackService", _DEFAULT_FEEDBACK_SERVICE
    )
    feedback_service = feedback_service_cls(
        feedback_manager=feedback_manager,
        llm_gateway=llm_gateway,
        repository=feedback_repository,
        preference_service=preference_service,
    )
    explanation_service_cls = _resolve_dependency(
        "ExplanationService", _DEFAULT_EXPLANATION_SERVICE
    )
    explanation_service = explanation_service_cls(
        llm_gateway=llm_gateway, prompt_service=prompt_service
    )
    simulation_service_cls = _resolve_dependency(
        "SimulationService", _DEFAULT_SIMULATION_SERVICE
    )
    simulation_service = simulation_service_cls(
        llm_gateway=llm_gateway, prompt_service=prompt_service
    )
    comparison_service_cls = _resolve_dependency(
        "ComparisonService", _DEFAULT_COMPARISON_SERVICE
    )
    comparison_service = comparison_service_cls(
        llm_gateway=llm_gateway, prompt_service=prompt_service
    )

    detect_workflow_cls = _resolve_dependency(
        "DetectTechniqueWorkflow", _DEFAULT_DETECT_WORKFLOW
    )
    plan_workflow_cls = _resolve_dependency(
        "GeneratePlanWorkflow", _DEFAULT_PLAN_WORKFLOW
    )
    feedback_workflow_cls = _resolve_dependency(
        "FeedbackWorkflow", _DEFAULT_FEEDBACK_WORKFLOW
    )
    config_update_workflow_cls = _resolve_dependency(
        "ConfigUpdateWorkflow", _DEFAULT_CONFIG_UPDATE_WORKFLOW
    )
    simulate_workflow_cls = _resolve_dependency(
        "SimulateTechniqueWorkflow", _DEFAULT_SIMULATE_WORKFLOW
    )
    compare_workflow_cls = _resolve_dependency(
        "CompareCandidatesWorkflow", _DEFAULT_COMPARE_WORKFLOW
    )
    orchestrator_cls = _resolve_dependency("Orchestrator", Orchestrator)

    orchestrator = orchestrator_cls(
        workflows={
            "detect_technique": detect_workflow_cls(selector=selector),
            "summarize_result": plan_workflow_cls(plan_generator=plan_generator),
            "feedback_loop": feedback_workflow_cls(feedback_service=feedback_service),
            "config_update": config_update_workflow_cls(),
            "simulate_technique": simulate_workflow_cls(
                simulation_service=simulation_service
            ),
            "compare_candidates": compare_workflow_cls(
                comparison_service=comparison_service
            ),
        }
    )

    state = AppState.load()
    state.llm_gateway = llm_gateway
    state.explanation_service = explanation_service
    state.preference_service = preference_service
    return orchestrator, state


def get_runtime() -> tuple[Orchestrator, AppState]:
    """Return the lazily-initialized orchestrator and CLI state."""

    global _runtime_cache
    if _runtime_cache is None:
        _runtime_cache = initialize_runtime()
    return _runtime_cache


def set_runtime(runtime: tuple[Orchestrator, AppState]) -> None:
    """Replace the cached runtime tuple."""

    global _runtime_cache
    _runtime_cache = runtime


def get_orchestrator() -> Orchestrator:
    """Return the cached orchestrator instance."""

    orchestrator, _ = get_runtime()
    return orchestrator


def get_state() -> AppState:
    """Return the cached application state."""

    _, state = get_runtime()
    return state


def refresh_runtime() -> None:
    """Reinitialize runtime dependencies while preserving session state."""

    _, current_state = get_runtime()
    new_orchestrator, refreshed_state = initialize_runtime()
    refreshed_state.problem_description = current_state.problem_description
    refreshed_state.last_recommendation = current_state.last_recommendation
    refreshed_state.last_explanation = current_state.last_explanation
    refreshed_state.last_simulation = current_state.last_simulation
    refreshed_state.last_comparison = current_state.last_comparison
    refreshed_state.context_history = current_state.context_history
    set_runtime((new_orchestrator, refreshed_state))


def _initialize_chroma_client(
    persist_directory: str, collection_name: str
) -> Optional[Any]:
    """Create a compatibility Chroma client honoring CLI monkeypatches."""

    return create_optional_chroma_client(
        persist_directory,
        collection_name,
        client_cls=_resolve_chroma_client(),
    )


def _resolve_chroma_client() -> Optional[Any]:
    """Return the current ChromaClient implementation (supports monkeypatching)."""

    from sys import modules

    cli_module = modules.get("src.cli")
    if cli_module is not None and hasattr(cli_module, "ChromaClient"):
        return getattr(cli_module, "ChromaClient")
    return _DEFAULT_CHROMA_CLIENT


def _resolve_dependency(name: str, default: Any) -> Any:
    """Return a dependency, preferring overrides on the cli package."""

    from sys import modules

    cli_module = modules.get("src.cli")
    if cli_module is not None and hasattr(cli_module, name):
        return getattr(cli_module, name)
    return default


__all__ = [
    "ChromaClient",
    "ConfigEditor",
    "compose_plan_summary",
    "create_catalog_service",
    "create_initializer",
    "create_search_service",
    "get_orchestrator",
    "get_runtime",
    "get_state",
    "initialize_runtime",
    "refresh_runtime",
    "set_runtime",
    "set_runtime_level",
]
