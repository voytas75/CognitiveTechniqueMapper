"""Command-line interface for the Cognitive Technique Mapper.

Updates:
    v0.1.0 - 2025-11-09 - Added module metadata and Google-style docstrings.
    v0.2.0 - 2025-11-09 - Added settings editor, refresh command, and structured outputs.
    v0.3.0 - 2025-05-09 - Added technique catalog CLI and structured telemetry logging.
    v0.3.1 - 2025-11-09 - Adopted lazy runtime initialization and absolute package imports.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.feedback_manager import FeedbackManager
from src.core.llm_gateway import LLMGateway
from src.core.logging_setup import configure_logging, set_runtime_level
from src.core.orchestrator import Orchestrator
from src.db.feedback_repository import FeedbackRepository
from src.db.preference_repository import PreferenceRepository
from src.db.sqlite_client import SQLiteClient
from src.services.comparison_service import ComparisonService
from src.services.config_editor import ConfigEditor
from src.services.config_service import ConfigService
from src.services.data_initializer import TechniqueDataInitializer
from src.services.embedding_gateway import EmbeddingGateway
from src.services.explanation_service import (
    ExplanationResult,
    ExplanationService,
)
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

try:
    from .db.chroma_client import ChromaClient
except RuntimeError:
    ChromaClient = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = Path(
    os.environ.get("CTM_STATE_PATH", PROJECT_ROOT / "data" / "state.json")
)

console = Console()
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
logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Serializable CLI runtime state."""

    problem_description: Optional[str] = None
    last_recommendation: Optional[dict] = None
    last_explanation: Optional[dict] = None
    last_simulation: Optional[dict] = None
    last_comparison: Optional[dict] = None
    context_history: list[dict] = field(default_factory=list)
    llm_gateway: Optional[LLMGateway] = field(default=None, repr=False, compare=False)
    explanation_service: Optional[ExplanationService] = field(
        default=None, repr=False, compare=False
    )
    preference_service: Optional[PreferenceService] = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> "AppState":
        """Load application state from disk.

        Args:
            path (Path): Path to the state JSON file.

        Returns:
            AppState: Restored state instance.
        """

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}
        return cls(
            problem_description=data.get("problem_description"),
            last_recommendation=data.get("last_recommendation"),
            last_explanation=data.get("last_explanation"),
            last_simulation=data.get("last_simulation"),
            last_comparison=data.get("last_comparison"),
            context_history=data.get("context_history", []),
        )

    def save(self, path: Path = STATE_PATH) -> None:
        """Persist application state to disk.

        Args:
            path (Path): Destination path for the state JSON file.
        """

        payload = {
            "problem_description": self.problem_description,
            "last_recommendation": self.last_recommendation,
            "last_explanation": self.last_explanation,
            "last_simulation": self.last_simulation,
            "last_comparison": self.last_comparison,
            "context_history": self.context_history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_log_override(log_level: Optional[str]) -> None:
    """Override logging level for the current invocation.

    Args:
        log_level (str | None): Logging level name to apply.

    Raises:
        typer.BadParameter: If the provided level is invalid.
    """

    if not log_level:
        return
    try:
        set_runtime_level(log_level)
        logger.info("Log level overridden to %s", log_level.upper())
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


def _compose_plan_summary(recommendation: dict[str, Any]) -> str:
    """Compose a concise summary for the plan generator from a recommendation."""

    technique = recommendation.get("suggested_technique") or ""
    why_it_fits = recommendation.get("why_it_fits") or ""
    steps = recommendation.get("steps") or []
    joined_steps = "; ".join(step for step in steps if step)
    segments = [
        f"Technique: {technique}" if technique else "",
        f"Why it fits: {why_it_fits}" if why_it_fits else "",
        f"Suggested steps: {joined_steps}" if joined_steps else "",
    ]
    return "\n".join(segment for segment in segments if segment)


def initialize_runtime() -> tuple[Orchestrator, AppState]:
    """Initialize core services and hydrated state for CLI usage.

    Returns:
        tuple[Orchestrator, AppState]: Configured orchestrator and application state.
    """

    config_service = ConfigService()
    configure_logging(config_service.logging_config)
    logger.debug("Runtime initialization starting.")
    db_config = config_service.database_config

    sqlite_client = SQLiteClient(db_config.get("sqlite_path", "./data/techniques.db"))
    sqlite_client.initialize_schema()

    chroma_client = None
    if ChromaClient:
        try:
            chroma_client = ChromaClient(
                persist_directory=db_config.get("chromadb_path", "./embeddings"),
                collection_name=db_config.get("chromadb_collection", "techniques"),
            )
        except Exception as exc:  # pragma: no cover - fallback path
            console.print(f"[yellow]ChromaDB disabled: {exc}[/]")
            chroma_client = None

    llm_gateway = LLMGateway(config_service=config_service)
    embedding_gateway = EmbeddingGateway(config_service=config_service)
    initializer = TechniqueDataInitializer(
        sqlite_client=sqlite_client,
        embedder=embedding_gateway,
        chroma_client=chroma_client,
    )
    initializer.initialize()
    logger.debug("Initialization completed (Chroma enabled=%s).", bool(chroma_client))

    prompt_service = PromptService()
    preference_repository = PreferenceRepository(sqlite_client=sqlite_client)
    preference_service = PreferenceService(repository=preference_repository)

    selector = TechniqueSelector(
        sqlite_client=sqlite_client,
        llm_gateway=llm_gateway,
        prompt_service=prompt_service,
        preprocessor=None,
        embedder=embedding_gateway,
        chroma_client=chroma_client,
        preference_service=preference_service,
    )
    plan_generator = PlanGenerator(llm_gateway=llm_gateway)
    feedback_manager = FeedbackManager()
    feedback_repository = FeedbackRepository(sqlite_client=sqlite_client)
    feedback_service = FeedbackService(
        feedback_manager=feedback_manager,
        llm_gateway=llm_gateway,
        repository=feedback_repository,
        preference_service=preference_service,
    )
    explanation_service = ExplanationService(
        llm_gateway=llm_gateway, prompt_service=prompt_service
    )
    simulation_service = SimulationService(
        llm_gateway=llm_gateway, prompt_service=prompt_service
    )
    comparison_service = ComparisonService(
        llm_gateway=llm_gateway, prompt_service=prompt_service
    )

    orchestrator = Orchestrator(
        workflows={
            "detect_technique": DetectTechniqueWorkflow(selector=selector),
            "summarize_result": GeneratePlanWorkflow(plan_generator=plan_generator),
            "feedback_loop": FeedbackWorkflow(feedback_service=feedback_service),
            "config_update": ConfigUpdateWorkflow(),
            "simulate_technique": SimulateTechniqueWorkflow(
                simulation_service=simulation_service
            ),
            "compare_candidates": CompareCandidatesWorkflow(
                comparison_service=comparison_service
            ),
        }
    )

    state = AppState.load()
    state.llm_gateway = llm_gateway
    state.explanation_service = explanation_service
    state.preference_service = preference_service
    return orchestrator, state


_RUNTIME: tuple[Orchestrator, AppState] | None = None


def get_runtime() -> tuple[Orchestrator, AppState]:
    """Return the lazily-initialized orchestrator and CLI state."""

    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = initialize_runtime()
    return _RUNTIME


def set_runtime(runtime: tuple[Orchestrator, AppState]) -> None:
    """Replace the cached runtime tuple."""

    global _RUNTIME
    _RUNTIME = runtime


def get_orchestrator() -> Orchestrator:
    """Return the cached orchestrator instance."""

    orchestrator, _ = get_runtime()
    return orchestrator


def get_state() -> AppState:
    """Return the cached application state."""

    _, state = get_runtime()
    return state


@app.command()
def describe(
    problem: str = typer.Argument(..., help="Describe your problem or challenge."),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation (e.g., DEBUG, INFO).",
    ),
) -> None:
    """Store the user's problem description for subsequent workflows.

    Args:
        problem (str): Problem description supplied by the user.
        log_level (str | None): Logging level override for this invocation.

    Raises:
        typer.BadParameter: If the provided log level is invalid.
    """
    _apply_log_override(log_level)

    state = get_state()
    state.problem_description = problem
    state.context_history.append({"problem_description": problem})
    state.save()
    logger.info("Problem description captured (length=%s)", len(problem))
    console.print(Panel(f"[bold]Problem captured:[/]\n{problem}", title="Describe"))


@app.command()
def analyze(
    show_candidates: bool = typer.Option(
        False,
        "--show-candidates/--hide-candidates",
        help="Display the candidate shortlist with similarity scores.",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    )
) -> None:
    """Trigger the detect_technique workflow.

    Args:
        log_level (str | None): Logging level override for this invocation.

    Raises:
        typer.BadParameter: If no problem description has been captured.
        typer.Exit: When analysis fails due to runtime errors.
    """
    state = get_state()
    if not state.problem_description:
        raise typer.BadParameter("No problem description found. Use `describe` first.")

    _apply_log_override(log_level)

    orchestrator = get_orchestrator()
    context = {"problem_description": state.problem_description}
    try:
        result = orchestrator.execute("detect_technique", context)
    except RuntimeError as exc:
        console.print(f"[red]Analyze failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    recommendation = result.get("recommendation") or {}

    plan_output: dict[str, Any] | None = None
    if recommendation:
        plan_summary = _compose_plan_summary(recommendation)
        if plan_summary:
            try:
                plan_output = orchestrator.execute(
                    "summarize_result", {"technique_summary": plan_summary}
                )
            except (RuntimeError, ValueError) as exc:  # pragma: no cover - LLM issues
                logger.warning("Plan generation failed: %s", exc)

    if plan_output and plan_output.get("plan"):
        result["plan"] = plan_output.get("plan")

    state.last_recommendation = result
    state.context_history.append(result)
    state.save()
    logger.info("Analysis completed.")
    _render_analysis_output(
        recommendation,
        result.get("plan"),
        preference_summary=result.get("preference_summary"),
        matches=result.get("matches") if show_candidates else None,
    )


def _render_analysis_output(
    recommendation: dict[str, Any],
    plan: Any,
    *,
    preference_summary: str | None = None,
    matches: Any = None,
) -> None:
    """Display structured recommendation and optional plan to the console."""

    if not recommendation:
        console.print(Panel("No recommendation returned.", title="Suggested Technique"))
        return

    technique = recommendation.get("suggested_technique") or "No recommendation"
    why_it_fits = recommendation.get("why_it_fits") or "No justification provided."
    steps = recommendation.get("steps") or []

    lines = [
        f"[bold]Suggested:[/]\n{technique}",
        f"[bold]Why it fits:[/]\n{why_it_fits}",
    ]

    if preference_summary:
        lines.append(f"[bold]Preference context:[/]\n{preference_summary}")

    if steps:
        lines.append("[bold]How to apply:[/]")
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")

    if plan:
        lines.append("\n[bold]Implementation plan:[/]")
        lines.append(str(plan))

    console.print(Panel("\n".join(lines), title="Suggested Technique"))

    if matches is not None:
        _render_candidate_matches(matches)


def _render_candidate_matches(matches: Any) -> None:
    if not matches:
        console.print(
            Panel("No candidate techniques were returned.", title="Candidate Matches")
        )
        return

    lines: list[str] = []
    for idx, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        name = metadata.get("name") or match.get("id") or "Unknown"
        score = match.get("score")
        score_display = (
            f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        )
        category = metadata.get("category") or ""
        lines.append(f"{idx}. {name} (score: {score_display})")
        if category:
            lines.append(f"   Category: {category}")
        description = metadata.get("description") or match.get("document") or ""
        if description:
            lines.append(f"   Summary: {description}")
    console.print(Panel("\n".join(lines), title="Candidate Matches"))


def _render_explanation_output(result: ExplanationResult) -> None:
    lines = []
    if result.overview:
        lines.append(f"[bold]Overview:[/]\n{result.overview}")

    if result.key_factors:
        lines.append("[bold]Key factors:[/]")
        for idx, factor in enumerate(result.key_factors, start=1):
            lines.append(f"{idx}. {factor}")

    if result.risks:
        lines.append("\n[bold]Risks & caveats:[/]")
        for idx, risk in enumerate(result.risks, start=1):
            lines.append(f"{idx}. {risk}")

    if result.next_steps:
        lines.append("\n[bold]Suggested next steps:[/]")
        for idx, step in enumerate(result.next_steps, start=1):
            lines.append(f"{idx}. {step}")

    content = "\n".join(lines) if lines else "No explanation details available."
    console.print(Panel(content, title="Explain Logic"))


def _render_simulation_output(simulation: dict[str, Any]) -> None:
    if not simulation:
        console.print(Panel("No simulation details available.", title="Simulation"))
        return

    lines: list[str] = []
    overview = simulation.get("simulation_overview")
    if overview:
        lines.append(f"[bold]Simulation overview:[/]\n{overview}")

    variations = simulation.get("scenario_variations") or []
    if variations:
        lines.append("\n[bold]Scenario variations:[/]")
        for entry in variations:
            name = entry.get("name") or "Scenario"
            outcome = entry.get("outcome") or ""
            guidance = entry.get("guidance") or ""
            lines.append(f"- {name}: {outcome}")
            if guidance:
                lines.append(f"  Guidance: {guidance}")

    cautions = simulation.get("cautions") or []
    if cautions:
        lines.append("\n[bold]Cautions:[/]")
        for idx, caution in enumerate(cautions, start=1):
            lines.append(f"{idx}. {caution}")

    follow_up = simulation.get("recommended_follow_up") or []
    if follow_up:
        lines.append("\n[bold]Recommended follow-up:[/]")
        for idx, action in enumerate(follow_up, start=1):
            lines.append(f"{idx}. {action}")

    console.print(Panel("\n".join(lines), title="Simulation"))


def _render_comparison_output(comparison: dict[str, Any]) -> None:
    if not comparison:
        console.print(Panel("No comparison available.", title="Comparison"))
        return

    lines: list[str] = []
    current = comparison.get("current_recommendation") or "Unknown"
    lines.append(f"[bold]Current recommendation:[/]\n{current}")

    alternative = comparison.get("best_alternative")
    if alternative:
        lines.append(f"\n[bold]Top alternative:[/]\n{alternative}")

    points = comparison.get("comparison_points") or []
    if points:
        lines.append("\n[bold]Comparison points:[/]")
        for point in points:
            technique = point.get("technique") or "Candidate"
            strengths = point.get("strengths") or ""
            risks = point.get("risks") or ""
            best_for = point.get("best_for") or ""
            lines.append(f"- {technique}")
            if strengths:
                lines.append(f"  Strengths: {strengths}")
            if risks:
                lines.append(f"  Risks: {risks}")
            if best_for:
                lines.append(f"  Best for: {best_for}")

    guidance = comparison.get("decision_guidance") or []
    if guidance:
        lines.append("\n[bold]Decision guidance:[/]")
        for idx, tip in enumerate(guidance, start=1):
            lines.append(f"{idx}. {tip}")

    confidence = comparison.get("confidence_notes")
    if confidence:
        lines.append(f"\n[bold]Confidence notes:[/]\n{confidence}")

    console.print(Panel("\n".join(lines), title="Comparison"))


def _active_preference_summary() -> Optional[str]:
    state = get_state()
    if not state.preference_service:
        return None
    summary = state.preference_service.preference_summary()
    return summary or None


def _infer_category_from_matches(matches: Any, technique: Optional[str]) -> Optional[str]:
    if not technique or not isinstance(matches, list):
        return None
    target = technique.lower()
    for match in matches:
        if not isinstance(match, dict):
            continue
        metadata = match.get("metadata") or {}
        name = metadata.get("name") or match.get("id")
        if isinstance(name, str) and name.lower() == target:
            category = metadata.get("category") or match.get("category")
            if isinstance(category, str) and category.strip():
                return category
    return None


@app.command()
def explain(
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    )
) -> None:
    """Explain the logic behind the last recommendation via the explain_logic workflow.

    Args:
        log_level (str | None): Logging level override for this invocation.

    Raises:
        typer.BadParameter: If no recommendation is available or the gateway is missing.
        typer.Exit: When the explain workflow fails.
    """
    state = get_state()
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")

    _apply_log_override(log_level)

    if not state.explanation_service:
        raise typer.BadParameter("Explanation service not initialized.")

    try:
        explanation = state.explanation_service.explain(
            state.last_recommendation or {},
            problem_description=state.problem_description,
        )
    except RuntimeError as exc:
        console.print(f"[red]Explain failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    logger.info("Explain workflow executed.")
    state.last_explanation = explanation.as_dict()
    state.context_history.append({"explanation": state.last_explanation})
    state.save()
    _render_explanation_output(explanation)


@app.command()
def simulate(
    scenario: Optional[str] = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Optional scenario focus or constraint to explore during simulation.",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Simulate applying the recommended technique across scenario variations."""

    state = get_state()
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")

    recommendation = state.last_recommendation.get("recommendation") or {}
    if not recommendation:
        raise typer.BadParameter("Current recommendation payload is empty.")

    _apply_log_override(log_level)
    preference_summary = _active_preference_summary()
    context = {
        "recommendation": recommendation,
        "problem_description": state.problem_description,
        "scenario": scenario or state.problem_description,
        "preference_summary": preference_summary,
    }
    orchestrator = get_orchestrator()
    try:
        result = orchestrator.execute("simulate_technique", context)
    except RuntimeError as exc:
        console.print(f"[red]Simulation failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    simulation = result.get("simulation") or {}
    state.last_simulation = simulation
    state.context_history.append({"simulation": simulation})
    state.save()
    logger.info("Simulation workflow executed.")
    _render_simulation_output(simulation)


@app.command()
def compare(
    focus: Optional[str] = typer.Option(
        None,
        "--focus",
        "-f",
        help="Optional technique name to prioritise in the comparison.",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Maximum number of candidates to include from the shortlist (0 = all).",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Compare shortlisted techniques and highlight trade-offs."""

    state = get_state()
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")

    recommendation = state.last_recommendation.get("recommendation") or {}
    matches = state.last_recommendation.get("matches") or []
    if not recommendation or not matches:
        raise typer.BadParameter(
            "Candidate shortlist unavailable. Re-run `analyze` to regenerate matches."
        )

    _apply_log_override(log_level)
    shortlist = matches if limit <= 0 else matches[:limit]
    preference_summary = _active_preference_summary()
    context = {
        "recommendation": recommendation,
        "matches": shortlist,
        "focus": focus,
        "preference_summary": preference_summary,
    }
    orchestrator = get_orchestrator()
    try:
        result = orchestrator.execute("compare_candidates", context)
    except RuntimeError as exc:
        console.print(f"[red]Comparison failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    comparison = result.get("comparison") or {}
    state.last_comparison = comparison
    state.context_history.append({"comparison": comparison})
    state.save()
    logger.info("Comparison workflow executed.")
    _render_comparison_output(comparison)


@settings_app.callback(invoke_without_command=True)
def settings_callback(ctx: typer.Context) -> None:
    """Default behaviour for the settings command when no subcommand is passed."""

    if ctx.invoked_subcommand is None:
        _show_settings()


@settings_app.command("show")
def settings_show() -> None:
    """Display the current configuration payload."""

    _show_settings()


@settings_app.command("update-workflow")
def settings_update_workflow(
    workflow: str | None = typer.Argument(None, help="Workflow name to modify."),
    model: str | None = typer.Option(None, help="Updated model identifier."),
    temperature: float | None = typer.Option(
        None, help="Temperature parameter between 0.0 and 2.0."
    ),
    provider: str | None = typer.Option(
        None, help="Provider key to associate with the workflow."
    ),
    max_tokens: int | None = typer.Option(
        None, help="Maximum completion tokens for the workflow."
    ),
    clear_max_tokens: bool = typer.Option(
        False, help="Remove the max_tokens entry for the workflow."
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt for values that were not supplied via options.",
    ),
) -> None:
    """Update workflow configuration in models.yaml and refresh runtime services."""

    config_service = ConfigService()
    workflow_configs = config_service.iter_workflow_configs()

    if workflow is None:
        if not interactive:
            available = ", ".join(sorted(workflow_configs)) or "(none configured)"
            raise typer.BadParameter(
                f"Workflow argument required. Available workflows: {available}."
            )
        default_workflow = next(iter(workflow_configs.keys()), "")
        workflow = _prompt_value("Workflow", default_workflow)
        if not workflow:
            raise typer.BadParameter("Workflow selection is required.")

    try:
        current = workflow_configs[workflow]
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if interactive:
        model = model or _prompt_value("Model", current.model)
        temperature = _prompt_float("Temperature", current.temperature)
        provider = provider or _prompt_value("Provider", current.provider)
        if not clear_max_tokens:
            max_tokens = _prompt_int("Max tokens", current.max_tokens)
            if max_tokens is None and current.max_tokens is not None:
                clear_max_tokens = True

    editor = ConfigEditor()
    try:
        editor.update_workflow_model(
            workflow,
            model=model,
            temperature=temperature,
            provider=provider,
            max_tokens=max_tokens,
            clear_max_tokens=clear_max_tokens,
        )
    except (KeyError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    ConfigService.clear_cache()
    _refresh_runtime()
    console.print(f"[green]Updated workflow '{workflow}'.[/]")
    _show_settings()


@settings_app.command("update-provider")
def settings_update_provider(
    provider: str | None = typer.Argument(None, help="Provider name to update."),
    api_base: str | None = typer.Option(None, help="HTTP base URL for the provider."),
    api_version: str | None = typer.Option(None, help="Optional API version."),
    api_key_env: str | None = typer.Option(
        None, help="Environment variable holding the API key."
    ),
    clear_api_version: bool = typer.Option(
        False, help="Remove the api_version entry for the provider."
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt for provider fields that were not supplied.",
    ),
) -> None:
    """Update provider metadata in providers.yaml and refresh runtime services."""

    config_service = ConfigService()
    providers = config_service.providers

    if provider is None:
        if not interactive:
            available = ", ".join(sorted(providers)) or "(none configured)"
            raise typer.BadParameter(
                f"Provider argument required. Available providers: {available}."
            )
        default_provider = next(iter(providers.keys()), "")
        provider = _prompt_value("Provider", default_provider)
        if not provider:
            raise typer.BadParameter("Provider selection is required.")

    current = providers.get(provider, {})

    if interactive:
        api_base = api_base or _prompt_value("API base", current.get("api_base"))
        api_version = _prompt_value("API version", current.get("api_version"))
        api_key_env = api_key_env or _prompt_value(
            "API key env", current.get("api_key_env")
        )
        if api_version is None and current.get("api_version") and not clear_api_version:
            clear_api_version = True

    editor = ConfigEditor()
    try:
        editor.update_provider(
            provider,
            api_base=api_base,
            api_version=api_version,
            api_key_env=api_key_env,
            clear_api_version=clear_api_version,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    ConfigService.clear_cache()
    _refresh_runtime()
    console.print(f"[green]Updated provider '{provider}'.[/]")
    refreshed = ConfigService().providers.get(provider, {})
    console.print_json(data={"provider": provider, "config": refreshed})


app.add_typer(history_app, name="history")
app.add_typer(preferences_app, name="preferences")
app.add_typer(techniques_app, name="techniques")
app.add_typer(settings_app, name="settings")


@app.command()
def refresh(
    rebuild_embeddings: bool = typer.Option(
        True,
        "--rebuild-embeddings/--skip-embeddings",
        help="Recompute and sync embeddings with the vector store.",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Reload the techniques dataset and optionally rebuild embeddings."""

    _apply_log_override(log_level)

    initializer, sqlite_client = _create_initializer()
    try:
        initializer.refresh(rebuild_embeddings=rebuild_embeddings)
    except Exception as exc:  # pragma: no cover - dependent on external services
        console.print(f"[red]Refresh failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        sqlite_client.close()

    _refresh_runtime()
    console.print(
        Panel("Dataset refreshed with latest configuration.", title="Refresh")
    )


@app.command()
def feedback(
    message: str = typer.Argument(..., help="Feedback message."),
    rating: Optional[int] = typer.Option(None, help="Optional rating 1-5."),
    technique: Optional[str] = typer.Option(
        None,
        "--technique",
        "-t",
        help="Technique the feedback refers to (defaults to last recommendation).",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Technique category the feedback refers to.",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Record user feedback and display the summary of recent entries.

    Args:
        message (str): Feedback body supplied by the user.
        rating (int | None): Optional rating between 1 and 5.
        log_level (str | None): Logging level override for this invocation.

    Raises:
        typer.Exit: When feedback operations fail.
    """
    state = get_state()
    orchestrator = get_orchestrator()

    _apply_log_override(log_level)
    if rating is not None and (rating < 1 or rating > 5):
        raise typer.BadParameter("Rating must be between 1 and 5.")

    if technique is None and state.last_recommendation:
        technique = (
            (state.last_recommendation.get("recommendation") or {}).get(
                "suggested_technique"
            )
        )
    if category is None and technique:
        category = _infer_category_from_matches(
            state.last_recommendation.get("matches")
            if state.last_recommendation
            else [],
            technique,
        )

    context = {
        "action": "record",
        "message": message,
        "rating": rating,
        "workflow": "detect_technique",
        "technique": technique,
        "category": category,
    }
    try:
        orchestrator.execute("feedback_loop", context)
        summary = orchestrator.execute("feedback_loop", {})
    except RuntimeError as exc:
        console.print(f"[red]Feedback failed: {exc}[/]")
        raise typer.Exit(code=1) from exc

    if state.preference_service:
        state.preference_service.record_preference(
            technique=technique,
            category=category,
            rating=rating,
            notes=message,
        )

    logger.info("Feedback recorded (rating=%s)", rating)
    console.print(
        Panel(summary.get("summary", "No summary available."), title="Feedback Summary")
    )


@history_app.command("show")
def history_show(
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        min=0,
        help="Number of most recent history entries to display (0 = all).",
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Emit raw JSON instead of rendered panels.",
    ),
) -> None:
    """Display recent session history captured by the CLI."""

    state = get_state()
    entries = state.context_history
    if not entries:
        console.print(Panel("History is empty.", title="History"))
        return

    subset = entries if limit == 0 else entries[-limit:]
    start_index = len(entries) - len(subset)

    if raw:
        console.print_json(data=subset)
        return

    for offset, entry in enumerate(subset, start=1):
        event_number = start_index + offset
        console.print(
            Panel(
                json.dumps(entry, ensure_ascii=False, indent=2),
                title=f"Event {event_number}",
            )
        )


@history_app.command("clear")
def history_clear(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Clear without confirmation prompt.",
    )
) -> None:
    """Erase the stored session history."""

    state = get_state()
    if not state.context_history:
        console.print("[yellow]History is already empty.[/]")
        return

    if not force and not typer.confirm("Clear all history entries?"):
        console.print("[yellow]History unchanged.[/]")
        return

    state.context_history.clear()
    state.save()
    console.print("[green]History cleared.[/]")


@preferences_app.command("summary")
def preferences_summary() -> None:
    """Show a human-readable summary of recorded preferences."""

    state = get_state()
    if not state.preference_service:
        console.print("[yellow]Preference service unavailable.[/]")
        return

    summary = state.preference_service.preference_summary()
    if not summary:
        console.print(Panel("No preference signals recorded yet.", title="Preferences"))
        return

    console.print(Panel(summary, title="Preference Summary"))


@preferences_app.command("export")
def preferences_export() -> None:
    """Export the full preference profile as JSON."""

    state = get_state()
    if not state.preference_service:
        console.print("[yellow]Preference service unavailable.[/]")
        return

    profile = state.preference_service.export_profile()
    console.print_json(data=asdict(profile))


@preferences_app.command("reset")
def preferences_reset(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Reset preferences without a confirmation prompt.",
    )
) -> None:
    """Remove all stored feedback-based preferences."""

    state = get_state()
    if not state.preference_service:
        console.print("[yellow]Preference service unavailable.[/]")
        return

    if not force and not typer.confirm("Clear all stored preferences?"):
        console.print("[yellow]Preferences unchanged.[/]")
        return

    state.preference_service.clear()
    console.print("[green]Preferences cleared.[/]")


@techniques_app.command("list")
def techniques_list() -> None:
    """Display techniques stored in the knowledge base."""

    catalog, sqlite_client = _create_catalog_service()
    try:
        entries = catalog.list()
    finally:
        sqlite_client.close()

    if not entries:
        console.print(Panel("No techniques available.", title="Techniques"))
        return

    table = Table(title="Techniques", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Category")
    table.add_column("Origin Year")
    table.add_column("Creator")
    table.add_column("Description", overflow="fold")

    for entry in entries:
        description = entry.get("description") or ""
        truncated = description if len(description) <= 120 else description[:117] + "..."
        table.add_row(
            entry.get("name") or "",
            entry.get("category") or "-",
            str(entry.get("origin_year") or "-"),
            entry.get("creator") or "-",
            truncated,
        )

    console.print(table)


@techniques_app.command("add")
def techniques_add(
    name: str = typer.Option(..., "--name", "-n", help="Technique name."),
    description: str = typer.Option(
        ..., "--description", "-d", help="Technique description."
    ),
    origin_year: Optional[int] = typer.Option(
        None, "--origin-year", help="Origin year of the technique."
    ),
    creator: Optional[str] = typer.Option(
        None, "--creator", help="Creator attribution for the technique."
    ),
    category: Optional[str] = typer.Option(
        None, "--category", help="Technique category or theme."
    ),
    core_principles: Optional[str] = typer.Option(
        None, "--core-principles", help="Key principles for the technique."
    ),
) -> None:
    """Add a new technique and refresh runtime services."""

    catalog, sqlite_client = _create_catalog_service()
    success = False
    try:
        entry = catalog.add(
            {
                "name": name,
                "description": description,
                "origin_year": origin_year,
                "creator": creator,
                "category": category,
                "core_principles": core_principles,
            }
        )
        success = True
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - defensive path
        console.print(f"[red]Failed to add technique: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        sqlite_client.close()

    if success:
        _refresh_runtime()
        console.print_json(data={"technique": entry})


@techniques_app.command("update")
def techniques_update(
    name: str = typer.Argument(..., help="Existing technique name."),
    new_name: Optional[str] = typer.Option(
        None, "--new-name", help="Rename the technique to this value."
    ),
    description: Optional[str] = typer.Option(
        None, "--description", help="Updated description."
    ),
    origin_year: Optional[int] = typer.Option(
        None, "--origin-year", help="Updated origin year."
    ),
    creator: Optional[str] = typer.Option(
        None, "--creator", help="Updated creator attribution."
    ),
    category: Optional[str] = typer.Option(
        None, "--category", help="Updated category."
    ),
    core_principles: Optional[str] = typer.Option(
        None, "--core-principles", help="Updated core principles."
    ),
) -> None:
    """Update an existing technique and refresh runtime services."""

    updates: dict[str, Any] = {}
    if new_name:
        updates["name"] = new_name
    if description is not None:
        updates["description"] = description
    if origin_year is not None:
        updates["origin_year"] = origin_year
    if creator is not None:
        updates["creator"] = creator
    if category is not None:
        updates["category"] = category
    if core_principles is not None:
        updates["core_principles"] = core_principles

    if not updates:
        raise typer.BadParameter("Provide at least one field to update.")

    catalog, sqlite_client = _create_catalog_service()
    success = False
    try:
        entry = catalog.update(name, updates)
        success = True
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - defensive path
        console.print(f"[red]Failed to update technique: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        sqlite_client.close()

    if success:
        _refresh_runtime()
        console.print_json(data={"technique": entry})


@techniques_app.command("remove")
def techniques_remove(
    name: str = typer.Argument(..., help="Technique name to remove."),
) -> None:
    """Remove a technique from the catalog and refresh runtime services."""

    catalog, sqlite_client = _create_catalog_service()
    success = False
    try:
        catalog.remove(name)
        success = True
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - defensive path
        console.print(f"[red]Failed to remove technique: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        sqlite_client.close()

    if success:
        _refresh_runtime()
        console.print(f"[green]Removed technique '{name}'.[/]")


@techniques_app.command("export")
def techniques_export(
    file: Path = typer.Option(
        ...,
        "--file",
        "-f",
        help="Destination path for exported techniques (JSON).",
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
    )
) -> None:
    """Export the current technique catalog to a JSON file."""

    catalog, sqlite_client = _create_catalog_service()
    try:
        path, count = catalog.export_to_file(file)
    except Exception as exc:
        console.print(f"[red]Export failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        sqlite_client.close()

    console.print_json(data={"file": str(path), "count": count})


@techniques_app.command("import")
def techniques_import(
    file: Path = typer.Option(
        ...,
        "--file",
        "-f",
        help="JSON file containing techniques to import.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    mode: str = typer.Option(
        "replace",
        "--mode",
        "-m",
        help="Import mode: replace existing data or append to it.",
    ),
    rebuild_embeddings: bool = typer.Option(
        True,
        "--rebuild-embeddings/--skip-embeddings",
        help="Recompute embeddings after import when a vector store is configured.",
    ),
) -> None:
    """Import techniques from a JSON file using append or replace semantics."""

    catalog, sqlite_client = _create_catalog_service()
    try:
        summary = catalog.import_from_file(
            file,
            mode=mode.lower(),
            rebuild_embeddings=rebuild_embeddings,
        )
    except Exception as exc:
        console.print(f"[red]Import failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        sqlite_client.close()

    _refresh_runtime()
    console.print_json(data={"mode": mode.lower(), **summary})


def _show_settings() -> None:
    orchestrator = get_orchestrator()
    config_summary = orchestrator.execute("config_update", {})
    console.print_json(data=config_summary)


def _refresh_runtime() -> None:
    current_orchestrator, current_state = get_runtime()
    new_orchestrator, refreshed_state = initialize_runtime()
    refreshed_state.problem_description = current_state.problem_description
    refreshed_state.last_recommendation = current_state.last_recommendation
    refreshed_state.last_explanation = current_state.last_explanation
    refreshed_state.last_simulation = current_state.last_simulation
    refreshed_state.last_comparison = current_state.last_comparison
    refreshed_state.context_history = current_state.context_history
    set_runtime((new_orchestrator, refreshed_state))


def _prompt_value(label: str, current: str | None) -> str | None:
    default_display = current if current is not None else ""
    response = typer.prompt(label, default=default_display)
    return response.strip() or current


def _prompt_float(label: str, current: float | None) -> float | None:
    default_display = "" if current is None else str(current)
    response = typer.prompt(label, default=default_display).strip()
    if not response:
        return current
    try:
        return float(response)
    except ValueError as exc:  # pragma: no cover - input validation
        raise typer.BadParameter(f"Invalid float for {label}: {response}") from exc


def _prompt_int(label: str, current: int | None) -> int | None:
    default_display = "" if current is None else str(current)
    response = typer.prompt(label, default=default_display).strip()
    if not response:
        return current
    try:
        return int(response)
    except ValueError as exc:  # pragma: no cover - input validation
        raise typer.BadParameter(f"Invalid integer for {label}: {response}") from exc


def _create_catalog_service() -> tuple[TechniqueCatalogService, SQLiteClient]:
    config_service = ConfigService()
    db_config = config_service.database_config

    sqlite_client = SQLiteClient(db_config.get("sqlite_path", "./data/techniques.db"))
    sqlite_client.initialize_schema()

    chroma_client = None
    if ChromaClient:
        try:
            chroma_client = ChromaClient(
                persist_directory=db_config.get("chromadb_path", "./embeddings"),
                collection_name=db_config.get("chromadb_collection", "techniques"),
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            console.print(f"[yellow]ChromaDB disabled: {exc}[/]")
            chroma_client = None

    embedder = EmbeddingGateway(config_service=config_service)
    dataset_path = PROJECT_ROOT / "data" / "techniques.json"
    catalog = TechniqueCatalogService(
        sqlite_client=sqlite_client,
        embedder=embedder,
        dataset_path=dataset_path,
        chroma_client=chroma_client,
    )
    return catalog, sqlite_client


def _create_initializer() -> tuple[TechniqueDataInitializer, SQLiteClient]:
    config_service = ConfigService()
    db_config = config_service.database_config

    sqlite_client = SQLiteClient(db_config.get("sqlite_path", "./data/techniques.db"))
    sqlite_client.initialize_schema()

    chroma_client = None
    if ChromaClient:
        try:
            chroma_client = ChromaClient(
                persist_directory=db_config.get("chromadb_path", "./embeddings"),
                collection_name=db_config.get("chromadb_collection", "techniques"),
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            console.print(f"[yellow]ChromaDB disabled: {exc}[/]")
            chroma_client = None

    embedder = EmbeddingGateway(config_service=config_service)
    initializer = TechniqueDataInitializer(
        sqlite_client=sqlite_client,
        embedder=embedder,
        chroma_client=chroma_client,
    )
    return initializer, sqlite_client


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
