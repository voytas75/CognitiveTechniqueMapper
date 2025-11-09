"""Command-line interface for the Cognitive Technique Mapper.

Updates:
    v0.1.0 - 2025-11-09 - Added module metadata and Google-style docstrings.
    v0.2.0 - 2025-11-09 - Added settings editor, refresh command, and structured outputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json as _json

import os
import typer
from rich.console import Console
from rich.panel import Panel

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATE_PATH = Path(
    os.environ.get("CTM_STATE_PATH", PROJECT_ROOT / "data" / "state.json")
)

from .core.feedback_manager import FeedbackManager
from .core.llm_gateway import LLMGateway
from .core.orchestrator import Orchestrator
from .core.logging_setup import configure_logging, set_runtime_level
from .db.sqlite_client import SQLiteClient
from .services.config_editor import ConfigEditor
from .services.config_service import ConfigService
from .services.data_initializer import TechniqueDataInitializer
from .services.embedding_gateway import EmbeddingGateway
from .services.feedback_service import FeedbackService
from .services.plan_generator import PlanGenerator
from .services.prompt_service import PromptService
from .services.technique_selector import TechniqueSelector
from .workflows.config_update import ConfigUpdateWorkflow
from .workflows.detect_technique import DetectTechniqueWorkflow
from .workflows.feedback_loop import FeedbackWorkflow
from .workflows.generate_plan import GeneratePlanWorkflow

try:
    from .db.chroma_client import ChromaClient
except RuntimeError:
    ChromaClient = None  # type: ignore

console = Console()
app = typer.Typer(add_completion=False, help="Cognitive Technique Mapper CLI")
settings_app = typer.Typer(
    add_completion=False, help="Inspect and edit application configuration."
)
logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Serializable CLI runtime state."""

    problem_description: Optional[str] = None
    last_recommendation: Optional[dict] = None
    context_history: list[dict] = field(default_factory=list)
    llm_gateway: Optional[LLMGateway] = field(default=None, repr=False, compare=False)

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
                data = _json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}
        return cls(
            problem_description=data.get("problem_description"),
            last_recommendation=data.get("last_recommendation"),
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
            "context_history": self.context_history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")


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

    selector = TechniqueSelector(
        sqlite_client=sqlite_client,
        llm_gateway=llm_gateway,
        prompt_service=prompt_service,
        preprocessor=None,
        embedder=embedding_gateway,
        chroma_client=chroma_client,
    )
    plan_generator = PlanGenerator(llm_gateway=llm_gateway)
    feedback_manager = FeedbackManager()
    feedback_service = FeedbackService(
        feedback_manager=feedback_manager, llm_gateway=llm_gateway
    )

    orchestrator = Orchestrator(
        workflows={
            "detect_technique": DetectTechniqueWorkflow(selector=selector),
            "summarize_result": GeneratePlanWorkflow(plan_generator=plan_generator),
            "feedback_loop": FeedbackWorkflow(feedback_service=feedback_service),
            "config_update": ConfigUpdateWorkflow(),
        }
    )

    state = AppState.load()
    state.llm_gateway = llm_gateway
    return orchestrator, state


orchestrator, state = initialize_runtime()


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

    state.problem_description = problem
    state.context_history.append({"problem_description": problem})
    state.save()
    logger.info("Problem description captured (length=%s)", len(problem))
    console.print(Panel(f"[bold]Problem captured:[/]\n{problem}", title="Describe"))


@app.command()
def analyze(
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
    if not state.problem_description:
        raise typer.BadParameter("No problem description found. Use `describe` first.")

    _apply_log_override(log_level)

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
    _render_analysis_output(recommendation, result.get("plan"))


def _render_analysis_output(recommendation: dict[str, Any], plan: Any) -> None:
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

    if steps:
        lines.append("[bold]How to apply:[/]")
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")

    if plan:
        lines.append("\n[bold]Implementation plan:[/]")
        lines.append(str(plan))

    console.print(Panel("\n".join(lines), title="Suggested Technique"))


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
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")

    _apply_log_override(log_level)

    prompt = (
        "Explain the reasoning that led to the following recommendation.\n"
        f"Recommendation payload:\n{json.dumps(state.last_recommendation, indent=2)}"
    )
    if not state.llm_gateway:
        raise typer.BadParameter("LLM gateway not initialized.")

    try:
        llm_output = state.llm_gateway.invoke("explain_logic", prompt)
    except RuntimeError as exc:
        console.print(f"[red]Explain failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    logger.info("Explain workflow executed.")
    console.print(Panel(llm_output, title="Explain Logic"))


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
    _apply_log_override(log_level)
    context = {
        "action": "record",
        "message": message,
        "rating": rating,
        "workflow": "detect_technique",
    }
    try:
        orchestrator.execute("feedback_loop", context)
        summary = orchestrator.execute("feedback_loop", {})
    except RuntimeError as exc:
        console.print(f"[red]Feedback failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    logger.info("Feedback recorded (rating=%s)", rating)
    console.print(
        Panel(summary.get("summary", "No summary available."), title="Feedback Summary")
    )


def _show_settings() -> None:
    config_summary = orchestrator.execute("config_update", {})
    console.print_json(data=config_summary)


def _refresh_runtime() -> None:
    global orchestrator, state

    orchestrator, refreshed_state = initialize_runtime()
    refreshed_state.problem_description = state.problem_description
    refreshed_state.last_recommendation = state.last_recommendation
    refreshed_state.context_history = state.context_history
    state = refreshed_state


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
