"""Command-line interface for the Cognitive Technique Mapper.

Updates:
    v0.1.0 - 2025-11-09 - Added module metadata and Google-style docstrings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
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
from .services.config_service import ConfigService
from .services.data_initializer import TechniqueDataInitializer
from .services.embedding_gateway import EmbeddingGateway
from .services.feedback_service import FeedbackService
from .services.plan_generator import PlanGenerator
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

    selector = TechniqueSelector(
        sqlite_client=sqlite_client,
        llm_gateway=llm_gateway,
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
    state.last_recommendation = result
    state.context_history.append(result)
    state.save()
    logger.info("Analysis completed.")
    console.print(
        Panel(
            result.get("suggested_technique", "No recommendation"),
            title="Suggested Technique",
        )
    )


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


@app.command()
def settings() -> None:
    """Display current application configuration values.

    Returns:
        None: This command prints configuration details to the console.
    """
    config_summary = orchestrator.execute("config_update", {})
    console.print_json(data=config_summary)


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


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
