from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from src.core.feedback_manager import FeedbackManager
from src.core.llm_gateway import LLMGateway
from src.core.orchestrator import Orchestrator
from src.db.sqlite_client import SQLiteClient
from src.services.config_service import ConfigService
from src.services.feedback_service import FeedbackService
from src.services.plan_generator import PlanGenerator
from src.services.technique_selector import TechniqueSelector
from src.workflows.config_update import ConfigUpdateWorkflow
from src.workflows.detect_technique import DetectTechniqueWorkflow
from src.workflows.feedback_loop import FeedbackWorkflow
from src.workflows.generate_plan import GeneratePlanWorkflow

try:
    from src.db.chroma_client import ChromaClient
except RuntimeError:
    ChromaClient = None  # type: ignore

console = Console()
app = typer.Typer(add_completion=False, help="Cognitive Technique Mapper CLI")


@dataclass
class AppState:
    problem_description: Optional[str] = None
    last_recommendation: Optional[dict] = None
    context_history: list[dict] = field(default_factory=list)
    llm_gateway: Optional[LLMGateway] = None


def initialize_runtime() -> tuple[Orchestrator, AppState]:
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
        except Exception as exc:  # pragma: no cover - fallback path
            console.print(f"[yellow]ChromaDB disabled: {exc}[/]")
            chroma_client = None

    llm_gateway = LLMGateway(config_service=config_service)
    selector = TechniqueSelector(sqlite_client=sqlite_client, llm_gateway=llm_gateway, chroma_client=chroma_client)
    plan_generator = PlanGenerator(llm_gateway=llm_gateway)
    feedback_manager = FeedbackManager()
    feedback_service = FeedbackService(feedback_manager=feedback_manager, llm_gateway=llm_gateway)

    orchestrator = Orchestrator(
        workflows={
            "detect_technique": DetectTechniqueWorkflow(selector=selector),
            "summarize_result": GeneratePlanWorkflow(plan_generator=plan_generator),
            "feedback_loop": FeedbackWorkflow(feedback_service=feedback_service),
            "config_update": ConfigUpdateWorkflow(),
        }
    )

    state = AppState(llm_gateway=llm_gateway)
    return orchestrator, state


orchestrator, state = initialize_runtime()


@app.command()
def describe(problem: str = typer.Argument(..., help="Describe your problem or challenge.")) -> None:
    """Store the user's problem description for subsequent workflows."""
    state.problem_description = problem
    state.context_history.append({"problem_description": problem})
    console.print(Panel(f"[bold]Problem captured:[/]\n{problem}", title="Describe"))


@app.command()
def analyze() -> None:
    """Trigger the detect_technique workflow."""
    if not state.problem_description:
        raise typer.BadParameter("No problem description found. Use `describe` first.")

    context = {"problem_description": state.problem_description}
    result = orchestrator.execute("detect_technique", context)
    state.last_recommendation = result
    state.context_history.append(result)
    console.print(Panel(result.get("suggested_technique", "No recommendation"), title="Suggested Technique"))


@app.command()
def explain() -> None:
    """Explain the logic behind the last recommendation via the explain_logic workflow."""
    if not state.last_recommendation:
        raise typer.BadParameter("No recommendation available. Run `analyze` first.")

    prompt = (
        "Explain the reasoning that led to the following recommendation.\n"
        f"Recommendation payload:\n{json.dumps(state.last_recommendation, indent=2)}"
    )
    if not state.llm_gateway:
        raise typer.BadParameter("LLM gateway not initialized.")

    llm_output = state.llm_gateway.invoke("explain_logic", prompt)
    console.print(Panel(llm_output, title="Explain Logic"))


@app.command()
def settings() -> None:
    """Display current application configuration values."""
    config_summary = orchestrator.execute("config_update", {})
    console.print_json(data=config_summary)


@app.command()
def feedback(
    message: str = typer.Argument(..., help="Feedback message."),
    rating: Optional[int] = typer.Option(None, help="Optional rating 1-5."),
) -> None:
    """Record user feedback and display the summary of recent entries."""
    context = {"action": "record", "message": message, "rating": rating, "workflow": "detect_technique"}
    orchestrator.execute("feedback_loop", context)
    summary = orchestrator.execute("feedback_loop", {})
    console.print(Panel(summary.get("summary", "No summary available."), title="Feedback Summary"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
