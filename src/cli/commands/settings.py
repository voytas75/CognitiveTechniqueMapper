"""Settings command group for the Cognitive Technique Mapper CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from src.cli.io import console
from src.cli.renderers import render_prompt_sample

from src.cli.utils import (
    prompt_float,
    prompt_int,
    prompt_value,
    refresh_runtime_and_preserve_state,
)


def _cli():
    return sys.modules["src.cli"]


def settings_show() -> None:
    """Display the current configuration payload."""

    orchestrator = _cli().get_orchestrator()
    config_summary = orchestrator.execute("config_update", {})
    console.print_json(data=config_summary)


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

    cli_module = _cli()
    config_service = cli_module.ConfigService()
    workflow_configs = config_service.iter_workflow_configs()

    if workflow is None:
        if not interactive:
            available = ", ".join(sorted(workflow_configs)) or "(none configured)"
            raise typer.BadParameter(
                f"Workflow argument required. Available workflows: {available}."
            )
        default_workflow = next(iter(workflow_configs.keys()), "")
        workflow = cli_module._prompt_value("Workflow", default_workflow)
        if not workflow:
            raise typer.BadParameter("Workflow selection is required.")

    try:
        current = workflow_configs[workflow]
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if interactive:
        model = model or cli_module._prompt_value("Model", current.model)
        temperature = cli_module._prompt_float("Temperature", current.temperature)
        provider = provider or cli_module._prompt_value("Provider", current.provider)
        if not clear_max_tokens:
            max_tokens = cli_module._prompt_int("Max tokens", current.max_tokens)
            if max_tokens is None and current.max_tokens is not None:
                clear_max_tokens = True

    editor = cli_module.ConfigEditor()
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

    cli_module.ConfigService.clear_cache()
    cli_module._refresh_runtime()
    console.print(f"[green]Updated workflow '{workflow}'.[/]")
    settings_show()


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

    cli_module = _cli()
    config_service = cli_module.ConfigService()
    providers = config_service.providers

    if provider is None:
        if not interactive:
            available = ", ".join(sorted(providers)) or "(none configured)"
            raise typer.BadParameter(
                f"Provider argument required. Available providers: {available}."
            )
        default_provider = next(iter(providers.keys()), "")
        provider = cli_module._prompt_value("Provider", default_provider)
        if not provider:
            raise typer.BadParameter("Provider selection is required.")

    current = providers.get(provider, {})

    if interactive:
        api_base = api_base or cli_module._prompt_value("API base", current.get("api_base"))
        api_version = cli_module._prompt_value("API version", current.get("api_version"))
        api_key_env = api_key_env or cli_module._prompt_value(
            "API key env", current.get("api_key_env")
        )
        if api_version is None and current.get("api_version") and not clear_api_version:
            clear_api_version = True

    editor = cli_module.ConfigEditor()
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

    cli_module.ConfigService.clear_cache()
    cli_module._refresh_runtime()
    console.print(f"[green]Updated provider '{provider}'.[/]")
    refreshed = cli_module.ConfigService().providers.get(provider, {})
    console.print_json(data={"provider": provider, "config": refreshed})


def settings_show_prompt(
    prompt: str = typer.Argument(..., help="Prompt name registered in prompts/registry.yaml."),
    include_example: bool = typer.Option(
        True,
        "--include-example/--skip-example",
        help="Display sample payload when prompts/examples/{name} exists.",
    ),
    include_recent: bool = typer.Option(
        True,
        "--include-recent/--skip-recent",
        help="Display the most recent workflow payload stored in history.",
    ),
) -> None:
    """Show the raw prompt template and optional sample payloads."""

    cli_module = _cli()
    prompt_service = cli_module.PromptService()
    try:
        template = prompt_service.get_prompt(prompt)
    except (KeyError, FileNotFoundError) as exc:
        available = ", ".join(sorted(prompt_service.registry))
        message = f"{exc}. Available prompts: {available}" if available else str(exc)
        raise typer.BadParameter(message) from exc

    example_payload: Optional[tuple[str, str]] = None
    if include_example:
        registry = prompt_service.registry
        prompt_path = registry.get(prompt)
        if prompt_path:
            example_dir = prompt_path.parent / "examples"
            example_payload = _load_prompt_example(example_dir, prompt)

    recent_payload: Optional[tuple[str, str]] = None
    if include_recent:
        state = cli_module.get_state()
        recent_payload = _find_recent_workflow_payload(state.context_history, prompt)

    render_prompt_sample(
        prompt,
        template,
        example=example_payload,
        recent=recent_payload,
    )


def _load_prompt_example(example_dir: Path, prompt: str) -> Optional[tuple[str, str]]:
    if not example_dir.exists():
        return None
    for extension in (".json", ".yaml", ".yml", ".txt", ".md"):
        candidate = example_dir / f"{prompt}{extension}"
        if candidate.exists():
            return ("Sample payload", candidate.read_text(encoding="utf-8"))
    return None


def _find_recent_workflow_payload(
    history: list[dict[str, Any]], prompt: str
) -> Optional[tuple[str, str]]:
    for record in reversed(history):
        if isinstance(record, dict) and record.get("workflow") == prompt:
            try:
                serialized = json.dumps(record, indent=2, ensure_ascii=False)
            except TypeError:
                serialized = str(record)
            return ("Recent workflow payload", serialized)
    return None


__all__ = [
    "settings_show",
    "settings_show_prompt",
    "settings_update_provider",
    "settings_update_workflow",
]
