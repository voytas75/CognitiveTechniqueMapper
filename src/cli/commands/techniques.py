"""Technique catalog management commands."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional

import typer
from rich.panel import Panel

from src.cli.io import console
from src.cli.renderers import render_technique_table
from src.cli.utils import apply_log_override


def _cli():
    return sys.modules["src.cli"]


def techniques_list() -> None:
    """Display techniques stored in the knowledge base."""

    catalog, sqlite_client = _cli()._create_catalog_service()
    try:
        entries = catalog.list()
    finally:
        sqlite_client.close()

    render_technique_table(entries)


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

    catalog, sqlite_client = _cli()._create_catalog_service()
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
        _cli()._refresh_runtime()
        console.print_json(data={"technique": entry})


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

    catalog, sqlite_client = _cli()._create_catalog_service()
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
        _cli()._refresh_runtime()
        console.print_json(data={"technique": entry})


def techniques_remove(
    name: str = typer.Argument(..., help="Technique name to remove."),
) -> None:
    """Remove a technique from the catalog and refresh runtime services."""

    catalog, sqlite_client = _cli()._create_catalog_service()
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
        _cli()._refresh_runtime()
        console.print(f"[green]Removed technique '{name}'.[/]")


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

    catalog, sqlite_client = _cli()._create_catalog_service()
    try:
        path, count = catalog.export_to_file(file)
    except Exception as exc:
        console.print(f"[red]Export failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        sqlite_client.close()

    console.print_json(data={"file": str(path), "count": count})


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
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Import techniques from a JSON file using append or replace semantics."""

    apply_log_override(log_level)

    catalog, sqlite_client = _cli()._create_catalog_service()
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

    _cli()._refresh_runtime()
    console.print_json(data={"mode": mode.lower(), **summary})


def techniques_refresh(
    rebuild_embeddings: bool = typer.Option(
        True,
        "--rebuild-embeddings/--skip-embeddings",
        help="Recompute embeddings after refreshing the dataset.",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Reload the canonical dataset and optionally rebuild embeddings."""

    apply_log_override(log_level)

    initializer, sqlite_client = _cli()._create_initializer()
    try:
        initializer.refresh(rebuild_embeddings=rebuild_embeddings)
    except Exception as exc:  # pragma: no cover - dependent on optional services
        console.print(f"[red]Technique refresh failed: {exc}[/]")
        raise typer.Exit(code=1) from exc
    finally:
        sqlite_client.close()

    _cli()._refresh_runtime()
    console.print(Panel("Technique dataset refreshed.", title="Techniques"))


__all__ = [
    "techniques_add",
    "techniques_export",
    "techniques_import",
    "techniques_refresh",
    "techniques_list",
    "techniques_remove",
    "techniques_update",
]
