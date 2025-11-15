"""Technique catalog management commands."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import typer
from rich.panel import Panel

from src.cli.io import console
from src.cli.renderers import render_coverage_summary, render_technique_table
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


def techniques_gaps(
    threshold: int = typer.Option(
        2,
        "--threshold",
        "-t",
        min=0,
        help="Minimum techniques per category before it is flagged as a gap.",
    ),
    include_preferences: bool = typer.Option(
        True,
        "--include-preferences/--skip-preferences",
        help="Display stored feedback trends when available.",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Highlight categories with sparse technique coverage."""

    apply_log_override(log_level)

    catalog, sqlite_client = _cli()._create_catalog_service()
    try:
        entries = catalog.list()
    finally:
        sqlite_client.close()

    effective_threshold = max(threshold, 0)
    category_summary = _aggregate_categories(entries)

    preference_data: Dict[str, Any] = {}
    if include_preferences and category_summary:
        state = _cli().get_state()
        preference_service = getattr(state, "preference_service", None)
        if preference_service is not None:
            preference_data = _preference_category_stats(preference_service)

    records = _build_gap_records(
        category_summary,
        effective_threshold,
        preference_data,
    )
    render_coverage_summary(records, threshold=effective_threshold)


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
    "techniques_gaps",
    "techniques_import",
    "techniques_status",
    "techniques_refresh",
    "techniques_list",
    "techniques_remove",
    "techniques_update",
]


def _format_timestamp(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def techniques_status(
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Override logging level for this invocation.",
    ),
) -> None:
    """Display dataset and embedding status for the technique catalog."""

    apply_log_override(log_level)

    catalog, sqlite_client = _cli()._create_catalog_service()
    try:
        entries = catalog.list()
        dataset_path_value = getattr(catalog, "dataset_path", None)
        dataset_path = Path(dataset_path_value) if dataset_path_value else None
        dataset_exists = dataset_path.exists() if dataset_path is not None else False
        dataset_info: dict[str, Any] = {
            "count": len(entries),
            "path": str(dataset_path) if dataset_path else None,
            "exists": dataset_exists,
        }
        timestamp = _format_timestamp(dataset_path if dataset_exists else None)
        if timestamp:
            dataset_info["last_modified"] = timestamp

        sqlite_path = None
        if hasattr(sqlite_client, "path"):
            sqlite_path = str(getattr(sqlite_client, "path"))
        elif hasattr(sqlite_client, "_db_path"):
            sqlite_path = str(getattr(sqlite_client, "_db_path"))
        if sqlite_path:
            dataset_info["sqlite_path"] = sqlite_path

        chroma_client = getattr(catalog, "chroma_client", None)
        embedding_info: dict[str, Any] = {"enabled": bool(chroma_client)}
        if chroma_client:
            try:
                identifiers = chroma_client.list_ids()
            except Exception as exc:  # pragma: no cover - optional dependency path
                embedding_info["error"] = str(exc)
            else:
                embedding_info["count"] = len(identifiers or [])
        else:
            embedding_info["count"] = 0

    finally:
        sqlite_client.close()

    console.print_json(
        data={
            "dataset": dataset_info,
            "embeddings": embedding_info,
        }
    )


def _aggregate_categories(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for entry in entries:
        raw = entry.get("category")
        display = raw.strip() if isinstance(raw, str) and raw.strip() else "Uncategorized"
        key = display.casefold()
        bucket = buckets.setdefault(key, {"category": display, "count": 0})
        if bucket["category"] == "Uncategorized" and display != "Uncategorized":
            bucket["category"] = display
        bucket["count"] += 1
    return buckets


def _preference_category_stats(preference_service: Any) -> dict[str, dict[str, Any]]:
    try:
        profile = preference_service.export_profile()
    except Exception:  # pragma: no cover - defensive against custom services
        return {}

    categories = getattr(profile, "categories", {})
    if not isinstance(categories, dict):
        return {}

    stats: dict[str, dict[str, Any]] = {}
    for name, bucket in categories.items():
        if not isinstance(name, str) or not isinstance(bucket, dict):
            continue
        key = name.strip().casefold()
        if not key:
            key = "uncategorized"
        stats[key] = {
            "avg_rating": _safe_average(bucket),
            "negative_ratio": _safe_negative_ratio(bucket),
        }
    return stats


def _safe_average(bucket: dict[str, Any]) -> Optional[float]:
    rating_count = bucket.get("rating_count")
    rating_sum = bucket.get("rating_sum")
    try:
        if rating_count and float(rating_count):
            return float(rating_sum or 0.0) / float(rating_count)
    except (TypeError, ZeroDivisionError):  # pragma: no cover - guard rails
        return None
    return None


def _safe_negative_ratio(bucket: dict[str, Any]) -> Optional[float]:
    count = bucket.get("count")
    negatives = bucket.get("negatives")
    try:
        if count and float(count):
            return float(negatives or 0.0) / float(count)
    except (TypeError, ZeroDivisionError):  # pragma: no cover - guard rails
        return None
    return None


def _build_gap_records(
    categories: dict[str, dict[str, Any]],
    threshold: int,
    preference_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not categories:
        return []

    records: list[dict[str, Any]] = []
    for key, bucket in categories.items():
        count = bucket.get("count", 0)
        pref_stats = preference_data.get(key)
        avg_rating = pref_stats.get("avg_rating") if pref_stats else None
        negative_ratio = pref_stats.get("negative_ratio") if pref_stats else None

        flags: list[str] = []
        if threshold and count < threshold:
            flags.append("⚠ Below target")
        if negative_ratio is not None and negative_ratio >= 0.5:
            flags.append("⚠ Negative trend")
        status = "OK" if not flags else " / ".join(dict.fromkeys(flags))

        records.append(
            {
                "category": bucket.get("category", "Uncategorized"),
                "count": int(count),
                "status": status,
                "avg_rating": avg_rating,
                "negative_ratio": negative_ratio,
            }
        )

    records.sort(key=lambda item: (item["count"], str(item["category"]).casefold()))
    return records
