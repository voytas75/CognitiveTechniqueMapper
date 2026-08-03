"""Read-only diagnostics for CTM configuration and catalog stores."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable, cast

import typer
import yaml

from src.cli.io import console
from src.core.config_loader import CONFIG_FILENAMES, PROJECT_CONFIG_PATH


def inspect_doctor(
    project_root: Path | None = None,
    config_path: Path | None = None,
    chroma_names_loader: Callable[[Path], set[str] | None] | None = None,
) -> dict[str, Any]:
    """Inspect YAML configuration and compare JSON, SQLite, and Chroma names."""
    root = project_root or PROJECT_CONFIG_PATH.parent
    config = config_path or Path(os.environ.get("CTM_CONFIG_PATH", root / "config"))
    config_errors: list[str] = []
    database: dict[str, object] = {}
    for name in CONFIG_FILENAMES:
        try:
            value: object = yaml.safe_load((config / name).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("root must be a YAML mapping")
            mapping = cast(dict[str, object], value)
            database_value = mapping.get("database")
            if name == "database.yaml" and isinstance(database_value, dict):
                database = cast(dict[str, object], database_value)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            config_errors.append(f"{name}: {exc}")
    sqlite_path = root / str(database.get("sqlite_path", "data/techniques.db"))
    chroma_path = root / str(database.get("chromadb_path", "embeddings"))
    errors: list[str] = []
    records: list[dict[str, object]] = []
    source: set[str] = set()
    rows: list[tuple[str]] = []
    sqlite: set[str] = set()
    try:
        raw_records: object = json.loads(
            (root / "data/techniques.json").read_text(encoding="utf-8")
        )
        if not isinstance(raw_records, list):
            raise ValueError("root must be a JSON list")
        records = cast(list[dict[str, object]], raw_records)
        source = {name for item in records if isinstance(name := item.get("name"), str)}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source, records = set(), []
        errors.append(f"source: {exc}")
    try:
        with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as connection:
            rows = cast(
                list[tuple[str]],
                connection.execute("SELECT name FROM techniques").fetchall(),
            )
        sqlite = {row[0] for row in rows}
    except sqlite3.Error as exc:
        sqlite, rows = set(), []
        errors.append(f"sqlite: {exc}")
    loader = chroma_names_loader or _load_chroma_names
    try:
        chroma = loader(chroma_path)
    except Exception as exc:
        chroma = None
        errors.append(f"chroma: {exc}")
    chroma_count = len(chroma) if chroma is not None else 0
    consistent = (
        not errors and source == sqlite and (chroma is None or source == chroma)
    )
    return {
        "ok": not config_errors and consistent,
        "config": {"ok": not config_errors, "errors": config_errors},
        "stores": {
            "source": {"records": len(records), "names": len(source)},
            "sqlite": {"records": len(rows), "names": len(sqlite)},
            "chroma": {
                "enabled": chroma is not None,
                "records": chroma_count,
                "names": chroma_count,
            },
            "consistent": consistent,
            "errors": errors,
        },
    }


def doctor() -> None:
    """Report read-only configuration and catalog health as JSON."""
    report = inspect_doctor()
    console.print_json(data=report)
    if not report["ok"]:
        raise typer.Exit(code=1)


def _load_chroma_names(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    try:
        import chromadb

        collection = chromadb.PersistentClient(path=str(path)).get_collection(
            "techniques"
        )
        snapshot = cast(dict[str, list[str]], collection.get(include=[]))
        return set(snapshot["ids"])
    except ImportError:
        return None
