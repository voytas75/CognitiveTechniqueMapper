"""Read-only diagnostics for CTM configuration and catalog stores."""

import json
import os
import shutil
import sqlite3
import tempfile
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


def apply_safe_fixes(
    project_root: Path, config_path: Path, *, custom_config: bool
) -> dict[str, list[str]]:
    """Bootstrap only a wholly absent default configuration directory."""
    result: dict[str, list[str]] = {"applied": [], "errors": []}
    if custom_config or config_path.exists():
        return result
    templates = project_root / "config.example"
    missing = [name for name in CONFIG_FILENAMES if not (templates / name).is_file()]
    if missing:
        result["errors"].append(f"missing config templates: {', '.join(missing)}")
        return result
    staging = Path(tempfile.mkdtemp(prefix=".config.doctor-", dir=config_path.parent))
    try:
        for name in CONFIG_FILENAMES:
            shutil.copy2(templates / name, staging / name)
        staging.replace(config_path)
        result["applied"].append("bootstrapped_default_config")
    except OSError as exc:
        result["errors"].append(f"config bootstrap failed: {exc}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return result


def doctor(
    fix: bool = typer.Option(
        False, "--fix", help="Apply safe configuration-only fixes."
    ),
) -> None:
    """Report health and optionally bootstrap only a missing default config."""
    root = PROJECT_CONFIG_PATH.parent
    custom_path = os.environ.get("CTM_CONFIG_PATH")
    config = Path(custom_path) if custom_path else root / "config"
    fix_result = (
        apply_safe_fixes(root, config, custom_config=bool(custom_path))
        if fix
        else {"applied": [], "errors": []}
    )
    report = inspect_doctor(project_root=root, config_path=config)
    report["fix"] = {"requested": fix, **fix_result}
    if not report["stores"]["consistent"]:
        report["manual_actions"] = [
            "python -m src.cli techniques refresh --rebuild-embeddings"
        ]
    console.print_json(data=report)
    if not report["ok"] or fix_result["errors"]:
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
