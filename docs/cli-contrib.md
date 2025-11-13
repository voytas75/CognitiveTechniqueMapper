# Contributing to the CLI

The Cognitive Technique Mapper CLI is organised as a small package under `src/cli/`. Every module has a single responsibility so that new commands and runtime wiring stay easy to navigate.

## Module Cheat Sheet

| Module | Purpose |
| --- | --- |
| `__init__.py` | Typer root app. Registers command groups and re-exports helpers. |
| `__main__.py` | Allows `python -m src.cli`. |
| `runtime.py` | Builds services, orchestrator, and provides `get_state()`/`create_*` helpers. |
| `state.py` | `AppState` dataclass and persistence helpers (`data/state.json` by default). |
| `commands/` | Typer command modules (`core.py`, `settings.py`, `techniques.py`, `history.py`, `preferences.py`). |
| `renderers.py` | Rich output helpers shared by commands. |
| `reporting.py` | Utilities that convert session state into shareable Markdown reports. |
| `utils.py` | Lightweight helpers (log overrides, prompts) that respect monkeypatched dependencies. |
| `io.py` | Shared Rich `Console` instance. |

## Adding a Command

1. **Pick a module:** Most user-facing flows live in `commands/core.py`. Admin functionality belongs in a dedicated module within `commands/`.
2. **Keep logic focused:** Command functions should gather inputs, delegate to services via `get_orchestrator()`/`get_state()`, and use renderers for output.
3. **Reuse helpers:** Call `apply_log_override`, `_create_*` helpers, and renderer functions rather than re-implementing boilerplate.
4. **Register the command:** Add it to the appropriate Typer app in `src/cli/__init__.py`.
5. **Write tests:** Use fixtures from `tests/helpers/cli.py` (`make_cli_runtime`, `RecordingOrchestrator`, `mute_console`) to simulate responses without hitting real services.

## Runtime Changes

- Extend `runtime.py` when introducing new services or workflows. Use `_resolve_dependency` so tests can override implementations by monkeypatching attributes on `src.cli`.
- Update `AppState` in `state.py` when persisting additional session data. Remember to adjust `save()` and `load()` behaviour.
- Prefer keeping display logic in `renderers.py` to ensure command modules remain small (target: < 100 lines per command module).

## Testing Expectations

- **Unit tests:** Target individual command modules with orchestrator stubs. Helpers under `tests/helpers/cli.py` eliminate duplication.
- **Integration tests:** Add or extend scenarios under `tests/test_cli_integration.py` to exercise full CLI flows via `CliRunner`.
- **Type safety & style:** Run `pyright` and `ruff` (per `AGENTS.md`). Ensure new modules include type hints and follow Google-style docstrings.

Following these patterns will keep the CLI compliant with the single-responsibility and file-size guidance from `AGENTS.md`, while making future enhancements straightforward.
