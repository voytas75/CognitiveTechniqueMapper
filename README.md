# Cognitive Technique Mapper

Cognitive Technique Mapper (CTM) is a modular CLI application that pairs problem descriptions with the most suitable cognitive or problem-solving techniques. It combines a structured SQLite knowledge base, optional Chroma vector search, and workflow-specific LLM reasoning orchestrated through `litellm`.

---

## 🚀 Features

### Core Workflows

- **Technique detection & planning:** `analyze` blends vector search with LLM reasoning, parses structured responses, and calls the plan generator for actionable steps.
- **Justification & exploration:** `explain`, `simulate`, and `compare` commands surface rationale, scenario walkthroughs, and trade-off comparisons for the selected technique.
- **Guided CLI flow:** Top-level commands (`describe`, `analyze`, `explain`, `simulate`, `compare`, `feedback`, `interactive-flow`) provide an end-to-end session from problem capture to feedback or manual decision trees.
- **Shareable reports:** `report` assembles the latest recommendation, explanation, simulation, and comparison into a Markdown snapshot for stakeholders.

### Personalization & History

- **Preference-aware recommendations:** Feedback ratings train a lightweight preference model that adjusts rankings and summaries.
- **Feedback loops:** `feedback` persists entries to SQLite and summarizes recent sentiment; `history` commands inspect or clear session context; `preferences` exposes or resets personalization signals.

### Configuration & Catalog Management

- **Config-driven setup:** YAML files under `config/` define app metadata, database paths, provider credentials, and workflow model mappings.
- **Interactive configuration tools:** `settings show`, `settings update-workflow`, and `settings update-provider` support inline edits (with optional interactive prompts).
- **Dataset lifecycle:** `TechniqueDataInitializer` seeds SQLite/Chroma from `data/techniques.json`; `refresh` reloads datasets and optionally rebuilds embeddings.
- **Catalog administration:** `techniques` subcommands list, add, update, remove, import, and export techniques with automatic synchronization to vectors and storage.
- **Persistent storage:** Session state (`data/state.json`) and feedback/preference tables ensure continuity across CLI runs.

---

## 🗂️ Project Structure (excerpt)

```
.
├── config/                # settings.yaml, models.yaml, providers.yaml
├── data/                  # techniques.json, state.json, SQLite DB
├── embeddings/            # Chroma DB persistence
├── prompts/               # Prompt templates and registry
├── src/
│   ├── cli/               # CLI entrypoint, commands, runtime wiring
│   ├── core/              # Orchestrator, LLM gateway, config loader, etc.
│   ├── db/                # SQLite and Chroma clients
│   ├── services/          # Technique selector, embeddings, plan generator…
│   └── workflows/         # Workflow definitions wired into the orchestrator
└── tests/                 # Pytest coverage for config, DB, workflows, prompts
```

---

## 📦 Prerequisites

- Python 3.12+
- Virtual environment (`python -m venv .venv` then activate)
- Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv pip sync requirements.lock
```

- The lock file already captures both runtime and development dependencies (e.g., pytest).

- Set provider credentials (example `.env` entry):

```
AZURE_API_BASE="https://<your-endpoint>.openai.azure.com/"
AZURE_OPENAI_KEY="..."
OPENAI_API_KEY="..."
ANTHROPIC_API_KEY="..."
```

Export or load them in your shell before running the CLI. Providers and models are configured in `config/models.yaml` and `config/providers.yaml`.

---

## ⚙️ Configuration

- `config/settings.yaml` — app metadata & logging defaults.
- `config/database.yaml` — SQLite path and Chroma persistence directory.
- `config/models.yaml` — workflow → model mappings, embedding model, provider defaults.
- `config/providers.yaml` — provider endpoints and env var names for API keys.

All files are loaded through `ConfigService` and cached for reuse.

---

## ▶️ Running the CLI

```bash
python -m src.cli describe "I can't decide between two great job offers."
python -m src.cli analyze
python -m src.cli explain
python -m src.cli simulate --scenario "Negotiating counter-offers"
python -m src.cli compare --limit 3
python -m src.cli settings show
python -m src.cli settings update-workflow detect_technique --model openai/gpt-4.1 --temperature 0.4
python -m src.cli refresh --skip-embeddings
python -m src.cli techniques refresh --rebuild-embeddings
python -m src.cli techniques status
python -m src.cli analyze --show-candidates
python -m src.cli feedback "Loved the recommendation" --rating 5 --technique "Decisional Balance"
python -m src.cli analyze --log-level DEBUG  # temporary verbose logging
python -m src.cli report --output latest-report.md
python -m src.cli interactive-flow --show-tree
```

---

## 🌐 Running the HTTP API

The same workflows are now exposed through a lightweight FastAPI service so
other tools or front‑ends can integrate without spawning the CLI.

### Prerequisites

```bash
# Install FastAPI / Uvicorn (already pinned in pyproject.toml)
uv pip sync requirements.lock        # or `uv pip install .[graphql]` for GraphQL
```

### Start the server

```bash
# Development (auto‑reload on file changes)
uvicorn src.api:app --reload

# Production example (4 workers on port 8080)
uvicorn src.api:app --host 0.0.0.0 --port 8080 --workers 4
```

### Endpoints

| Method | Path                         | Description                              |
|--------|------------------------------|------------------------------------------|
| GET    | `/health`                   | Liveness probe                           |
| GET    | `/workflows`                | List registered workflow names           |
| POST   | `/workflow/{name}`          | Execute workflow with JSON *context*     |
| any    | `/docs`, `/redoc`           | Interactive Swagger / ReDoc API docs     |
| any    | `/graphql` (optional)       | GraphiQL playground when `strawberry` is installed |

#### Example request

```bash
curl -X POST http://127.0.0.1:8000/workflow/detect_technique \
     -H "Content-Type: application/json" \
     -d '{"problem_description": "I can\'t decide between two job offers."}'
```

GraphQL equivalent (requires `pip install .[graphql]`):

```graphql
query {
  runWorkflow(
    name: "detect_technique",
    context: { problem_description: "Can't decide between two job offers." }
  )
}
```

This returns the same JSON structure the CLI prints, making it easy to embed
CTM recommendations in dashboards or chatbots.

Notes:
- `describe` saves the problem description and persists it to `data/state.json`.
- `analyze` runs the `detect_technique` workflow (LLM + vector search).
- `explain` requests the `explain_logic` workflow to justify the recommendation.
- `settings show` prints a JSON snapshot of the current config values. Use `update-workflow` and `update-provider` to make inline edits (supports `--interactive`).
- `feedback` stores feedback and summarizes recent entries via LLM.
- `simulate` replays the current recommendation with scenario variations and safeguards.
- `compare` analyzes the candidate shortlist, highlighting trade-offs and flow-on guidance.
- `refresh` reloads `data/techniques.json`, replaces existing rows, and optionally rebuilds embeddings.
- `techniques refresh` refreshes catalog data via the `techniques` subcommand (same effect as the top-level command but scoped to catalog administration).
- `techniques status` reports dataset statistics and whether embeddings exist in the configured vector store, so you can confirm catalog health before running workflows.
- `analyze --show-candidates` prints the shortlist with similarity scores for transparency.
- `interactive-flow` walks the built-in decision tree interactively and surfaces the corresponding technique (optionally printing the tree structure with `--show-tree`).

If the LLM provider rejects a parameter (e.g., unsupported temperature), adjust `config/models.yaml` or set `litellm.drop_params = True` before running the CLI.

---

## 🧭 CLI Architecture

The CLI lives under `src/cli/` and is broken into focused modules so commands remain easy to extend:

- `__init__.py` — Typer root app. Wires command groups and re-exports runtime helpers for compatibility.
- `__main__.py` — Enables `python -m src.cli`.
- `runtime.py` — Builds the orchestrator, initializes services, and exposes helpers like `get_state()` and `create_catalog_service()`.
- `state.py` — Dataclass for persisted CLI state (`data/state.json` by default).
- `commands/` — One module per concern (`core.py`, `settings.py`, `techniques.py`, `history.py`, `preferences.py`). Each module focuses on Typer command bindings only.
- `renderers.py` — Rich output helpers shared by commands.
- `reporting.py` — Builds shareable Markdown reports from the current session state.
- `utils.py` — Small wrappers (logging overrides, prompt helpers) that respect monkeypatched dependencies.
- `io.py` — Shared `Console` instance to keep Rich output consistent.

### Adding a new command

1. Decide whether it belongs in `commands/core.py` (primary flows) or a dedicated module under `commands/`.
2. Implement the command function, reusing helpers such as `apply_log_override`, `get_orchestrator()`, and `render_*` utilities.
3. Register the command in `src/cli/__init__.py` via `app.command()` or the relevant sub-app (e.g., `settings_app`).
4. Update or create tests using fixtures from `tests/helpers/cli.py` to simulate orchestrator behaviour and mute console output.

### Extending runtime behaviour

- Use `runtime.py` for wiring new services/workflows. The `_resolve_dependency` helper lets tests override components by monkeypatching attributes on `src.cli`.
- Persist additional state by extending `AppState` in `state.py`; remember to update `save()` and `load()`.
- Keep render logic in `renderers.py` so Typer command modules stay thin.

This separation keeps individual files well below the 300–400 line target in `AGENTS.md` and makes it obvious where to place new functionality.

---

## 🧠 Knowledge Base & Embeddings

1. Seed techniques in `data/techniques.json`.
2. On startup, `TechniqueDataInitializer` populates SQLite (if empty) and attempts to embed each technique using the configured embedding model. When Chroma is available, embeddings are stored and used for semantic retrieval.
3. If embedding APIs are unreachable, a deterministic hash-based fallback keeps the system functional (with reduced semantic quality).

To extend the library:
- Add rows to `data/techniques.json`.
- Run `python -m src.cli refresh` to reload the dataset (use `--skip-embeddings` when you only need SQLite updates).

---
## 🧪 Testing

After syncing dependencies with `uv pip sync requirements.lock`, run:

```bash
pytest
```

Coverage includes config loading, SQLite operations, prompt registry validation, data initialization, and orchestrator execution.

---

## 🔄 Development Notes

- Use the `apply_patch` helper (if working in the Codex CLI) for targeted edits.
- Avoid committing provider secrets. `.env` is listed for guidance only.
- The project favors clear, traceable service layers; keep additions modular and register new workflows with the orchestrator.
- For future enhancements consider:
  - Additional workflows (e.g., scenario simulation).
  - Extended prompts and richer feedback analysis.
- See `docs/cli-contrib.md` for CLI-specific contribution tips and testing expectations.

---

## 📄 License

Cognitive Technique Mapper is distributed under the MIT License. Review `LICENSE` for the complete terms, including usage permissions, conditions, and disclaimers.

---

Happy mapping! 🎯✨
