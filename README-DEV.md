# Cognitive Technique Mapper — Developer Guide

## Current Status
- Implemented Typer-based local CLI with a local-only FastAPI wrapper for registered workflow automation; first-release acceptance remains pending a controlled live-provider smoke of `describe` → `analyze` → `explain`.
- SQLite catalog seeded from `data/techniques.json`; local Chroma embeddings are optional and require the explicit `chroma` extra.
- Preference tracking, history, feedback loops, and Markdown reporting already implemented; new work typically adds workflows, prompts, or catalog management tooling.

## Environment & Tooling
- Python 3.12+ with `uv`; `pyproject.toml` declares dependencies and the tracked `uv.lock` fixes their resolution.
- Base local CLI: `uv sync --frozen`. Optional local Chroma embeddings: `uv sync --extra chroma --frozen`. Full development environment: `uv sync --all-extras --frozen`.
- Linting/formatting: Ruff with 88-character line length and import-order checks.
- Type checking: Pyright in strict mode (config in `pyrightconfig.json`).
- Tests: `uv run --all-extras --frozen pytest -n auto --cov=src --cov-fail-under=85 --disable-warnings -q`.

## Configuration & Secrets
All runtime credentials are loaded from environment variables (optionally via `.env`). The tracked default template uses direct OpenAI with `gpt-4o-mini` and `text-embedding-3-small`, so set:

- `OPENAI_API_KEY` — API key for the default `config.example/` configuration.

Alternative providers are opt-in through local configuration:

- `AZURE_API_BASE` and `AZURE_OPENAI_KEY` — required after selecting `azure_openai` and Azure deployments in local `config/models.yaml`.
- `ANTHROPIC_API_KEY` — required only for workflows explicitly mapped to Anthropic models.

Tracked `config.example/` holds credential-free defaults for models, providers, database paths, and logging. On first use CTM copies all templates into ignored local `config/*.yaml` only when the default `config/` directory is wholly absent; it never overwrites an existing config or a path selected with `CTM_CONFIG_PATH`. Model/provider wiring then lives in local `config/models.yaml` and `config/providers.yaml`; database paths and logging defaults live in local `config/database.yaml` and `config/settings.yaml`. Always validate edits by running `python -m src.cli settings show`.

## Getting Started
```bash
git clone <repo-url>
cd CognitiveTechniqueMapper
uv sync --all-extras --frozen
cp .env.example .env  # fill in provider keys listed above

# Initialize or refresh technique data
uv run --frozen python -m src.cli refresh --rebuild-embeddings

# Smoke-test the primary workflow
uv run --frozen python -m src.cli describe "Need a framework for prioritizing conflicting projects."
uv run --frozen python -m src.cli analyze --show-candidates
uv run --frozen python -m src.cli explain

# Optional local-only API surface
uv run --frozen uvicorn src.api:app --reload --host 127.0.0.1
```

## Local API Contract
- The FastAPI and optional GraphQL surfaces are loopback-only development tools; do not expose them beyond `127.0.0.1` without a separate authentication, authorization, CORS, deployment, and threat-model slice.
- `/workflows` lists the authoritative HTTP contract. Only registered orchestrator workflows may be executed through `POST /workflow/{name}`.
- `POST /workflow/detect_technique` requires a JSON object with non-empty `problem_description`; optional `include_diagnostics` enables comparison diagnostics. For example: `{"problem_description":"Prioritize two projects.","include_diagnostics":false}`.
- Invalid JSON or invalid workflow context returns HTTP 400; unknown workflows return HTTP 404. Internal workflow failures are logged server-side and return a generic HTTP 500 payload without raw exception text.
- `explain` / `explain_logic` stays CLI-only and must return HTTP 404 rather than bypassing the orchestrator.

## Directory Highlights
- `src/cli/` — Typer entrypoint, runtime wiring, and command modules (keep commands thin; reuse services from `src/core/`).
- `src/core/` — Orchestrator, LLM gateway, config loader, and shared services.
- `src/services/` — Technique selection, embeddings, plan generation, and supporting domain services.
- `src/db/` — SQLite/Chroma clients and data initializers.
- `prompts/` — Prompt templates consumed by workflows.
- `tests/` — Pytest suites covering config loaders, database access, workflows, and prompt validation helpers.

## Feature Insights & Extension Points
- **Workflows:** Declared under `src/workflows/` and orchestrated via the CLI/API. To add one, define the workflow, wire it in `src/core` services, then expose Typer/FastAPI entrypoints.
- **Technique Catalog:** Managed through `python -m src.cli techniques ...` commands. When modifying `data/techniques.json`, rerun `refresh` (optionally `--skip-embeddings` if embeddings are already current).
- **Reports & History:** `report` command pulls the latest recommendation, explanation, simulation, and comparisons into Markdown. History/preferences/feedback commands persist state in SQLite and JSON files under `data/`.
- **Logging & Error Handling:** Use structured logging (via `structlog` or stdlib JSON handlers configured in settings). Wrap external calls with timeouts and apply `tenacity` retries (`wait_exponential(multiplier=1, min=4, max=10)`, max 5 attempts).

## Quality Gates
1. `uv run --all-extras --frozen ruff check src tests` — linting and import ordering.
2. `uv run --all-extras --frozen ruff format --check src tests` — formatting (line length 88).
3. `uv run --all-extras --frozen pyright` — strict type checking.
4. `uv run --all-extras --frozen pytest -n auto --cov=src --cov-fail-under=85 --disable-warnings -q` — coverage and regression checks.

## Troubleshooting
- **Missing embeddings:** Remove `embeddings/` and rerun `uv run --frozen python -m src.cli refresh --rebuild-embeddings`.
- **Config drift:** Run `uv run --frozen python -m src.cli settings show` to inspect current values; use `settings update-workflow`/`settings update-provider` for adjustments.
- **Provider issues:** When parameters are rejected (e.g., unsupported temperature), edit `config/models.yaml` or set `litellm.drop_params = True` in the relevant config block.

For additional CLI-specific contribution notes, see [docs/cli-contrib.md](docs/cli-contrib.md). Keep `CHANGELOG.md` updated with ISO-8601 dates for every notable change.
