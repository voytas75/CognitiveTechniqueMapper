# Cognitive Technique Mapper — Developer Guide

## Current Status
- Mature Typer-based CLI with FastAPI mirror for workflow automation.
- SQLite catalog seeded from `data/techniques.json` and optional Chroma embeddings for semantic retrieval.
- Preference tracking, history, feedback loops, and Markdown reporting already implemented; new work typically adds workflows, prompts, or catalog management tooling.

## Environment & Tooling
- Python 3.12+ with `uv` for dependency syncing (`uv pip sync requirements.lock`).
- Recommended local virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Linting/formatting: `ruff --fix` and `black --line-length 88` (imports sorted via isort profile bundled with Ruff).
- Type checking: `pyright` in strict mode (config in `pyrightconfig.json`).
- Tests: `pytest -n auto --cov=src --cov-fail-under=85 --disable-warnings -q`.

## Configuration & Secrets
All runtime credentials are loaded from environment variables (optionally via `.env`). Set at least the following before invoking workflows:

- `AZURE_API_BASE` — Azure OpenAI base URL when using Azure-backed models.
- `AZURE_OPENAI_KEY` — API key for the Azure deployment.
- `OPENAI_API_KEY` — Direct OpenAI key when bypassing Azure or using fallback models.
- `ANTHROPIC_API_KEY` — Required for workflows mapped to Anthropic models.

Model/provider wiring lives under `config/models.yaml` and `config/providers.yaml`. Settings such as database paths or logging defaults live in `config/settings.yaml`. Always validate edits by running `python -m src.cli settings show`.

## Getting Started
```bash
git clone <repo-url>
cd CognitiveTechniqueMapper
python -m venv .venv && source .venv/bin/activate
uv pip sync requirements.lock
cp .env.example .env  # fill in provider keys listed above

# Initialize or refresh technique data
python -m src.cli refresh --rebuild-embeddings

# Smoke-test the primary workflow
python -m src.cli describe "Need a framework for prioritizing conflicting projects."
python -m src.cli analyze --show-candidates
python -m src.cli explain

# Optional API surface
uvicorn src.api:app --reload
```

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
1. `ruff --fix` — import sorting and lint fixes.
2. `black .` — consistent formatting (line length 88).
3. `pyright` — strict typing; no regressions accepted.
4. `pytest -n auto --cov=src --cov-fail-under=85 --disable-warnings -q` — enforce coverage and regression checks.

## Troubleshooting
- **Missing embeddings:** Remove `embeddings/` and rerun `python -m src.cli refresh --rebuild-embeddings`.
- **Config drift:** Run `python -m src.cli settings show` to inspect current values; use `settings update-workflow`/`settings update-provider` for adjustments.
- **Provider issues:** When parameters are rejected (e.g., unsupported temperature), edit `config/models.yaml` or set `litellm.drop_params = True` in the relevant config block.

For additional CLI-specific contribution notes, see [docs/cli-contrib.md](docs/cli-contrib.md). Keep `CHANGELOG.md` updated with ISO-8601 dates for every notable change.
