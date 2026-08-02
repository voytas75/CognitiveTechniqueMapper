# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Security
- Replaced the stale locked dependency set with a tracked `uv.lock` and updated all currently fixable `pip-audit` findings. `chromadb` remains at `1.5.9` with CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c pending an upstream fixed release; CTM uses only its local `PersistentClient` path.

### Fixed
- Restored the declared `src.cli.TechniqueSearchService` compatibility export and explicitly listed `ConfigEditor` in the CLI public surface.
- Moved the explicit `create_search_service`, `create_catalog_service`, and `create_initializer` factories into `src.cli.service_factories`; `src.cli.runtime` keeps compatibility re-exports while retaining only runtime orchestration.

### Changed
- Formally limited the FastAPI and optional GraphQL surfaces to loopback-only development use; REST exposes registered orchestrator workflows only and does not return raw workflow exceptions.
- Made `pyproject.toml` plus `uv.lock` the reproducible dependency source of truth; full development setup now uses `uv sync --all-extras --frozen`.
- Added Black and isort to the development extra to match the documented quality gates.

### Removed
- Removed mypy dependency and configuration; Pyright now serves as the sole static type checker.

## [0.2.0] - 2025-11-09

### Added
- Structured `detect_technique` responses with JSON parsing and automatic plan generation.
- `settings` subcommands to edit workflow models and provider metadata with optional interactive prompts.
- `refresh` CLI command to reload the techniques dataset and rebuild embeddings on demand.
- Structured `explain_logic` prompt with JSON parsing plus CLI rendering of key factors, risks, and next steps.
- SQLite-backed feedback persistence with preloaded history and summaries.
- `analyze --show-candidates` flag to display similarity-ranked technique matches.
- `simulate` workflow and CLI command for scenario walkthroughs with cautions and follow-up actions.
- `compare` workflow to contrast candidate techniques and surface the best alternative.
- Preference service that converts feedback into personalization signals for recommendations and prompts.
- Structured JSON logging with workflow duration metrics.
- Tenacity-powered retries and timeouts for LLM and embedding gateways.
- `techniques` CLI group for listing, adding, updating, and removing catalog entries.
- `history` CLI group to inspect or clear session records.
- `preferences` CLI group to review/export/reset personalization signals.
- `techniques import`/`techniques export` commands for bulk catalog management.
- Comprehensive CLI regression tests with ≥85% coverage and CI enforcement for linting and type checks.

### Changed
- Align `max_tokens` handling with LiteLLM per-model limits to avoid provider rejections.
- Fix `settings` CLI command by serializing workflow configs via dataclass helpers.
- Extend `settings` CLI output with embedding configuration metadata.
- Adopt `pyproject.toml` with uv-generated `requirements.lock` for dependency management (single lock covers runtime and dev dependencies).
- Prevent `explain` command from raising a `NameError` during module import by scoping console rendering correctly.
- Lazily initialize CLI runtime dependencies to improve import ergonomics and testability.
- Added GitHub Actions workflow plus mypy/pyright strict configuration for core modules.
