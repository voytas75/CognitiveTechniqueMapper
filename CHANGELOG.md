# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

### Changed
- Align `max_tokens` handling with LiteLLM per-model limits to avoid provider rejections.
- Fix `settings` CLI command by serializing workflow configs via dataclass helpers.
- Extend `settings` CLI output with embedding configuration metadata.
- Adopt `pyproject.toml` with uv-generated `requirements.lock` for dependency management (single lock covers runtime and dev dependencies).
- Prevent `explain` command from raising a `NameError` during module import by scoping console rendering correctly.
