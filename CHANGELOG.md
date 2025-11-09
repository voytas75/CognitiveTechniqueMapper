# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Structured `detect_technique` responses with JSON parsing and automatic plan generation.
- `settings` subcommands to edit workflow models and provider metadata with optional interactive prompts.
- `refresh` CLI command to reload the techniques dataset and rebuild embeddings on demand.

### Changed
- Align `max_tokens` handling with LiteLLM per-model limits to avoid provider rejections.
- Fix `settings` CLI command by serializing workflow configs via dataclass helpers.
- Extend `settings` CLI output with embedding configuration metadata.
- Adopt `pyproject.toml` with uv-generated `requirements.lock` for dependency management (single lock covers runtime and dev dependencies).
- Prevent `explain` command from raising a `NameError` during module import by scoping console rendering correctly.
