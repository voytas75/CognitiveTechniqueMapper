# Python Best Practices for application

- Python version: 3.12.
- Code Formatting: All code must be automatically formatted using Black with a max line length of 88. Import ordering must be managed by isort.
- Docstring Standard: Use the Google style docstring format for all functions and classes, including type information for all arguments.
- Testing Quality: A minimum code coverage threshold of 85% must be maintained for all core logic modules.
- Asynchronous Boundary: All long-running I/O operations (network calls, database access) must utilize async/await patterns using libraries like httpx or an async database adapter.
- Integration Workflow: All code changes must pass all continuous integration (CI) checks, achieve 100% type-checking compliance, and require approval from at least one human reviewer before merging.

## Core Principles

### KISS (Keep It Simple, Stupid)

- Choose the simplest solution that meets requirements
- Avoid over-engineering; prefer readable code over clever code
- When in doubt, favor explicit over implicit behavior
- Use built-in Python features before creating custom abstractions

### DRY (Don't Repeat Yourself)

- Extract common functionality into reusable modules
- Use configuration files to avoid hardcoded values

## Code Style and Structure

- Use descriptive module, class, and function names; keep functions under ~100 lines.
- Prefer `dataclasses` or `TypedDict` for structured data over loosely-typed dicts.

## Type Safety and Documentation

- Enable `mypy` (or `pyright`) in strict mode for core modules; add `typing` annotations everywhere.
- Document functions with concise docstrings describing purpose, inputs, and outputs.

### Documentation Maintenance

- Update each Python script's docstring 'updates' section when making significant changes; internal reference docs (e.g., AGENTS.md) are exempt.
- Maintain change log for scripts. Document significant modifications with version and date.
- Use consistent format: `Updates: v{version} - {current date} - {description}`. Do not make up dates, always check current date.
- Keep the `README.md` file updated for the description of application.
- Keep the `CHANGELOG.md` file updated. All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/) and dates use ISO-8601 (YYYY-MM-DD).

## Error Handling and Resilience

- Wrap external calls (network, filesystem, subprocess) with timeouts and explicit exceptions.
- Raise custom exception classes for predictable failure modes; avoid bare `except`.
- Include context in exceptions (ie. `raise ConfigError(f"Invalid API key: {key}")`).
- Implement all retry and backoff logic using the `tenacity` library, configured with exponential backoff and explicit stop/wait conditions.

## Logging and Observability

- Use the standard `logging` module with structured fields. Use ERROR for predictable failures, WARNING for recoverable issues, INFO for successful tool call summaries, and DEBUG for detailed internal reasoning and trace spans.
- Emit debug spans around tool calls and key reasoning steps; keep logs readable and redaction-safe.

## Configuration and Secrets

- Centralize configuration in environment variables or a `pydantic` settings model.
- Never hardcode secrets; load via secret managers or `.env` files excluded from version control.
- Validate configuration eagerly at startup to fail fast with clear messaging.

## Dependency Management

- The single source of truth for all dependencies is `pyproject.toml`. Use `uv` to generate a single, locked dependency file (e.g., `requirements.lock`) from `pyproject.toml`.
- Regularly audit dependencies and patch security vulnerabilities promptly.
- Keep optional integrations optional by guarding imports and adding extras (`pip install .[slack]`).

## Testing Strategy

- Write unit tests for core reasoning logic and tool adapters (`pytest` + fixtures).
- Add integration tests that mock external APIs where possible; use `vcrpy` for HTTP recordings.
- Include regression tests for critical decision flows and prompt templates.

## Performance and Resource Use

- Stream responses or chunk large payloads to avoid memory spikes.
- Use async (`asyncio`, `httpx`) when concurrency benefits outweigh complexity.

## Security and Compliance

- Sanitize user inputs before executing commands or formatting prompts.
- Enforce capability filters on tools that access the system or network.

## Project blueprint

Project blueprint is in file ./ctmblueprint.md
