# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Align `max_tokens` handling with LiteLLM per-model limits to avoid provider rejections.
- Fix `settings` CLI command by serializing workflow configs via dataclass helpers.
- Extend `settings` CLI output with embedding configuration metadata.
- Adopt `pyproject.toml` with uv-generated `requirements.lock` for dependency management.
