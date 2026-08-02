# Cognitive Technique Mapper

Cognitive Technique Mapper (CTM) pairs real-world problem descriptions with the most suitable cognitive or problem-solving techniques using a configurable CLI and optional FastAPI surface.

## Installation

```bash
# Python 3.12+; install the reproducible full development environment.
uv sync --all-extras --frozen

# Configure provider credentials (example)
cp .env.example .env  # or export directly
export AZURE_API_BASE="https://<your-endpoint>.openai.azure.com/"
export AZURE_OPENAI_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Quick Start

```bash
# Seed or refresh local data
uv run --frozen python -m src.cli refresh --skip-embeddings

# Capture a problem description, analyze it, and generate justification
uv run --frozen python -m src.cli describe "I'm torn between two job offers."
uv run --frozen python -m src.cli analyze --show-candidates
uv run --frozen python -m src.cli explain

# Optional: run the FastAPI service
uv run --frozen uvicorn src.api:app --reload
```

## Features

- Configurable workflows (`describe`, `analyze`, `explain`, `simulate`, `compare`, `feedback`, `interactive-flow`) that blend SQLite data, vector search, and LLM reasoning through `litellm`.
- Shareable Markdown reports plus preference-aware recommendations backed by feedback loops and history inspection commands.
- YAML-driven configuration (`config/settings.yaml`, `config/models.yaml`, `config/providers.yaml`) for providers, models, and storage paths.
- Technique catalog lifecycle commands (`techniques list|add|update|import|export|refresh`) that keep SQLite and optional Chroma embeddings synchronized.
- Lightweight FastAPI server mirroring the CLI, enabling programmatic workflow invocation, health checks, and optional GraphQL access.

## Developer

For detailed setup, architectural notes, environment variables, and contribution workflows, see [README-DEV.md](README-DEV.md). Release history follows [Keep a Changelog](https://keepachangelog.com); consult `CHANGELOG.md` for the latest updates.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for full terms.
