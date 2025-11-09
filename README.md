# Cognitive Technique Mapper

Cognitive Technique Mapper (CTM) is a modular CLI application that pairs problem descriptions with the most suitable cognitive or problem-solving techniques. It combines a structured SQLite knowledge base, optional Chroma vector search, and workflow-specific LLM reasoning orchestrated through `litellm`.

---

## 🚀 Features

- **Technique Knowledge Base:** SQLite schema for storing named techniques with descriptions, origins, categories, and core principles. Optional Chroma collection holds vector embeddings for semantic search.
- **Workflow-Oriented Reasoning:** `litellm` gateway routes requests to different models per workflow (detection, explanation, summarization, feedback).
- **Structured Recommendations:** `analyze` now parses LLM replies into technique, rationale, and concrete steps, and automatically generates an implementation plan.
- **Structured Explanations:** `explain` renders JSON-backed insights covering key factors, risks, and suggested follow-ups.
- **CLI Experience:** `describe`, `analyze`, `explain`, `settings`, `refresh`, and `feedback` commands guide users from problem input to actionable recommendations.
- **Config-Driven:** YAML files under `config/` define app metadata, database paths, model mappings, and providers.
- **Bootstrap & Persistence:** `TechniqueDataInitializer` seeds the database (and Chroma) from `data/techniques.json`, while CLI state persistence (`data/state.json`) allows multi-command sessions.
- **Interactive Configuration:** `settings show`, `settings update-workflow`, and `settings update-provider` offer in-CLI editing with optional interactive prompts.
- **Dataset Refresh:** `refresh` rebuilds the SQLite dataset and (optionally) regenerates vector embeddings without manual file management.
- **Persistent Feedback:** Feedback entries are stored in SQLite, so summaries span sessions and retain historical context.

---

## 🗂️ Project Structure (excerpt)

```
.
├── config/                # settings.yaml, models.yaml, providers.yaml
├── data/                  # techniques.json, state.json, SQLite DB
├── embeddings/            # Chroma DB persistence
├── prompts/               # Prompt templates and registry
├── src/
│   ├── cli.py             # Typer CLI entry point
│   ├── core/              # Orchestrator, LLM gateway, config loader, etc.
│   ├── db/                # SQLite and Chroma clients
│   ├── services/          # Technique selector, embeddings, plan generator…
│   └── workflows/         # Workflow definitions wired into the orchestrator
└── tests/                 # Pytest coverage for config, DB, workflows, prompts
```

---

## 📦 Prerequisites

- Python 3.11+ (recommended 3.12)
- Virtual environment (`python -m venv .venv` then activate)
- Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv pip sync requirements.lock
```

- The lock file already captures both runtime and development dependencies (e.g., pytest).

- Set provider credentials (example `.env` entry):

```
AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
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
python -m src.cli settings show
python -m src.cli settings update-workflow detect_technique --model openai/gpt-4.1 --temperature 0.4
python -m src.cli refresh --skip-embeddings
python -m src.cli analyze --show-candidates
python -m src.cli feedback "Loved the recommendation" --rating 5
python -m src.cli analyze --log-level DEBUG  # temporary verbose logging
```

Notes:
- `describe` saves the problem description and persists it to `data/state.json`.
- `analyze` runs the `detect_technique` workflow (LLM + vector search).
- `explain` requests the `explain_logic` workflow to justify the recommendation.
- `settings show` prints a JSON snapshot of the current config values. Use `update-workflow` and `update-provider` to make inline edits (supports `--interactive`).
- `feedback` stores feedback and summarizes recent entries via LLM.
- `refresh` reloads `data/techniques.json`, replaces existing rows, and optionally rebuilds embeddings.
- `analyze --show-candidates` prints the shortlist with similarity scores for transparency.

If the LLM provider rejects a parameter (e.g., unsupported temperature), adjust `config/models.yaml` or set `litellm.drop_params = True` before running the CLI.

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

---

## 📄 License

Project licensing has not been specified. Add a `LICENSE` file if you plan to open-source the repository.

---

Happy mapping! 🎯✨
