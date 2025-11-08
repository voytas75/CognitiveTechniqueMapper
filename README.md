# Cognitive Technique Mapper

Cognitive Technique Mapper (CTM) is a modular CLI application that pairs problem descriptions with the most suitable cognitive or problem-solving techniques. It combines a structured SQLite knowledge base, optional Chroma vector search, and workflow-specific LLM reasoning orchestrated through `litellm`.

---

## 🚀 Features

- **Technique Knowledge Base:** SQLite schema for storing named techniques with descriptions, origins, categories, and core principles. Optional Chroma collection holds vector embeddings for semantic search.
- **Workflow-Oriented Reasoning:** `litellm` gateway routes requests to different models per workflow (detection, explanation, summarization, feedback).
- **CLI Experience:** `describe`, `analyze`, `explain`, `settings`, and `feedback` commands guide users from problem input to actionable recommendations.
- **Config-Driven:** YAML files under `config/` define app metadata, database paths, model mappings, and providers.
- **Bootstrap & Persistence:** `TechniqueDataInitializer` seeds the database (and Chroma) from `data/techniques.json`, while CLI state persistence (`data/state.json`) allows multi-command sessions.

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
- Install dependencies:

```bash
pip install -r requirements.txt
```

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
python -m src.cli settings
python -m src.cli feedback "Loved the recommendation" --rating 5
```

Notes:
- `describe` saves the problem description and persists it to `data/state.json`.
- `analyze` runs the `detect_technique` workflow (LLM + vector search).
- `explain` requests the `explain_logic` workflow to justify the recommendation.
- `settings` prints a JSON snapshot of the current config values.
- `feedback` stores feedback and summarizes recent entries via LLM.

If the LLM provider rejects a parameter (e.g., unsupported temperature), adjust `config/models.yaml` or set `litellm.drop_params = True` before running the CLI.

---

## 🧠 Knowledge Base & Embeddings

1. Seed techniques in `data/techniques.json`.
2. On startup, `TechniqueDataInitializer` populates SQLite (if empty) and attempts to embed each technique using the configured embedding model. When Chroma is available, embeddings are stored and used for semantic retrieval.
3. If embedding APIs are unreachable, a deterministic hash-based fallback keeps the system functional (with reduced semantic quality).

To extend the library:
- Add rows to `data/techniques.json`.
- Re-run the CLI (it automatically re-seeds if the DB is empty). For full refresh scenarios, delete `data/techniques.db` and the Chroma directory, then restart the CLI.

---

## 🧪 Testing

Run the test suite with:

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
  - A refresh CLI command to re-embed techniques.
  - Additional workflows (e.g., scenario simulation).
  - Extended prompts and richer feedback analysis.

---

## 📄 License

Project licensing has not been specified. Add a `LICENSE` file if you plan to open-source the repository.

---

Happy mapping! 🎯✨
