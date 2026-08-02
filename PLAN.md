# Project Plan

## Goal
Cognitive Technique Mapper is a local CLI for the primary user and, indirectly, AI agents. It accepts a problem, task, or statement and returns one justified cognitive-technique recommendation together with three candidate alternatives and paths to approach the problem.

## Problem and users
- **Problem:** People and AI agents need a structured way to consider alternative cognitive or problem-solving approaches for a stated task or problem.
- **Primary users:** The repository owner first; AI agents may consume the CLI indirectly.

## First release — must have
- [ ] `describe`, `analyze`, and `explain` flows produce one justified recommendation plus three candidates.
- [ ] Technique catalog lifecycle commands maintain the local technique data.
- [ ] Feedback and history flows persist and expose user feedback and prior decisions.
- [ ] `compare` and `simulate` flows remain available for evaluating alternatives.
- [ ] Interactive flow remains available for guided input.
- [ ] Markdown report export remains available.

## Later / explicitly deferred
- Public or production API deployment. The current HTTP surface remains a local loopback development utility until a dedicated security and deployment project is approved.

## Anti-scope
- Multi-user operation and multi-user hosting are not part of the first stable release.

## Definition of done
- The local CLI accepts a problem and returns one justified recommendation plus three candidates.
- Local quality gates and CI are green.

## Technology decisions
- **Runtime/language:** Python 3.12+ with `uv` and Typer CLI.
- **Data and retrieval:** SQLite with optional local Chroma embeddings.
- **LLM integration:** LiteLLM with project-managed provider/model configuration.
- **HTTP surface:** Optional FastAPI/GraphQL loopback-only development utility; no public bind without a separate approved security/deployment slice.
- **Dependencies:** Approval-only for every new dependency.

## Open decisions
- No open first-release scope decisions. A future public API requires a new security and deployment decision record.
