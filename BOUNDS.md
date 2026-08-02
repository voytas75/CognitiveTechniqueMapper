# Project Bounds

## Source of truth
This file and `PLAN.md` govern the Cognitive Technique Mapper scope. Conflicts require explicit user approval.

## Scope controls
- Implement only the first-release capabilities accepted in `PLAN.md`.
- Preserve the primary outcome: one justified recommendation and exactly three candidates for a supplied problem, task, or statement.
- Later ideas, new integrations, and broad refactors require an explicit scope change.
- Do not add speculative abstractions or product features while repairing quality or architecture.

## Technical constraints
- Allowed stack: Python 3.12+, `uv`, Typer, SQLite, optional local Chroma, and LiteLLM.
- HTTP/GraphQL stays bound to loopback as a development utility. Any public deployment requires a dedicated authentication, authorization, CORS, threat-model, and deployment slice.
- New dependencies are approval-only.
- Configuration secrets remain in environment variables or untracked local configuration; never commit credentials.

## Change budget
- Ordinary slice: at most 3 files, 120 net changed lines, and 1 new file.
- Ordinary slice: no new dependency.
- Before exceeding the budget: explain the impact and obtain explicit approval.

## Agent operating rules
1. Work only after an explicit user command, such as `proceduj`.
2. Inspect `PLAN.md` and `BOUNDS.md` before a material change.
3. Keep every slice reversible, tested where applicable, and committed locally as a coherent unit.
4. Report changed files, verification performed, and deferred suggestions at each checkpoint.
5. Do not stage or modify unrelated local user changes.

## Prohibited without explicit approval
- Multi-user features or multi-user hosting.
- Public API deployment or non-loopback HTTP binding.
- A new dependency.
- A scope expansion, broad folder restructure, or refactor beyond the accepted slice.
- Changing security settings, access controls, or credential handling.

## Checkpoints
- Before a slice exceeds its change budget.
- Before public API/deployment work, multi-user work, a new dependency, or a new external integration.
- Before pushing accumulated local commits or including a user-owned local configuration change.
