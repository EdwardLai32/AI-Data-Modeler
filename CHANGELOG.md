# Changelog

User-visible changes are recorded here. Unreleased changes are developed on feature branches and reviewed before merging into `main`.

## Unreleased

### Added

- Anthropic and OpenAI provider adapters with ephemeral, tested API-key connections.
- Provider-specific model dropdowns that preserve configured server defaults, including custom model IDs.
- Per-run call/token budgets, optional estimated spending limits, and supervised training cancellation/deadlines.
- Authenticated local checkpoints with code/environment compatibility checks and explicit CLI/dashboard recovery.
- Fold-local learned preprocessing, separate final test scoring, and exported fitted prediction pipelines.
- Repeatable offline/provider evaluation commands and GitHub Actions checks for backend tests, offline evaluations, TypeScript, and frontend builds.
- Harness documentation, an end-to-end walkthrough, and a documented Git workflow.

### Changed

- Dashboard usage, scoring, and event-stream recovery reflect harness limits and execution state.
- API access uses an owner token for remote requests, exact browser-origin checks, and opt-in remote sources.
- Settings load provider credentials explicitly and redact them from persisted output.
- Upload handling enforces size boundaries while preserving file bytes.
- Claude connection probes keep adaptive thinking enabled for models that require it.

### Fixed

- Provider-specific defaults no longer mix Claude and OpenAI model names.
- Source routes under `frontend/src/app/runs/` are tracked while root-level run artifacts remain ignored.
- Dotenv precedence and relocated Windows editable-install instructions are documented and covered by tests where applicable.

These changes do not introduce a published package release. Package versions remain unchanged until a release is prepared.
