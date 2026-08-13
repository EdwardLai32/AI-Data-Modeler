# AI Data Modeler — dashboard

The web front end for the multi-agent AI data scientist. Next.js (App Router) +
TypeScript (strict) + Tailwind CSS v4, talking to the FastAPI backend over REST
and Server-Sent Events.

Its job is to make the *reasoning* visible. A black-box AutoML tool shows you a
leaderboard; this shows you the plan an agent wrote, the rationale attached to
every cleaning and feature decision, the baseline the winner had to beat, the
evaluation verdict that gated it, and the token cost of getting there — while the
run is still happening.

---

## Install, dev, build

Requires Node 20+ (developed on Node 24) and npm.

```bash
cd frontend
npm install
npm run dev     # http://localhost:3000
```

```bash
npm run build   # production build; also type-checks
npm run start   # serve the production build
npm run typecheck
```

The backend must be running for anything to load:

```bash
amla serve      # from the repository root — starts the API on 127.0.0.1:8000
```

If it is not, every page says so explicitly and repeats that command rather than
rendering an empty dashboard.

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Origin of the FastAPI backend. |

Copy `.env.local.example` to `.env.local` to override it:

```bash
cp .env.local.example .env.local
```

Two constraints on that value:

- It is read **in the browser**, so it must be reachable from the browser — not
  a container-internal hostname.
- It is inlined **at build time** (`NEXT_PUBLIC_*`), so changing it means
  rebuilding, not just restarting.
- The origin serving this app must appear in the API's CORS allow-list
  (`AUTOML_CORS_ORIGINS`, default `http://localhost:3000`).

---

## Pages

| Route | What it is |
| --- | --- |
| `/` | Run history with status, best score, task type and duration, plus the **New analysis** launcher (drag-and-drop upload, target column, run options). |
| `/runs/[runId]` | The live run view — pipeline stepper, streaming event feed, every agent's output as it lands, token/cost meter, approval modal, and "Ask about this run". |
| `/runs/[runId]/report` | The finished report, with a format switcher (rendered / markdown / standalone HTML / JSON, plus PDF and PPTX download links when those files exist). |

### The live run view

- **Pipeline stepper** — merges the Planning agent's steps (title, objective,
  per-step rationale, destructive flag) with the orchestrator's step records
  (status, attempts, elapsed). Records with no matching plan step are shown too:
  ingestion and profiling happen before a plan exists.
- **Event feed** — `EventSource` on `GET /api/runs/{id}/stream`. Agent thinking
  and decisions render as attributed prose; logs stay a dense mono line. Follows
  the tail, pauses the moment you scroll up, and offers "Jump to latest (N new)".
- **Panels** — dataset profile, dataset understanding, problem definition,
  execution plan, cleaning decisions, engineered features, model selection,
  leaderboard, tuning, feature importance, evaluation verdict, business
  insights, charts. Each fills in when its object appears on the run summary and
  shows an explicit "not yet" state until then.
- **Tokens & cost** — `RunSummary.usage`, with the time budget and prompt-cache
  hit rate as the two bounded ratios worth metering.
- **Approval modal** — appears when a run suspends on a destructive step, states
  exactly which columns and how many rows are affected, and posts the decision
  back to resume the run. Escape dismisses the dialog but not the decision: the
  run stays suspended and a banner offers it back.

---

## Architecture

```
src/
  app/                     route shells only — no markup, no classes (see the note below)
    layout.tsx             chrome, theme bootstrap, skip link
    page.tsx               dashboard
    runs/[runId]/          → components/run/RunView
    runs/[runId]/report/   → components/report/ReportPageView
  components/
    dashboard/             run list, new-analysis form
    run/                   RunView and every live-run panel
    report/                rendered report + markdown assembly
    ui/                    Panel, Badge, Button, Field, Metrics, States, Rationale
  hooks/
    useRunStream.ts        EventSource lifecycle, cursor, reconnect, poll fallback
    useRunSummary.ts       the run blackboard, refreshed on interesting events
    useRuns.ts             dashboard list, polls only while a run is active
    useHealth.ts           reachability probe with retry
    useNow.ts              ticking clock for elapsed readouts
  lib/
    api.ts                 typed client, ApiError, artifact URLs
    format.ts              numbers, durations, timestamps — "—" when absent
    labels.ts              enum → label, enum → status tone
    metrics.ts             leaderboard ordering and bar scaling
    runCard.ts             normalises a run-list row
  types/
    api.ts                 TypeScript mirror of core/schemas.py
    plotly.d.ts            ambient types for plotly.js-dist-min
```

### Types mirror the Pydantic models

`src/types/api.ts` mirrors `automl_architect/core/schemas.py` (and the HTTP-only
models in `automl_architect/api/schemas.py`) field-for-field, in snake_case, so a
JSON body can be assigned straight to the interface with no renaming layer. `X | None` becomes `X | null`, `datetime` becomes `string`
(ISO-8601, as `model_dump(mode="json")` emits), and `list[Param]` stays
`Param[]`. Enums are string union types, not TS `enum`s, because the wire format
*is* the string.

When a schema field changes, change it here in the same commit.

### Never invent a number

Every value on screen comes from the API. Absent values render `—`, not `0`. The
leaderboard's bars are scaled to the range of scores actually on the board (a
0-1 axis makes every ROC AUC look identical) and each row carries its literal
score, so bar length is never the only encoding. Chart figures are rendered from
the backend's plotly JSON with only the *chrome* re-themed — the traces, values,
and colours are the pipeline's.

### Design tokens

Light and dark are two selected themes, not an inversion: `globals.css` defines
its own steps for each. Status hues (good / warning / serious / critical) are
reserved and always paired with a text label, so no state is signalled by colour
alone. The theme choice (light / dark / system) persists in `localStorage` and is
applied by a blocking inline script before first paint.

### ⚠️ Route files must stay class-free

Tailwind's source scan does not reach `src/app/runs/[runId]/**` — a dynamic
route's bracketed directory name is read as a glob character class, and an
explicit `@source` glob does not rescue it. A utility class written inside a
route folder therefore produces **no CSS, silently**. Route files are thin
shells that render a component from `src/components`, which is scanned. Keep it
that way.

---

## HTTP contract

Everything the dashboard calls. Paths are relative to `NEXT_PUBLIC_API_URL`. The
authority is `automl_architect/api/routes.py` and `api/schemas.py`; the
request/response shapes are mirrored in `src/types/api.ts`.

| Method | Path | Used for |
| --- | --- | --- |
| `GET` | `/api/health` | Reachability probe → `HealthResponse`. `version` and `workspace` are displayed when present. |
| `GET` | `/api/runs?project=&limit=` | Run list → `{project, count, runs: RunListItem[]}`. A bare array is also accepted. |
| `POST` | `/api/upload?filename=` | Raw dataset bytes (`application/octet-stream`) → `UploadResponse`. Used instead of a multipart post because it works whether or not the backend has `python-multipart`. |
| `POST` | `/api/runs` | Start a run. JSON `StartRunRequest`: one of `source` (a nested `DataSource`), `uri`, or `upload_path`, plus the run knobs. → `202` + `StartRunResponse`. |
| `GET` | `/api/runs/{id}` | The full `RunSummary`. |
| `POST` | `/api/runs/{id}/cancel` | Cancel an in-flight run → `CancelResponse`. |
| `GET` | `/api/runs/{id}/events?after=N` | Replay events after sequence `N` → `EventPage`. A bare array is also accepted. |
| `GET` | `/api/runs/{id}/stream?last_sequence=N` | SSE stream of `RunEvent` JSON. The `event:` name is the event kind; `message` is handled too. `:` comment keepalives are ignored. |
| `POST` | `/api/runs/{id}/ask` | `{question}` → `QuestionAnswer`. |
| `POST` | `/api/runs/{id}/approvals/{request_id}` | `{decision: "approved"｜"rejected", note?, decided_by?}` → `ApprovalDecisionResponse`; the run resumes. |
| `GET` | `/api/runs/{id}/artifacts` | Index of files written by the run → `ArtifactListResponse`. A 404 is tolerated. |
| `GET` | `/api/runs/{id}/artifacts/{path}` | Raw bytes of one artifact. `path` is relative to the run directory — the portion of the executor's absolute path after the run id, e.g. `charts/roc_curve.json`. Required for embedded charts and the markdown/HTML report views. |

Contract details that are easy to get wrong:

- **`StartRunRequest` forbids unknown fields.** Sending `source_uri`,
  `source_kind`, `primary_metric_override`, or `task_type_override` is a 422, not
  a silently ignored field. The HTTP names are `uri`, `primary_metric`, and
  `task_type`; a source kind can only be forced through the nested `source`
  object, which is why run creation posts JSON rather than a form.
- **`RunListItem` calls the target `target` and the grade `grade`**, not
  `target_column` / `overall_grade` (those are the *`RunSummary`* spellings).
- **`ArtifactEntry.url` is already server-relative** (`/api/runs/…`). Prefix it
  with `API_BASE`; do not put it through `artifactUrl`, which expects a
  filesystem path.
- **Errors.** FastAPI's `{"detail": ...}` is understood as a string or as a
  validation-error array, as is the `{error, detail, run_id}` envelope from
  `ErrorResponse`. A transport failure and an HTTP error are surfaced
  differently in the UI, so the status code matters.
- **Sequence numbers.** `RunEvent.sequence` is the resume cursor: monotonic per
  run, assigned before the event is published.
- **CORS.** Needed on every route above, including the artifact route and the SSE
  stream. The backend's allow-list is `AUTOML_CORS_ORIGINS` and defaults to
  `http://localhost:3000` only — open the dashboard on `localhost`, not
  `127.0.0.1`, or add that origin.

## Accessibility

Real `<label for>` on every control (never a placeholder as a label), hints wired
with `aria-describedby`, `role="log"` + `aria-live="polite"` on the event feed,
`role="progressbar"` with `aria-valuetext` on meters, a focus-trapped and
Escape-dismissible approval dialog, a skip link, visible focus rings everywhere,
and `aria-pressed` on the filter and format toggles. Layout is responsive down to
tablet width, where the two-column run view stacks with the live column first.
