# HTTP API

FastAPI, mounted under `/api`. Interactive docs at `/docs`, OpenAPI schema at
`/openapi.json`. The schema is generated from the same Pydantic models the pipeline
uses internally, so it is always current — prefer it over this document when the
two disagree.

```bash
pip install -e ".[api]"
amla serve --host 127.0.0.1 --port 8000
# or:
uvicorn "automl_architect.api.app:create_app" --factory --port 8000
```

## Contents

- [Shape of the API](#shape-of-the-api)
- [Errors](#errors)
- [System](#system)
- [Data](#data)
- [Runs](#runs)
- [Events and streaming](#events-and-streaming)
- [Approvals](#approvals)
- [Questions](#questions)
- [Artifacts](#artifacts)
- [Client walkthrough](#client-walkthrough)
- [Notes for deployment](#notes-for-deployment)

---

## Shape of the API

Runs are long — minutes, not milliseconds — so `POST /api/runs` returns **202
Accepted** immediately with a run id and three follow URLs. Execution happens on a
background thread managed by `api/service.py::RunManager`. You then either poll
`GET /api/runs/{run_id}` or subscribe to the event stream.

There is no authentication layer. This is a deliberate omission, not an oversight:
the API is designed to sit behind whatever your deployment already uses. See
[Notes for deployment](#notes-for-deployment).

---

## Errors

Every non-2xx response uses one envelope:

```json
{
  "error": "IngestionError",
  "detail": "could not read /data/missing.csv: file not found",
  "run_id": "run_ab12cd34ef56"
}
```

The package's exception hierarchy maps onto status codes, so a client can branch on
the code rather than parsing prose. From `api/app.py::_STATUS_BY_ERROR`, first
match wins:

| Exception | Status | Meaning |
|---|---|---|
| `MissingDependencyError` | 503 | An optional extra is not installed in this deployment — a server capability gap, not a bad request |
| `ConfigurationError` | 503 | This deployment is misconfigured (missing credential, unusable database) |
| `ArtifactAccessError` | 404 | No such artifact, or a path outside the run directory |
| `ValidationError` | 422 | Loaded data failed structural validation |
| `IngestionError` (incl. `UnsupportedSourceError`) | 400 | The source could not be read |
| anything else | 500 | Logged with a traceback server-side |

Raised directly by the route handlers rather than through that table:

| Condition | Status |
|---|---|
| Malformed or empty request body, out-of-range field | 422 |
| Unknown run, approval, or artifact | 404 |
| A run is already active for that id | 409 |

Note that `MissingDependencyError` and `ConfigurationError` are **503**, not 4xx:
both mean the server cannot do the thing, so a client should surface them as a
deployment problem rather than telling the user their request was wrong.

---

## System

### `GET /`

Orientation for a browser that lands on the root. Returns the name, version, links
to `/docs` and `/api/health`, and the ids of currently active runs.

### `GET /api/health`

Answers "is this deployment actually able to run anything?" — not just "is the
process up". Use it as a readiness probe.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model": "claude-opus-5",
  "python_version": "3.11.9",
  "credentials_configured": true,
  "workspace": "/var/lib/automl/workspace",
  "workspace_writable": true,
  "database_url": "postgresql+psycopg://automl:***@postgres:5432/automl",
  "database_reachable": true,
  "active_runs": 1,
  "stored_runs": 42,
  "features": [
    {"name": "xgboost", "available": true},
    {"name": "shap", "available": true},
    {"name": "catboost", "available": false}
  ],
  "warnings": []
}
```

`status` is `"ok"` or `"degraded"`. Degraded means the process is serving but
something will fail — no credentials, an unwritable workspace, an unreachable
database — and `warnings` says which. The database URL is redacted.

### `GET /api/projects`

Project namespaces that have at least one stored run.

---

## Data

### `POST /api/upload`

`multipart/form-data` with a `file` field. Stores the dataset under the workspace
and returns the path plus a ready-made `DataSource`.

```json
{
  "filename": "churn.csv",
  "path": "/var/lib/automl/workspace/uploads/2026/churn-a1b2c3.csv",
  "size_bytes": 412903,
  "source": {"kind": "csv", "uri": "/var/lib/automl/workspace/uploads/2026/churn-a1b2c3.csv"}
}
```

Pass the returned `path` as `upload_path` to `POST /api/runs`. Two requests rather
than one is the right shape when the file is large and the operator wants to
confirm the inferred schema before spending anything.

---

## Runs

### `POST /api/runs` → 202

Accepts **either** JSON or `multipart/form-data`. The multipart form takes the same
field names plus a `file`, which is stored and turned into `upload_path` for you —
so a browser can start a run in one request.

Supply exactly one source: `source` (a full `DataSource` object), `uri` (a path or
URL), or `upload_path`.

```json
{
  "uri": "examples/churn.csv",
  "target_column": "churned",
  "project": "retention",
  "time_budget_seconds": 900,
  "max_experiments": 6,
  "max_rows": null,
  "test_size": 0.2,
  "validation_size": 0.15,
  "cv_folds": 5,
  "random_state": 42,
  "task_type": null,
  "primary_metric": null,
  "require_approval": false,
  "enable_tuning": true,
  "enable_explainability": true,
  "enable_self_improvement": true,
  "max_replans": 2,
  "min_acceptable_score": null,
  "report_formats": ["markdown", "html", "json"],
  "fairness_attributes": ["region"],
  "notes": "Quarterly retention review."
}
```

Only the source is required; everything else has the `RunConfig` default. Numeric
fields are range-validated in the request model — `cv_folds` must be 2–20,
`time_budget_seconds` 10–86,400, `max_experiments` 1–64 — so a bad value is a 422
rather than a run that misbehaves on a worker thread.

**Response**

```json
{
  "run_id": "run_ab12cd34ef56",
  "project": "retention",
  "status": "running",
  "stream_url": "/api/runs/run_ab12cd34ef56/stream",
  "events_url": "/api/runs/run_ab12cd34ef56/events",
  "run_url": "/api/runs/run_ab12cd34ef56"
}
```

### `GET /api/runs/{run_id}`

The complete `RunSummary`: config, ingestion result, profile, and every agent's
output, plus step records, approvals, usage totals, warnings, and the artifact
directory. This is the same object `analyse()` returns from the library and the
same object stored in the database, so a UI needs no second shape.

Poll `status` until it reaches a terminal value: `completed`, `failed`,
`cancelled`, or `awaiting_approval`.

### `GET /api/runs`

Query: `project` (optional filter), `limit` (1–500, default 50). Returns a
lightweight index — id, project, status, timestamps, task type, best score, best
family — rather than full summaries, so a run list stays cheap on a large database.

### `GET /api/runs/{run_id}/experiments`

The leaderboard on its own: one `ExperimentResult` per trained model with metrics,
CV scores, timings, and failure reasons.

### `GET /api/runs/{run_id}/similar`

Structurally similar past runs from dataset memory, with a similarity score and a
`why_similar` explanation.

### `POST /api/runs/{run_id}/cancel`

Requests cancellation. The orchestrator checks a flag between steps, so
cancellation is cooperative — it takes effect at the next boundary, not mid-fit.
404 for an unknown run.

---

## Events and streaming

Every meaningful action emits a `RunEvent` with a monotonic `sequence`. That log is
the substrate for the live UI, the report's audit trail, and the Q&A agent's
evidence base.

Event kinds: `run_started`, `run_completed`, `run_failed`, `run_cancelled`,
`step_started`, `step_completed`, `step_failed`, `step_skipped`, `step_retried`,
`agent_thinking`, `agent_decision`, `llm_call`, `artifact_written`,
`approval_requested`, `approval_resolved`, `replan_triggered`, `metric_recorded`,
`warning`, `log`.

### `GET /api/runs/{run_id}/events`

Query: `after` (sequence cursor, default 0), `limit`.

```json
{
  "run_id": "run_ab12cd34ef56",
  "events": [
    {
      "event_id": "ev_...",
      "run_id": "run_ab12cd34ef56",
      "sequence": 17,
      "kind": "agent_decision",
      "agent": "cleaning",
      "step_id": "clean_data",
      "message": "Cleaning Agent: 3 decisions, 2 columns dropped",
      "payload": [{"key": "dropped", "value": "[\"cancellation_tickets\", \"customer_id\"]"}],
      "at": "2026-07-28T12:34:56.789Z",
      "duration_seconds": 4.21,
      "tokens_in": 812,
      "tokens_out": 1450,
      "cache_read_tokens": 6120,
      "cost_usd": 0.0428
    }
  ],
  "next_after": 17
}
```

`payload` is a `list[Param]` rather than a free-form object — the same constraint
the agent schemas carry, kept here so an event round-trips through the same typed
path. Values are JSON-encoded strings; decode them client-side.

### `GET /api/runs/{run_id}/stream`

Server-sent events. Replays history first, then streams live, then closes when the
run reaches a terminal state.

```javascript
const source = new EventSource(`${API}/api/runs/${runId}/stream`);
source.onmessage = (message) => {
  const event = JSON.parse(message.data);
  render(event);
};
source.addEventListener("run_completed", () => source.close());
```

Each SSE frame carries the event kind as its `event:` name and the JSON `RunEvent`
as `data:`, with the sequence as `id:`. A reconnecting `EventSource` sends
`Last-Event-ID` automatically and resumes from that cursor — which is why the CORS
config exposes that header. Behind a proxy, disable response buffering for this
path or events arrive in batches.

---

## Approvals

Only relevant when a run was started with `require_approval: true`. The run
suspends before any step the plan marked `destructive`, its status becomes
`awaiting_approval`, and the summary is persisted — so the decision can arrive
minutes or days later.

### `GET /api/runs/{run_id}/approvals`

```json
{
  "run_id": "run_ab12cd34ef56",
  "approvals": [
    {
      "request_id": "apr_7f3a2b1c9d0e",
      "step_id": "clean_data",
      "agent": "cleaning",
      "action_summary": "Drop 2 columns: cancellation_tickets (leakage), customer_id (identifier)",
      "details": [
        "cancellation_tickets: AUC 0.9975 against the target; recorded post-outcome",
        "customer_id: 3,000 unique values over 3,000 rows"
      ],
      "affected_columns": ["cancellation_tickets", "customer_id"],
      "affected_row_estimate": 0,
      "severity": "critical",
      "decision": "pending",
      "created_at": "2026-07-28T12:34:56Z"
    }
  ]
}
```

### `POST /api/runs/{run_id}/approvals/{request_id}`

```json
{"decision": "approved", "note": "confirmed with the data owner", "decided_by": "elai"}
```

`decision` is `"approved"` or `"rejected"`. The response echoes the resolved
approval and a `resumed` flag saying whether a run thread was actually released by
it. A rejection skips the step or ends the run; either way the decision, the
decider, and the note are recorded permanently.

---

## Questions

### `POST /api/runs/{run_id}/ask`

```json
{"question": "Why was ROC AUC chosen over accuracy?"}
```

```json
{
  "question": "Why was ROC AUC chosen over accuracy?",
  "answer": "The target is 25.97% positive, so predicting the majority class alone scores 74% accuracy while having no predictive value — the dummy baseline in this run did exactly that. ROC AUC is threshold-free and unaffected by that base rate.",
  "evidence": [
    "profile: churned class distribution 0: 2,221 (74.03%), 1: 779 (25.97%)",
    "experiment baseline_dummy: accuracy 0.740, roc_auc 0.500",
    "problem.metric_rationale, recorded at 12:31:04Z"
  ],
  "confidence": "high",
  "caveats": [],
  "suggested_followups": ["Would average precision have been a better primary metric?"]
}
```

Answers are grounded in the run's recorded history. A counterfactual the run never
tested gets an honest "here is what the evidence suggests and what would have to be
re-run", not a fabricated number. This endpoint makes an LLM call, so it is not
free.

---

## Artifacts

### `GET /api/runs/{run_id}/report`

The rendered report. Query `format` to choose among what was produced
(`markdown`, `html`, `pdf`, `pptx`, `json`); defaults to the richest available.

### `GET /api/runs/{run_id}/artifacts`

Lists everything in the run directory with a relative path, a kind, and a size.

### `GET /api/runs/{run_id}/artifacts/{artifact_path}`

Serves one file. Query `download=true` for a `Content-Disposition` attachment.

Paths are resolved and confirmed to be **inside** that run's directory; traversal
attempts get a 4xx. `tests/test_api.py` covers several encodings of that attack,
because a free-form path parameter serving files from disk is worth pinning down.

---

## Client walkthrough

```python
import httpx

API = "http://localhost:8000"

with httpx.Client(base_url=API, timeout=30.0) as client:
    health = client.get("/api/health").json()
    assert health["credentials_configured"], health["warnings"]

    started = client.post(
        "/api/runs",
        json={
            "uri": "examples/churn.csv",
            "target_column": "churned",
            "max_experiments": 5,
            "fairness_attributes": ["region"],
        },
    )
    started.raise_for_status()
    run_id = started.json()["run_id"]

    # Follow the reasoning as it happens.
    with client.stream("GET", f"/api/runs/{run_id}/stream", timeout=None) as stream:
        for line in stream.iter_lines():
            if line.startswith("data:"):
                print(line[5:].strip()[:160])

    summary = client.get(f"/api/runs/{run_id}").json()
    print(summary["status"], summary["evaluation"]["overall_grade"])
    print(summary["evaluation"]["verdict_rationale"])

    answer = client.post(
        f"/api/runs/{run_id}/ask",
        json={"question": "Which decision most affected the score?"},
    ).json()
    print(answer["answer"])
```

---

## Notes for deployment

**No authentication is built in.** Put the API behind your existing gateway,
reverse proxy, or service mesh. Anything that can call it can start runs that cost
money and read datasets from paths the process can see.

**`uri` reads from the server's filesystem.** In a multi-tenant deployment,
restrict clients to `POST /api/upload` and `upload_path` rather than accepting
arbitrary `uri` values, or an untrusted caller can read any file the process can.

**Runs are threads in one process.** `RunManager` bounds concurrency; a run
already active for an id gives 409. Horizontal scaling needs a shared
`AUTOML_DATABASE_URL` (Postgres) *and* a shared workspace volume, because
artifacts are written to disk and served from there. Without the shared volume,
`/artifacts` will 404 on whichever replica did not run the job.

**SSE and proxies.** Disable response buffering on the `/stream` path, keep
idle timeouts above your longest run, and make sure `Last-Event-ID` survives the
proxy or reconnects replay from zero.

**Cost.** Each run makes 13+ frontier-model calls at high reasoning effort. Rate
limits belong at the gateway; `summary.usage` records exactly what each run cost so
you can set them from data rather than guesswork.
