# Deployment

How to run AI Data Modeler somewhere other than a laptop, and what breaks if you
skip a step.

Read [Before you expose this](#before-you-expose-this) first. The API has no
authentication by design, and the default `uri` handling reads files from the
server's filesystem.

## Contents

- [Before you expose this](#before-you-expose-this)
- [Deployment shapes](#deployment-shapes)
- [Docker Compose](#docker-compose)
- [Building the images](#building-the-images)
- [Configuration reference](#configuration-reference)
- [Database](#database)
- [Storage and the workspace](#storage-and-the-workspace)
- [Reverse proxy and SSE](#reverse-proxy-and-sse)
- [Scaling](#scaling)
- [Resource sizing](#resource-sizing)
- [Observability](#observability)
- [Cost control](#cost-control)
- [Kubernetes](#kubernetes)
- [Upgrades and backup](#upgrades-and-backup)
- [Troubleshooting](#troubleshooting)

---

## Before you expose this

Four properties of the system that matter more than anything else on this page.

**1. There is no authentication.** Not a missing feature to be added later — a
deliberate omission, because every organisation already has an answer. Anything
that can reach `POST /api/runs` can start runs that cost real money at a frontier
model's list price. Put an authenticating gateway, reverse proxy, or service mesh
in front of it before it is reachable from anywhere you do not control.

**2. `uri` reads the server's filesystem.** `{"uri": "/etc/passwd"}` is a valid
run request and the profiler will happily describe the file. In any deployment
where callers are not fully trusted, restrict them to `POST /api/upload` plus
`upload_path`, and reject `uri` at the gateway.

**3. Agent-authored plans execute against your data.** Nothing is `eval`'d — plans
are typed decisions dispatched to a fixed set of handlers, and column references
are filtered against the real schema — but the process does read, transform, and
write data. Run it as a non-root user (the image does), with a workspace volume
and nothing else writable.

**4. Datasets and reports persist.** The workspace holds the data that was
analysed, the fitted models, and reports quoting real values. Treat that volume
with whatever policy the source data carries.

---

## Deployment shapes

| Shape | Storage | Good for | Notes |
|---|---|---|---|
| **Laptop / CLI** | SQLite under `./workspace` | Exploration, one-off analyses | `pip install -e .` and go. No server needed |
| **Single container** | SQLite on a volume | A shared instance for a small team | Simplest thing that keeps run history |
| **Compose stack** | Postgres + volume | The intended default | API + Postgres + dashboard, one command |
| **Kubernetes** | Postgres + RWX volume | Multi-tenant, multi-replica | Needs a **shared** workspace volume, see [Scaling](#scaling) |
| **Batch / cron** | Postgres | Scheduled retraining reviews | Skip the API; call `analyse()` from a job |

---

## Docker Compose

The included `docker-compose.yml` brings up the API, Postgres, and the dashboard.

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY in .env
docker compose up --build -d

# dashboard  http://localhost:3000
# api docs   http://localhost:8000/docs
docker compose logs -f api
```

Verify the deployment is genuinely able to run something, not just serving:

```bash
curl -s localhost:8000/api/health | python -m json.tool
```

`status: "degraded"` means it is up but something will fail, and `warnings` says
what — no credentials, an unwritable workspace, an unreachable database. Then a
real end-to-end check:

```bash
curl -s -X POST localhost:8000/api/runs \
  -H 'content-type: application/json' \
  -d '{"uri":"/data/churn.csv","target_column":"churned","max_experiments":2,"enable_tuning":false}'
```

`./examples` is mounted read-only at `/data`, so that path works out of the box.

### Compose knobs

Every one of these is read from `.env` or the host environment:

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Passed through; never baked into the image |
| `API_PORT` / `FRONTEND_PORT` / `POSTGRES_PORT` | 8000 / 3000 / 5432 | Host port mappings |
| `API_EXTRAS` | `.[api,boost,tuning,explain,charts,reports,sql]` | Build arg. Trim it for a smaller image |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | **Build-time.** The origin the *browser* resolves — not the compose service name |
| `AUTOML_CORS_ORIGINS` | `["http://localhost:3000", ...]` | Must include every origin the dashboard is served from |
| `POSTGRES_USER` / `_PASSWORD` / `_DB` | `automl` | Change the password before this leaves your machine |
| `API_CPUS` / `API_MEMORY` | 4 / 6g | Container limits |
| `AUTOML_N_JOBS` | 4 | Keep at or below `API_CPUS` |

Two of those cause most of the confusion:

- **`NEXT_PUBLIC_API_URL` is inlined at build time.** Next.js bakes
  `NEXT_PUBLIC_*` into the client bundle. An image built against `localhost:8000`
  cannot be repointed by restarting it — rebuild the frontend image.
- **It cannot be `http://api:8000`.** That name resolves inside the Docker network,
  and the code using it runs in a browser on the host.

---

## Building the images

```bash
# API, default extras
docker build -t automl-architect:0.1.0 .

# Leaner: drop cloud drivers and static-image export
docker build --build-arg EXTRAS=".[api,boost,tuning]" -t automl-architect:lean .

# Dashboard, pointed at a real API origin
docker build --build-arg NEXT_PUBLIC_API_URL=https://automl.example.com \
             -t automl-frontend:0.1.0 frontend/
```

The API image is multi-stage: compilers, `gfortran`, `libpq-dev`, and pip caches
live in the build stage; the runtime stage gets a populated virtualenv plus
`libgomp1` (required by xgboost, lightgbm, and sklearn's OpenMP paths) and
`libpq5`. It runs as uid 1001, and `/var/lib/automl` is declared as a volume.

The healthcheck greps for `"status":"ok"` rather than trusting the status code,
because `/api/health` returns 200 with `"degraded"` when the process is serving but
unable to complete a run — which is exactly the state a healthcheck should catch.

---

## Configuration reference

Settings are read from the environment with an `AUTOML_` prefix, then `.env`, then
`.env.local`. `.env.example` documents every field; the ones that matter for a
deployment:

| Variable | Deployment guidance |
|---|---|
| `ANTHROPIC_API_KEY` | Inject as a secret. Never in an image or a compose file committed to git |
| `AUTOML_DATABASE_URL` | Postgres for anything shared. Empty = SQLite under the workspace |
| `AUTOML_WORKSPACE` | Must be a persistent volume, **shared** across replicas |
| `AUTOML_API_HOST` | `0.0.0.0` in a container; `127.0.0.1` is unreachable from outside it |
| `AUTOML_CORS_ORIGINS` | Exact origins, no wildcard. The browser blocks the fetch before the API sees it |
| `AUTOML_N_JOBS` | Set to the CPU limit. `-1` in a limited container oversubscribes and runs *slower* |
| `AUTOML_MAX_PROFILE_ROWS` | Lower it if profiling wide tables is the bottleneck. The true row count is still reported |
| `AUTOML_MAX_TRAIN_ROWS` | Caps training memory. Above this, training subsamples |
| `AUTOML_DEFAULT_EFFORT` | The main cost/quality dial. `high` is the default; `medium` cuts cost noticeably |
| `AUTOML_MAX_OUTPUT_TOKENS` | Covers thinking *plus* output on Opus 5. Lowering it too far truncates deep reasoning |
| `AUTOML_ENABLE_PROMPT_CACHING` | Leave on. With 13 agents per run this is most of the prompt cost |
| `AUTOML_LOG_JSON` | `true` for a log aggregator |

---

## Database

SQLite is the default and is fine for one process. It is not fine for two: the run
manager writes from worker threads, and concurrent writers on a network filesystem
will corrupt or lock.

Use Postgres for anything shared:

```bash
AUTOML_DATABASE_URL=postgresql+psycopg://automl:secret@postgres:5432/automl
```

Requires `pip install -e ".[sql]"` for the `psycopg` driver — the compose image
includes it by default.

Schema creation is automatic on first use; there is no migration step to run. Four
things are stored: run summaries (the full typed `RunSummary` as JSON), events (one
row each, for the audit trail and the Q&A evidence base), experiment results, and
dataset fingerprints (which is what makes dataset memory work across runs).

Event volume is the thing that grows. A run produces a few hundred events. If you
run thousands of analyses, prune old events on a schedule — the run summaries are
small and worth keeping regardless.

---

## Storage and the workspace

```
$AUTOML_WORKSPACE/
├── automl_architect.db          SQLite, if no AUTOML_DATABASE_URL is set
├── uploads/                     datasets received via POST /api/upload
└── runs/
    └── run_ab12cd34ef56/
        ├── run_summary.json     the complete typed record of the run
        ├── charts/              html, png, json per chart
        ├── models/              fitted estimators (joblib)
        └── report/              report.md, report.html, report.pdf, ...
```

`GET /api/runs/{id}/artifacts/{path}` serves files from `runs/<run_id>/`, with the
resolved path confirmed to be inside that directory.

Two consequences:

- **The volume must persist.** Losing it loses every report and model. Run
  summaries survive in the database, but the rendered artifacts do not.
- **The volume must be shared across replicas.** A replica that did not execute a
  run has no artifacts for it and returns 404. See [Scaling](#scaling).

---

## Reverse proxy and SSE

The event stream is server-sent events, which most default proxy configurations
break. nginx:

```nginx
location /api/ {
    proxy_pass         http://api:8000;
    proxy_http_version 1.1;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Real-IP         $remote_addr;
    proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;

    # SSE: buffering batches events into unusable chunks, and a short read
    # timeout kills the stream mid-run.
    proxy_buffering    off;
    proxy_cache        off;
    proxy_read_timeout 3600s;
    chunked_transfer_encoding on;
}
```

Checklist, because each of these has its own failure signature:

- `proxy_buffering off` on `/api/runs/*/stream`, or events arrive in bursts.
- Read timeout above your longest run, or the stream drops partway.
- `Last-Event-ID` must survive the proxy, or a reconnecting client replays from
  zero. The API already exposes it via CORS.
- Forward `X-Forwarded-Proto`; the container runs uvicorn with `--proxy-headers`.
- Body size limit above your largest upload if you allow `POST /api/upload`.

---

## Scaling

Runs execute as threads inside the API process. `RunManager` bounds concurrency and
returns 409 for a run id already active.

**Vertical first.** Runs are CPU-bound in scikit-learn and memory-bound in SHAP.
More cores and more RAM on one instance is simpler and usually cheaper than
horizontal scaling here.

**Horizontal, if you must.** Two hard requirements:

1. A shared `AUTOML_DATABASE_URL` (Postgres). Without it, replicas cannot see each
   other's runs.
2. A shared workspace volume, `ReadWriteMany`. Without it, `/artifacts` and
   `/report` 404 on whichever replica did not run the job, and the failure looks
   like data loss rather than a mount problem.

Neither the event stream nor the SSE reconnect needs sticky sessions — events are
persisted and replayed from the database by sequence cursor. But a run in progress
lives on one replica, so `POST /api/runs/{id}/cancel` and approval resolution only
release a thread on that replica. If you scale out with approvals enabled, route by
`run_id` or accept that resolution is recorded (persisted, visible in the summary)
without immediately unblocking the paused thread.

For genuinely high throughput, skip the API's threading model: run the pipeline as
a batch job from `analyse()`, one process per dataset, sharing only the database.

---

## Resource sizing

These are starting points, not benchmarks. Only the first row is grounded in
anything measured here — the offline test suite completes a full pipeline over
`examples/churn.csv` (3,000 × 16, three model families, no tuning) in 40–90
seconds of compute, and a real run adds the model-call latency on top. The other
two rows are extrapolations from where the cost sits, offered so you have a number
to start from. Measure your own and set limits from that.

| Dataset | Wall clock | Where the cost is |
|---|---|---|
| 3k rows × 16 cols | 2–5 min | Model-call latency dominates; compute is seconds |
| 100k × 50 | 5–15 min | Training and SHAP; SHAP is also the memory peak |
| 1M × 100 | 20 min+ | Profiling and training. Set `AUTOML_MAX_TRAIN_ROWS`, or sample first |

Guidance:

- **2 GB RAM minimum**, 6–8 GB comfortable. SHAP on a wide frame is the peak.
- **4+ cores.** Set `AUTOML_N_JOBS` to the CPU limit, not `-1`. In a limited
  container, `-1` sees the host's core count, oversubscribes, and runs slower.
- **Disk**: a few MB per run, more with PNG charts and PDF reports.
- **No GPU path.** Everything is CPU.

---

## Observability

**Logs.** Standard `logging` to stdout. `AUTOML_LOG_LEVEL` and
`AUTOML_LOG_JSON=true` for structured output.

**Health.** `GET /api/health` is a readiness probe that actually probes: workspace
writability, database reachability, credential presence, active run count, and
which optional features resolved.

**Per-run telemetry.** Every `RunSummary` carries `usage` — LLM calls, input,
output and cache-read tokens, and estimated USD — plus `steps` with per-step
duration, attempt count, and error. That is the data to build dashboards from;
scrape it from the database rather than parsing logs.

**Events.** The event log is the audit trail. `GET /api/runs/{id}/events` for a
finished run, `/stream` for a live one.

**MLflow.** Set `AUTOML_MLFLOW_TRACKING_URI` and install `mlflow` to log
experiments. Optional and off by default; absent, the feature degrades with a
warning.

Metrics worth alerting on:

- `status == "failed"` rate.
- `usage.cost_usd` per run, and its trend.
- `replans > 0` rate — high means the Evaluation Agent is rejecting a lot, which is
  a signal about the data, not a bug.
- Wall clock against `time_budget_seconds`.
- `cache_read_tokens == 0` on a multi-agent run — that means prompt caching stopped
  working, and the bill went up without anything erroring.

---

## Cost control

A run makes 13+ calls to a frontier model at high reasoning effort. Levers, in
descending order of effect:

1. **`amla profile` is free.** No model calls at all. Read the facts before
   spending anything.
2. **Keep prompt caching on.** The frozen run context is read from cache by every
   agent after the first, at a tenth of the input price.
3. **Lower `AUTOML_DEFAULT_EFFORT`.** `medium` cuts thinking tokens noticeably.
   Agents that override upward (planner, evaluation) still do.
4. **Cap `max_experiments`.** Fewer models means less for the Experiment,
   Evaluation, and Report agents to reason over.
5. **`enable_tuning=false`.** The Tuning Agent still decides, but skipping the step
   removes a call and the search.
6. **`max_replans=0`.** Each replan re-runs the planner and everything after it.
7. **Rate-limit at the gateway.** Set the limits from `summary.usage` data rather
   than guessing.

`AUTOML_ENABLE_REFUSAL_FALLBACK=true` (default) re-serves a policy decline through
a fallback model instead of failing the run. The substitution is recorded as a
warning on the summary.

---

## Kubernetes

No manifests are shipped, deliberately — the shape depends too much on your
cluster. What a working deployment needs:

- **Deployment** for the API. `AUTOML_API_HOST=0.0.0.0`, the credential from a
  `Secret`, resource requests and limits, and `AUTOML_N_JOBS` matching the CPU
  limit.
- **PersistentVolumeClaim** for the workspace, **`ReadWriteMany`** if more than one
  replica. This is the requirement people miss.
- **Postgres**, managed or in-cluster, with the URL in a `Secret`.
- **Readiness probe** `GET /api/health`, `initialDelaySeconds: 20`. **Liveness** on
  the same path with a longer period.
- **`terminationGracePeriodSeconds`** longer than a typical run, or a rolling
  update kills in-flight work. Runs are not resumable mid-step; a killed run is a
  failed run.
- **Ingress** with SSE buffering disabled and a long read timeout on
  `/api/runs/*/stream`.
- **Separate Deployment** for the frontend, built with the public API origin.

A `HorizontalPodAutoscaler` on CPU works, but read [Scaling](#scaling) first — the
shared-volume requirement is not optional.

---

## Upgrades and backup

**Upgrading.** Pull the image, restart. Schema changes are applied automatically on
first use. Stored `RunSummary` objects are versioned by the schema that wrote them;
a field removed from `core/schemas.py` makes older stored runs fail to deserialise,
so the repository returns `None` for an undecodable row rather than raising —
history degrades, the API keeps serving.

**Backup.**

- The database holds run summaries, events, experiments, and fingerprints. Standard
  `pg_dump`.
- The workspace volume holds reports, charts, models, and uploads. Snapshot it.
- Neither is derivable from the other, and neither is derivable from the source
  data — the reasoning in a run is not reproducible byte-for-byte.

---

## Troubleshooting

**Health says `degraded`.** Read `warnings`. Almost always missing credentials, an
unwritable workspace, or an unreachable database.

**Runs fail immediately with a credentials error.** `docker compose exec api amla
doctor` reports which credential the SDK resolved, if any. Note the container needs
the variable in *its* environment — check the `environment:` block, not just your
shell.

**Dashboard shows nothing / CORS errors in the console.** The dashboard's origin is
not in `AUTOML_CORS_ORIGINS`, or `NEXT_PUBLIC_API_URL` was baked at build time
pointing somewhere else. Both need fixing; the second needs a rebuild.

**Events arrive in bursts, or the stream dies partway.** Proxy buffering, or a read
timeout shorter than the run. See [Reverse proxy and SSE](#reverse-proxy-and-sse).

**`/artifacts` returns 404 for a run that clearly finished.** Two replicas without a
shared workspace volume. The run is in the database; the files are on the other
pod's disk.

**Slower than expected on a big box.** `AUTOML_N_JOBS=-1` in a CPU-limited
container. joblib sees the host's cores, oversubscribes, and thrashes. Set it to
the limit.

**OOM during explainability.** SHAP on a wide frame. Lower
`AUTOML_MAX_TRAIN_ROWS`, raise the memory limit, or set
`enable_explainability=false` for that run — the pipeline degrades to permutation
importance rather than failing.

**`MissingDependencyError` for a source or a model family.** An optional extra is
not installed in the image. Rebuild with a wider `EXTRAS` build arg; the error
message names the package and the install command.

**Postgres connection refused on a cold start.** Compose gates the API on the
Postgres healthcheck, so this should not happen with the shipped file. Outside
compose, the API's startup ping logs the failure and keeps serving — the first run
is what fails. Check the URL and that the database accepts connections.
