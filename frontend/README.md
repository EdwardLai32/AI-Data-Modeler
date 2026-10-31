# AI Data Modeler dashboard

The Next.js 16 / React 19 interface for the Python AutoML backend. It shows the analysis plan, live events, decisions, model comparisons, evaluation, explanations, reports, and recorded token usage.

See the [root README](../README.md) for backend installation and the [project walkthrough](../PROJECT_WALKTHROUGH.md) for the end-to-end explanation.

## Run locally

Requires Node.js 20.9 or newer and npm. The frontend Docker image uses Node 22. Use the committed lockfile for repeatable installs. The minimum follows the [Next.js requirements](https://nextjs.org/docs/app/getting-started/installation).

First start the backend from the repository root, in PowerShell:

```powershell
.\.venv\Scripts\amla.exe serve --offline
```

In a second terminal from this `frontend/` directory:

```powershell
npm ci
npm run dev
```

Open **http://localhost:3000**. Upload `../examples/churn.csv`, enter `churned` as the target, and start a run. A small experiment count with tuning disabled is useful for the first demonstration.

For online reasoning, use **Connect your AI** to choose Claude/OpenAI, select a model from the provider's dropdown, and enter an API key. The list includes the server's configured default. **Connect and test** makes a small billed request to verify your account can use the selected model. The key is cleared from the form on success and held in server memory, never browser storage or saved runs. Reconnect after reloading or restarting the server. Alternatively, configure server credentials in the root `.env`.

**AI usage limits** controls request, token, and estimated dollar limits. Run details show the selected provider/model, checkpoint, validation score, and separate final test metrics. **Resume checkpoint** continues compatible interrupted runs. See [HARNESS.md](../docs/HARNESS.md).

If the backend requires `AUTOML_API_TOKEN`, use **Server sign-in** inside the connection panel. For the HttpOnly session cookie to work, frontend and API must use the same site (use `localhost` for both locally); configure exact CORS origins.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend origin reachable from the user's browser |

Create a local override only if one does not exist:

```powershell
if (-not (Test-Path .env.local)) { Copy-Item .env.local.example .env.local }
```

This value is embedded in the browser bundle at build time. Rebuild production output after changing it. A Docker-internal hostname will not work unless the browser can resolve and reach it.

The backend CORS allow-list defaults to `http://localhost:3000`. For another browser origin, edit `AUTOML_CORS_ORIGINS` in the root `.env` and restart the backend. For example:

```dotenv
AUTOML_CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]
```

## Pages and code map

| Route | Component | Purpose |
| --- | --- | --- |
| `/` | `dashboard/RunList`, `dashboard/NewAnalysisPanel` | Browse runs and submit datasets |
| `/runs/[runId]` | `run/RunView` | Inspect the live pipeline and recorded results |
| `/runs/[runId]/report` | `report/ReportPageView` | Read the analysis report |

Components live under `src/components/`. Route files in `src/app/` remain thin wrappers, with presentation and utility classes kept in components.

**Keep route files visible to Git.** The root `.gitignore` must use `/runs/`, not an unanchored `runs/` pattern. The latter also ignores `src/app/runs/`, which can make the dashboard work locally while losing both pages in a fresh clone.

| File or directory | Responsibility |
| --- | --- |
| `src/lib/api.ts` | Typed requests, uploads, errors, artifact URLs |
| `src/types/api.ts` | TypeScript mirrors of backend schemas |
| `src/hooks/useRunStream.ts` | Event replay, SSE, reconnection, polling fallback |
| `src/hooks/useRunSummary.ts` | Refresh the run snapshot |
| `src/hooks/useRuns.ts`, `useHealth.ts` | Dashboard list and backend availability |
| `src/components/run/` | Pipeline panels, charts, Q&A, approvals, event feed |
| `src/components/report/` | Report views and Markdown assembly |
| `src/app/globals.css` | Themes and global styles |

## How data reaches the screen

1. The browser sends dataset bytes to `POST /api/upload?filename=...` and receives a stored path. It sends that as `upload_path` to `POST /api/runs`.
2. The backend returns HTTP 202 with a run ID. The dashboard loads the summary and replays events that already occurred.
3. SSE streams subsequent `RunEvent` objects. Sequence numbers support replay and deduplication after reconnecting; REST polling is a fallback.
4. Relevant events trigger summary refreshes. Panels fill from the corresponding summary objects and expose unavailable or pending results.
5. Reports and charts are fetched from the run's artifact endpoints.

`src/types/api.ts` uses the backend's snake_case names. Update those types alongside changes to `core/schemas.py` or `api/schemas.py`. HTTP uses `primary_metric`, for example, whereas Python's `RunConfig` uses `primary_metric_override`.

Key endpoints include `/api/health`, `/api/runs`, `/api/runs/{id}/events`, `/api/runs/{id}/stream`, `/api/runs/{id}/ask`, approvals, and artifacts. Refer to [docs/API.md](../docs/API.md) and the backend's `/docs` for the full contract.

The backend upload limit defaults to 256 MiB, configured with `AUTOML_MAX_UPLOAD_BYTES`. HTTP 413 means the file is too large for the configured upload path. The UI's raw-byte upload is checked as the backend receives it.

## Check and build

```powershell
npm run typecheck
npm run build
npm run start
```

Build output should list `/`, `/runs/[runId]`, and `/runs/[runId]/report`. The last command serves the production build; run it instead of the development server on that port. There is currently no dedicated frontend unit-test or browser-test suite in `package.json`.

For a manual check, open a completed run, inspect the leaderboard and warnings, open its report, and try an unknown run ID. During a new run, verify that events arrive and approval controls appear when approval mode requires them.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Backend unreachable | Start `amla serve`; check `/api/health` and the configured API URL |
| Browser CORS error | Match the browser's exact origin in `AUTOML_CORS_ORIGINS`, then restart the backend |
| Run/report page missing after cloning | Confirm both route files are included and the root ignore rule is `/runs/` |
| Production calls an old backend | Rebuild after changing `NEXT_PUBLIC_API_URL` |
| Stream does not update behind a proxy | Disable SSE buffering and allow long connections; see the deployment guide |
| Optional chart or report is absent | Inspect run warnings and backend optional dependencies |
| HTTP 422 when starting a run | Read validation details; the API rejects unknown or invalid fields |

Provider credentials belong only on the backend. The current API assumes trusted access and does not implement authentication or tenant isolation. See [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md).
