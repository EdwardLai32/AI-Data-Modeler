/**
 * Typed client for the AI Data Modeler FastAPI backend.
 *
 * Two rules shape this module:
 *
 * 1.  A dropped connection is a first-class outcome, not an exception to be
 *     swallowed. `ApiError.unreachable` distinguishes "the API said no" from
 *     "there is no API", because the UI shows a different thing for each — the
 *     second one needs the `amla serve` hint, not a red toast.
 * 2.  Nothing here invents data. When a field is absent the client returns
 *     `null`/`undefined` and the component renders "unknown".
 */

import type {
  ApprovalDecisionBody,
  ApprovalDecisionResponse,
  ArtifactRef,
  CancelResponse,
  CreateRunResponse,
  EventPage,
  HealthResponse,
  NewRunOptions,
  PlotlyFigure,
  QuestionAnswer,
  RunEvent,
  RunListItem,
  RunSummary,
  SourceKind,
  StartRunBody,
  UploadResponse,
} from "@/types/api";

/** Backend origin. Overridable at build time via `NEXT_PUBLIC_API_URL`. */
export const API_BASE: string = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

/** The command that starts the backend, quoted in every unreachable-state UI. */
export const SERVE_COMMAND = "amla serve";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  /** True when the request never reached a server (DNS, refused, CORS, offline). */
  readonly unreachable: boolean;
  readonly url: string;

  constructor(opts: {
    message: string;
    status?: number;
    detail?: string;
    unreachable?: boolean;
    url?: string;
  }) {
    super(opts.message);
    this.name = "ApiError";
    this.status = opts.status ?? 0;
    this.detail = opts.detail ?? opts.message;
    this.unreachable = opts.unreachable ?? false;
    this.url = opts.url ?? "";
  }
}

/** Narrow an unknown thrown value to `ApiError` without a cast at each call. */
export function asApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  if (error instanceof Error) return new ApiError({ message: error.message });
  return new ApiError({ message: String(error) });
}

/** FastAPI puts a string, a validation-error array, or nothing in `detail`. */
function readDetail(body: unknown, fallback: string): string {
  if (typeof body === "string" && body.trim()) return body.trim();
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) return detail.trim();
    if (Array.isArray(detail)) {
      const parts = detail
        .map((item) => {
          if (item && typeof item === "object") {
            const rec = item as { loc?: unknown; msg?: unknown };
            const loc = Array.isArray(rec.loc) ? rec.loc.join(".") : "";
            const msg = typeof rec.msg === "string" ? rec.msg : JSON.stringify(item);
            return loc ? `${loc}: ${msg}` : msg;
          }
          return String(item);
        })
        .filter(Boolean);
      if (parts.length) return parts.join("; ");
    }
    const message = (body as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) return message.trim();
  }
  return fallback;
}

async function parseBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

interface RequestOptions {
  method?: string;
  body?: BodyInit | null;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  /** Treat a 404 as "not there yet" and resolve to null instead of throwing. */
  nullOn404?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: options.method ?? "GET",
      body: options.body ?? null,
      headers: options.headers,
      signal: options.signal,
      cache: "no-store",
      mode: "cors",
    });
  } catch (error) {
    // fetch only rejects on a transport failure; an HTTP error still resolves.
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError({
      message: `Cannot reach the AI Data Modeler API at ${API_BASE}.`,
      unreachable: true,
      url,
      detail:
        error instanceof Error
          ? error.message
          : "The request failed before a response was received.",
    });
  }

  if (response.status === 204) return null as T;

  const body = await parseBody(response);

  if (!response.ok) {
    throw new ApiError({
      message: readDetail(body, `${response.status} ${response.statusText}`),
      status: response.status,
      detail: readDetail(body, `${response.status} ${response.statusText}`),
      url,
    });
  }

  return body as T;
}

function query(params: Record<string, string | number | boolean | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    search.set(key, String(value));
  }
  const rendered = search.toString();
  return rendered ? `?${rendered}` : "";
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

/** `GET /api/health` — also the reachability probe for the whole UI. */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", { signal });
}

/**
 * `GET /api/runs` — the dashboard list.
 *
 * Accepts either a bare array or `{runs: [...]}`; both shapes occur depending
 * on whether the endpoint paginates.
 */
export async function listRuns(
  opts: { project?: string; limit?: number; signal?: AbortSignal } = {},
): Promise<RunListItem[]> {
  const body = await request<RunListItem[] | { runs?: RunListItem[] }>(
    `/api/runs${query({ project: opts.project, limit: opts.limit ?? 50 })}`,
    { signal: opts.signal },
  );
  if (Array.isArray(body)) return body;
  return body?.runs ?? [];
}

/** `GET /api/runs/{id}` — the full run blackboard. */
export function getRun(runId: string, signal?: AbortSignal): Promise<RunSummary> {
  return request<RunSummary>(`/api/runs/${encodeURIComponent(runId)}`, { signal });
}

/** `POST /api/runs/{id}/cancel`. */
export function cancelRun(runId: string): Promise<CancelResponse> {
  return request<CancelResponse>(`/api/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
  });
}

/**
 * `GET /api/runs/{id}/events?after=N` — the replay half of the live feed.
 *
 * Used to backfill before the stream opens and to close any gap after a
 * reconnect, so a dropped connection never loses an event.
 */
export async function getEvents(
  runId: string,
  after = 0,
  signal?: AbortSignal,
): Promise<RunEvent[]> {
  const body = await request<RunEvent[] | EventPage>(
    `/api/runs/${encodeURIComponent(runId)}/events${query({ after })}`,
    { signal },
  );
  if (Array.isArray(body)) return body;
  return body?.events ?? [];
}

/** URL for the SSE endpoint. `EventSource` cannot send headers, hence a query cursor. */
export function streamUrl(runId: string, lastSequence: number): string {
  return `${API_BASE}/api/runs/${encodeURIComponent(runId)}/stream${query({
    last_sequence: lastSequence,
  })}`;
}

/** `POST /api/runs/{id}/ask` — grounded natural-language Q&A over the run. */
export function askRun(runId: string, question: string): Promise<QuestionAnswer> {
  return request<QuestionAnswer>(`/api/runs/${encodeURIComponent(runId)}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

/** `POST /api/runs/{id}/approvals/{request_id}` — resolve a suspended run. */
export function decideApproval(
  runId: string,
  requestId: string,
  body: ApprovalDecisionBody,
): Promise<ApprovalDecisionResponse> {
  return request<ApprovalDecisionResponse>(
    `/api/runs/${encodeURIComponent(runId)}/approvals/${encodeURIComponent(requestId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

/**
 * `GET /api/runs/{id}/artifacts` — chart/report/model files on disk.
 *
 * Optional: when the backend does not expose it, the caller falls back to the
 * paths already present on `RunSummary.visualizations`.
 */
export async function listArtifacts(
  runId: string,
  signal?: AbortSignal,
): Promise<ArtifactRef[]> {
  const body = await request<ArtifactRef[] | { artifacts?: ArtifactRef[] }>(
    `/api/runs/${encodeURIComponent(runId)}/artifacts`,
    { signal },
  );
  if (Array.isArray(body)) return body;
  return body?.artifacts ?? [];
}

/**
 * URL that serves one artifact's bytes.
 *
 * The executors write absolute paths (`<workspace>/runs/<run_id>/charts/roc.html`).
 * The portion after the run id is the artifact's key, so it is recovered here
 * rather than requiring the backend to echo a URL back for every file.
 */
export function artifactUrl(runId: string, serverPath: string): string {
  const normalised = serverPath.replace(/\\/g, "/");
  const marker = `/${runId}/`;
  const index = normalised.indexOf(marker);
  const relative =
    index >= 0
      ? normalised.slice(index + marker.length)
      : (normalised.split("/").pop() ?? normalised);
  const encoded = relative
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `${API_BASE}/api/runs/${encodeURIComponent(runId)}/artifacts/${encoded}`;
}

/** Fetch a plotly figure written by `plotly.io.write_json`. */
export async function getFigure(
  runId: string,
  jsonPath: string,
  signal?: AbortSignal,
): Promise<PlotlyFigure> {
  const figure = await request<PlotlyFigure>(artifactUrl(runId, jsonPath), { signal });
  if (!figure || !Array.isArray(figure.data)) {
    throw new ApiError({
      message: "Chart JSON is not a plotly figure (no `data` array).",
      url: jsonPath,
    });
  }
  return figure;
}

/** Fetch an artifact as text — used for the raw-markdown report view. */
export async function getArtifactText(
  runId: string,
  serverPath: string,
  signal?: AbortSignal,
): Promise<string> {
  const url = artifactUrl(runId, serverPath);
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store", mode: "cors", signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError({
      message: `Cannot reach the AI Data Modeler API at ${API_BASE}.`,
      unreachable: true,
      url,
    });
  }
  if (!response.ok) {
    throw new ApiError({
      message: `Artifact unavailable (${response.status}).`,
      status: response.status,
      url,
    });
  }
  return response.text();
}

// ---------------------------------------------------------------------------
// Run creation
// ---------------------------------------------------------------------------

export interface CreateRunInput {
  /** A browser-side file, uploaded before the run starts. Excludes `sourceUri`. */
  file?: File | null;
  /** A server-side path, URL, or connection string. */
  sourceUri?: string;
  sourceKind?: SourceKind;
  options: NewRunOptions;
}

/** Infer the `SourceKind` from a filename or URI extension. */
export function inferSourceKind(name: string): SourceKind {
  const lower = name.toLowerCase().split("?")[0] ?? "";
  if (lower.endsWith(".csv") || lower.endsWith(".tsv") || lower.endsWith(".txt")) return "csv";
  if (lower.endsWith(".xlsx") || lower.endsWith(".xls") || lower.endsWith(".xlsm")) return "excel";
  if (lower.endsWith(".parquet") || lower.endsWith(".pq")) return "parquet";
  if (lower.endsWith(".json") || lower.endsWith(".jsonl") || lower.endsWith(".ndjson"))
    return "json";
  if (lower.endsWith(".duckdb") || lower.endsWith(".ddb")) return "duckdb";
  if (lower.startsWith("s3://")) return "s3";
  if (lower.startsWith("gs://")) return "gcs";
  if (lower.startsWith("postgres://") || lower.startsWith("postgresql://")) return "postgres";
  if (lower.startsWith("mysql://")) return "mysql";
  if (lower.startsWith("http://") || lower.startsWith("https://")) return "rest_api";
  return "csv";
}

/**
 * Translate the form's own field names into `StartRunRequest`'s.
 *
 * The two differ deliberately: the panel labels these as *overrides* (which is
 * what `RunConfig` calls them), while the HTTP model calls them `task_type` and
 * `primary_metric`. `StartRunRequest` is `extra="forbid"`, so sending the
 * `*_override` spelling is a 422, not a silently ignored field.
 */
function optionsToBody(options: NewRunOptions): StartRunBody {
  const body: StartRunBody = {
    project: options.project.trim() || "default",
    time_budget_seconds: options.time_budget_seconds,
    max_experiments: options.max_experiments,
    max_rows: options.max_rows,
    test_size: options.test_size,
    cv_folds: options.cv_folds,
    random_state: options.random_state,
    require_approval: options.require_approval,
    enable_tuning: options.enable_tuning,
    enable_explainability: options.enable_explainability,
    enable_self_improvement: options.enable_self_improvement,
    report_formats: options.report_formats,
    fairness_attributes: splitList(options.fairness_attributes),
  };
  if (options.target_column.trim()) body.target_column = options.target_column.trim();
  if (options.primary_metric_override.trim())
    body.primary_metric = options.primary_metric_override.trim();
  if (options.task_type_override) body.task_type = options.task_type_override;
  if (options.notes.trim()) body.notes = options.notes.trim();
  return body;
}

/** Split a comma/newline separated free-text list into trimmed entries. */
export function splitList(raw: string): string[] {
  return raw
    .split(/[,\n]/)
    .map((part) => part.trim())
    .filter(Boolean);
}

/**
 * `POST /api/upload` — put a browser-side file where the server can read it.
 *
 * Uses the raw-body form of the endpoint (bytes plus `?filename=`) rather than
 * multipart: it is the one upload path that works whether or not the backend
 * has `python-multipart` installed, and it needs no boundary handling here.
 */
export async function uploadDataset(file: File): Promise<UploadResponse> {
  return request<UploadResponse>(`/api/upload${query({ filename: file.name })}`, {
    method: "POST",
    body: file,
    headers: { "Content-Type": "application/octet-stream" },
  });
}

/**
 * Start an analysis.
 *
 * A chosen file is uploaded first (`POST /api/upload`), then the run is started
 * with a JSON body (`POST /api/runs`). JSON rather than the multipart form of
 * `/api/runs` because only JSON can carry a nested `DataSource`, which is the
 * only way to honour an explicit source-kind choice — `StartRunRequest` has no
 * scalar `source_kind` field and rejects unknown keys.
 */
export async function createRun(input: CreateRunInput): Promise<CreateRunResponse> {
  const body = optionsToBody(input.options);

  if (input.file) {
    const uploaded = await uploadDataset(input.file);
    if (input.sourceKind) {
      body.source = { ...uploaded.source, kind: input.sourceKind };
    } else {
      body.upload_path = uploaded.path;
    }
    return request<CreateRunResponse>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  const uri = (input.sourceUri ?? "").trim();
  if (!uri) {
    throw new ApiError({ message: "Attach a file or provide a source path/URI." });
  }
  if (input.sourceKind) {
    body.source = {
      kind: input.sourceKind,
      uri,
      query: null,
      options: [],
      secret_env: [],
    };
  } else {
    body.uri = uri;
  }
  return request<CreateRunResponse>("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Defaults for the "New analysis" panel, matching `RunConfig`'s own defaults. */
export function defaultRunOptions(): NewRunOptions {
  return {
    project: "default",
    target_column: "",
    time_budget_seconds: 900,
    max_experiments: 8,
    max_rows: null,
    test_size: 0.2,
    cv_folds: 5,
    random_state: 42,
    require_approval: false,
    enable_tuning: true,
    enable_explainability: true,
    enable_self_improvement: true,
    primary_metric_override: "",
    task_type_override: "",
    fairness_attributes: "",
    report_formats: ["markdown", "html", "json"],
    notes: "",
  };
}
