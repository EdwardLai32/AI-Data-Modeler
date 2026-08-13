"use client";

import { useCallback, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/Button";
import { Checkbox, Field, Select, TextArea, TextInput } from "@/components/ui/Field";
import { Chip } from "@/components/ui/Badge";
import { ApiFailure } from "@/components/ui/States";
import { Panel } from "@/components/ui/Panel";
import {
  ApiError,
  asApiError,
  createRun,
  defaultRunOptions,
  inferSourceKind,
} from "@/lib/api";
import { bytes } from "@/lib/format";
import { taskLabel } from "@/lib/labels";
import type { NewRunOptions, SourceKind, TaskType } from "@/types/api";

/**
 * Launch a run.
 *
 * The target column is optional on purpose: leaving it blank is a supported
 * path, because the Dataset agent ranks candidate targets from the profile. The
 * field offers a datalist of the dropped file's own header row — parsed in the
 * browser from the file the user chose, so the suggestions are real column names
 * and no upload is needed to see them.
 */

const ACCEPT = ".csv,.tsv,.txt,.json,.jsonl,.ndjson,.parquet,.pq,.xlsx,.xls,.xlsm";
const HEADER_SNIFF_BYTES = 65_536;

const TASK_OPTIONS: TaskType[] = [
  "binary_classification",
  "multiclass_classification",
  "regression",
  "time_series_forecasting",
  "clustering",
  "anomaly_detection",
];

const REPORT_FORMATS = ["markdown", "html", "json", "pdf", "pptx"] as const;

/** Split a header line on the delimiter that actually dominates it. */
function parseHeader(line: string): string[] {
  const cleaned = line.replace(/^\uFEFF/, "").replace(/\r$/, "");
  const counts: Array<[string, number]> = [
    [",", (cleaned.match(/,/g) ?? []).length],
    ["\t", (cleaned.match(/\t/g) ?? []).length],
    [";", (cleaned.match(/;/g) ?? []).length],
    ["|", (cleaned.match(/\|/g) ?? []).length],
  ];
  counts.sort((a, b) => b[1] - a[1]);
  const [delimiter, count] = counts[0] as [string, number];
  if (count === 0) return [];
  return cleaned
    .split(delimiter)
    .map((cell) => cell.trim().replace(/^["']|["']$/g, ""))
    .filter(Boolean);
}

async function sniffColumns(file: File): Promise<string[]> {
  const name = file.name.toLowerCase();
  const delimited =
    name.endsWith(".csv") || name.endsWith(".tsv") || name.endsWith(".txt");
  if (!delimited) return [];
  const text = await file.slice(0, HEADER_SNIFF_BYTES).text();
  const firstLine = text.split(/\r?\n/, 1)[0] ?? "";
  return parseHeader(firstLine);
}

export function NewAnalysisPanel({ onStarted }: { onStarted?: () => void }) {
  const router = useRouter();
  const ids = useId();
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [sourceUri, setSourceUri] = useState("");
  const [sourceKind, setSourceKind] = useState<SourceKind | "">("");
  const [dragging, setDragging] = useState(false);
  const [options, setOptions] = useState<NewRunOptions>(defaultRunOptions());
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const field = (name: string) => `${ids}-${name}`;

  const patch = useCallback(<K extends keyof NewRunOptions>(key: K, value: NewRunOptions[K]) => {
    setOptions((previous) => ({ ...previous, [key]: value }));
  }, []);

  const acceptFile = useCallback(async (next: File | null) => {
    setFile(next);
    setColumns([]);
    setNotice(null);
    if (!next) return;
    setSourceUri("");
    try {
      const sniffed = await sniffColumns(next);
      setColumns(sniffed);
      if (!sniffed.length) {
        setNotice(
          "Column names are read from the header row of delimited text files only; type the target manually or leave it blank.",
        );
      }
    } catch {
      // Sniffing is a convenience, never a gate on submitting the run.
      setNotice("Could not read the header row locally; type the target manually if you need one.");
    }
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragging(false);
      const dropped = event.dataTransfer.files?.[0];
      if (dropped) void acceptFile(dropped);
    },
    [acceptFile],
  );

  const submit = useCallback(async () => {
    setError(null);
    setSubmitting(true);
    try {
      const response = await createRun({
        file,
        sourceUri,
        sourceKind: sourceKind || undefined,
        options,
      });
      if (!response?.run_id) {
        throw new ApiError({
          message: "The API accepted the request but did not return a run id.",
        });
      }
      onStarted?.();
      router.push(`/runs/${response.run_id}`);
    } catch (raw) {
      setError(asApiError(raw));
    } finally {
      setSubmitting(false);
    }
  }, [file, sourceUri, sourceKind, options, onStarted, router]);

  const ready = Boolean(file) || Boolean(sourceUri.trim());
  const kindGuess = file
    ? inferSourceKind(file.name)
    : sourceUri.trim()
      ? inferSourceKind(sourceUri.trim())
      : null;

  return (
    <Panel
      title="New analysis"
      subtitle="Drop a dataset, name the target if you know it, and the planner takes it from there."
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (ready && !submitting) void submit();
        }}
        className="space-y-4"
      >
        {/* ---- source ---------------------------------------------------- */}
        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`rounded-xl border-2 border-dashed px-4 py-5 text-center transition-colors ${
            dragging ? "border-accent bg-accent-wash" : "border-hairline-strong bg-surface-2"
          }`}
        >
          <input
            ref={inputRef}
            id={field("file")}
            type="file"
            accept={ACCEPT}
            className="peer sr-only"
            onChange={(event) => void acceptFile(event.target.files?.[0] ?? null)}
          />
          <label
            htmlFor={field("file")}
            className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-hairline-strong bg-surface px-3 py-2 text-sm font-medium text-ink hover:bg-surface-2 peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent"
          >
            Choose a dataset file
          </label>
          <p className="mt-2 text-xs text-ink-3">
            or drag it here — CSV, TSV, Excel, Parquet, JSON
          </p>

          {file ? (
            <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
              <Chip mono title={file.name}>
                {file.name}
              </Chip>
              <Chip>{bytes(file.size)}</Chip>
              {kindGuess ? <Chip>{kindGuess}</Chip> : null}
              {columns.length ? <Chip>{columns.length} columns detected</Chip> : null}
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  void acceptFile(null);
                  if (inputRef.current) inputRef.current.value = "";
                }}
              >
                Remove
              </Button>
            </div>
          ) : null}
          {notice ? <p className="mt-2 text-[11px] text-ink-3">{notice}</p> : null}
        </div>

        <Field
          label="Or a path the server can read"
          htmlFor={field("uri")}
          hint="An absolute path, URL, or connection string. Use this when the file is already on the machine running the API."
        >
          <TextInput
            id={field("uri")}
            value={sourceUri}
            placeholder="C:\data\churn.csv"
            aria-describedby={`${field("uri")}-hint`}
            disabled={Boolean(file)}
            onChange={(event) => setSourceUri(event.target.value)}
          />
        </Field>

        {/* ---- target + basics ------------------------------------------- */}
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Target column"
            htmlFor={field("target")}
            hint="Leave blank to let the Dataset agent rank candidate targets from the profile."
          >
            <TextInput
              id={field("target")}
              list={columns.length ? field("columns") : undefined}
              value={options.target_column}
              placeholder="optional"
              aria-describedby={`${field("target")}-hint`}
              onChange={(event) => patch("target_column", event.target.value)}
            />
            {columns.length ? (
              <datalist id={field("columns")}>
                {columns.map((column) => (
                  <option key={column} value={column} />
                ))}
              </datalist>
            ) : null}
          </Field>

          <Field label="Project" htmlFor={field("project")}>
            <TextInput
              id={field("project")}
              value={options.project}
              onChange={(event) => patch("project", event.target.value)}
            />
          </Field>

          <Field
            label="Time budget (seconds)"
            htmlFor={field("budget")}
            hint="Soft cap the orchestrator plans against."
          >
            <TextInput
              id={field("budget")}
              type="number"
              min={30}
              step={30}
              value={options.time_budget_seconds}
              aria-describedby={`${field("budget")}-hint`}
              onChange={(event) =>
                patch("time_budget_seconds", Number(event.target.value) || 900)
              }
            />
          </Field>

          <Field label="Max experiments" htmlFor={field("experiments")}>
            <TextInput
              id={field("experiments")}
              type="number"
              min={1}
              max={50}
              value={options.max_experiments}
              onChange={(event) => patch("max_experiments", Number(event.target.value) || 8)}
            />
          </Field>
        </div>

        <div className="space-y-2.5">
          <Checkbox
            id={field("approval")}
            label="Pause for approval before destructive steps"
            hint="Dropping columns or rows suspends the run until you approve it here."
            checked={options.require_approval}
            onChange={(value) => patch("require_approval", value)}
          />
          <Checkbox
            id={field("tuning")}
            label="Hyperparameter tuning"
            checked={options.enable_tuning}
            onChange={(value) => patch("enable_tuning", value)}
          />
          <Checkbox
            id={field("explain")}
            label="Explainability (SHAP / permutation importance)"
            checked={options.enable_explainability}
            onChange={(value) => patch("enable_explainability", value)}
          />
        </div>

        {/* ---- advanced -------------------------------------------------- */}
        <details className="group rounded-lg border border-hairline bg-surface-2 px-3 py-2">
          <summary className="cursor-pointer list-none text-xs font-medium text-ink-2 hover:text-ink">
            <span aria-hidden="true" className="mr-1.5 inline-block transition-transform group-open:rotate-90">
              ▸
            </span>
            Advanced options
          </summary>

          <div className="mt-3 space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <Field
                label="Task type override"
                htmlFor={field("task")}
                hint="Only set this if the Problem agent's inference is wrong."
              >
                <Select
                  id={field("task")}
                  value={options.task_type_override}
                  aria-describedby={`${field("task")}-hint`}
                  onChange={(event) =>
                    patch("task_type_override", event.target.value as TaskType | "")
                  }
                >
                  <option value="">Let the agent decide</option>
                  {TASK_OPTIONS.map((task) => (
                    <option key={task} value={task}>
                      {taskLabel(task)}
                    </option>
                  ))}
                </Select>
              </Field>

              <Field
                label="Primary metric override"
                htmlFor={field("metric")}
                hint="e.g. roc_auc, f1, rmse, mae, r2"
              >
                <TextInput
                  id={field("metric")}
                  value={options.primary_metric_override}
                  placeholder="agent's choice"
                  aria-describedby={`${field("metric")}-hint`}
                  onChange={(event) => patch("primary_metric_override", event.target.value)}
                />
              </Field>

              <Field label="Test size" htmlFor={field("test")}>
                <TextInput
                  id={field("test")}
                  type="number"
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  value={options.test_size}
                  onChange={(event) => patch("test_size", Number(event.target.value) || 0.2)}
                />
              </Field>

              <Field label="CV folds" htmlFor={field("folds")}>
                <TextInput
                  id={field("folds")}
                  type="number"
                  min={2}
                  max={20}
                  value={options.cv_folds}
                  onChange={(event) => patch("cv_folds", Number(event.target.value) || 5)}
                />
              </Field>

              <Field
                label="Row cap"
                htmlFor={field("rows")}
                hint="Sample very large tables. Blank means load everything."
              >
                <TextInput
                  id={field("rows")}
                  type="number"
                  min={100}
                  step={1000}
                  value={options.max_rows ?? ""}
                  placeholder="no cap"
                  aria-describedby={`${field("rows")}-hint`}
                  onChange={(event) =>
                    patch("max_rows", event.target.value ? Number(event.target.value) : null)
                  }
                />
              </Field>

              <Field label="Random state" htmlFor={field("seed")}>
                <TextInput
                  id={field("seed")}
                  type="number"
                  value={options.random_state}
                  onChange={(event) => patch("random_state", Number(event.target.value) || 42)}
                />
              </Field>
            </div>

            <Field
              label="Fairness attributes"
              htmlFor={field("fairness")}
              hint="Comma-separated column names to slice evaluation by."
            >
              <TextInput
                id={field("fairness")}
                value={options.fairness_attributes}
                placeholder="region, age_band"
                aria-describedby={`${field("fairness")}-hint`}
                onChange={(event) => patch("fairness_attributes", event.target.value)}
              />
            </Field>

            <fieldset>
              <legend className="text-xs font-medium text-ink-2">Report formats</legend>
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-2">
                {REPORT_FORMATS.map((format) => (
                  <Checkbox
                    key={format}
                    id={field(`format-${format}`)}
                    label={format}
                    checked={options.report_formats.includes(format)}
                    onChange={(checked) =>
                      patch(
                        "report_formats",
                        checked
                          ? [...options.report_formats, format]
                          : options.report_formats.filter((item) => item !== format),
                      )
                    }
                  />
                ))}
              </div>
            </fieldset>

            <Checkbox
              id={field("self")}
              label="Self-improvement (replan when evaluation rejects the model)"
              checked={options.enable_self_improvement}
              onChange={(value) => patch("enable_self_improvement", value)}
            />

            <Field
              label="Source kind"
              htmlFor={field("kind")}
              hint={
                kindGuess
                  ? `Detected "${kindGuess}" from the file extension.`
                  : "Detected from the file extension."
              }
            >
              <Select
                id={field("kind")}
                value={sourceKind}
                aria-describedby={`${field("kind")}-hint`}
                onChange={(event) => setSourceKind(event.target.value as SourceKind | "")}
              >
                <option value="">Detect automatically</option>
                {(
                  [
                    "csv",
                    "excel",
                    "json",
                    "parquet",
                    "sql",
                    "postgres",
                    "mysql",
                    "duckdb",
                    "s3",
                    "gcs",
                    "azure_blob",
                    "rest_api",
                    "kaggle",
                  ] as SourceKind[]
                ).map((kind) => (
                  <option key={kind} value={kind}>
                    {kind}
                  </option>
                ))}
              </Select>
            </Field>

            <Field label="Notes" htmlFor={field("notes")}>
              <TextArea
                id={field("notes")}
                rows={2}
                value={options.notes}
                placeholder="Context the agents should know about this dataset."
                onChange={(event) => patch("notes", event.target.value)}
              />
            </Field>
          </div>
        </details>

        {error ? <ApiFailure error={error} context="Starting the run" /> : null}

        <div className="flex items-center gap-3">
          <Button type="submit" variant="primary" disabled={!ready || submitting}>
            {submitting ? "Starting…" : "Start analysis"}
          </Button>
          {!ready ? (
            <p className="text-xs text-ink-3">Attach a file or enter a source path first.</p>
          ) : null}
        </div>
      </form>
    </Panel>
  );
}
