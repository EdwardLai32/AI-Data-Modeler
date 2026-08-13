/**
 * Normalise a run-list row into the handful of facts the dashboard shows.
 *
 * `GET /api/runs` may return whole `RunSummary` rows or a trimmed projection
 * (see `RunListItem`). This reads whichever is present and leaves the field
 * `null` when neither is — the card then renders "—" rather than a zero.
 */

import type { ModelFamily, RunListItem, RunStatus, RunSummary, TaskType } from "@/types/api";
import { rankExperiments } from "@/lib/metrics";

export interface RunCard {
  runId: string;
  project: string;
  status: RunStatus;
  startedAt: string | null;
  finishedAt: string | null;
  durationSeconds: number | null;
  error: string | null;
  taskType: TaskType | null;
  targetColumn: string | null;
  primaryMetric: string | null;
  bestScore: number | null;
  bestFamily: ModelFamily | null;
  nRows: number | null;
  nColumns: number | null;
  nExperiments: number | null;
  grade: string | null;
  sourceUri: string | null;
}

function firstNumber(...values: Array<number | null | undefined>): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function firstString<T extends string>(...values: Array<T | null | undefined>): T | null {
  for (const value of values) {
    if (typeof value === "string" && value) return value;
  }
  return null;
}

export function deriveRunCard(item: RunListItem): RunCard {
  const experiments = item.experiments ?? null;
  const ranked = experiments ? rankExperiments(experiments) : null;
  const best = ranked?.best ?? null;

  return {
    runId: item.run_id,
    project: item.project ?? item.config?.project ?? "default",
    status: item.status,
    startedAt: item.started_at ?? null,
    finishedAt: item.finished_at ?? null,
    durationSeconds: firstNumber(item.duration_seconds),
    error: item.error ?? null,
    taskType: firstString<TaskType>(
      item.task_type,
      item.problem?.task_type,
      item.config?.task_type_override,
    ),
    targetColumn: firstString(
      // `GET /api/runs` calls this column `target`; a RunSummary row has neither.
      item.target,
      item.problem?.target_column,
      item.config?.target_column,
    ),
    primaryMetric: firstString(
      item.primary_metric,
      experiments?.primary_metric,
      item.problem?.primary_metric,
      item.config?.primary_metric_override,
    ),
    bestScore: firstNumber(item.best_score, best?.primary_score),
    bestFamily: firstString<ModelFamily>(item.best_family, best?.family),
    nRows: firstNumber(item.n_rows, item.ingestion?.n_rows),
    nColumns: firstNumber(item.n_columns, item.ingestion?.n_columns),
    nExperiments: firstNumber(item.n_experiments, experiments?.results?.length),
    grade: firstString(item.grade, item.evaluation?.overall_grade),
    sourceUri: firstString(item.config?.source?.uri),
  };
}

/** The same derivation from a full summary, for the run page header. */
export function summaryToCard(summary: RunSummary): RunCard {
  return deriveRunCard(summary);
}
