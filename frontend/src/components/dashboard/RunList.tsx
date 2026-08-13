"use client";

import Link from "next/link";

import { Badge, Chip } from "@/components/ui/Badge";
import { EmptyState, PanelSkeleton } from "@/components/ui/States";
import {
  duration,
  elapsedSeconds,
  EMPTY,
  integer,
  metricLabel,
  metricValue,
  relativeTime,
} from "@/lib/format";
import { familyLabel, gradeTone, isRunActive, runStatusLabel, runStatusTone, taskLabel } from "@/lib/labels";
import { useNow } from "@/hooks/useNow";
import type { RunCard } from "@/lib/runCard";

/**
 * The run history.
 *
 * One markup tree serves every width: the cells carry their own labels below
 * `lg`, where the column header row is hidden. A duplicated mobile layout would
 * drift out of sync with the desktop one.
 */

const GRID =
  "grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4 lg:grid-cols-[minmax(0,2.4fr)_minmax(0,1.4fr)_minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,0.8fr)] lg:gap-y-0 lg:items-center";

function Cell({
  label,
  children,
  className = "",
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`min-w-0 ${className}`}>
      <span className="block text-[10px] font-medium uppercase tracking-wide text-ink-3 lg:hidden">
        {label}
      </span>
      {children}
    </div>
  );
}

export function RunList({
  runs,
  loading,
}: {
  runs: RunCard[];
  loading: boolean;
}) {
  // Only ticks while something is still running, so a settled list is static.
  const now = useNow(1000, runs.some((run) => isRunActive(run.status)));

  if (loading && !runs.length) return <PanelSkeleton rows={5} />;

  if (!runs.length) {
    return (
      <EmptyState
        title="No runs yet"
        hint="Start one with the New analysis panel. Every run you launch appears here with its status, best score, and duration."
      />
    );
  }

  return (
    <div>
      <div
        className={`${GRID} hidden border-b border-hairline px-3 pb-2 lg:grid`}
        aria-hidden="true"
      >
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink-3">Run</span>
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink-3">Task</span>
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink-3">
          Best score
        </span>
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink-3">
          Duration
        </span>
        <span className="text-right text-[10px] font-medium uppercase tracking-wide text-ink-3">
          Status
        </span>
      </div>

      <ul className="divide-y divide-hairline">
        {runs.map((run) => (
          <li key={run.runId}>
            <Link
              href={`/runs/${run.runId}`}
              className={`${GRID} rounded-lg px-3 py-3 transition-colors hover:bg-surface-2`}
            >
              <Cell label="Run" className="col-span-2 sm:col-span-2 lg:col-span-1">
                <span className="block truncate font-mono text-[13px] text-ink">{run.runId}</span>
                <span className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-3">
                  <span className="truncate">{run.project}</span>
                  <span aria-hidden="true">·</span>
                  <span>{relativeTime(run.startedAt)}</span>
                  {run.nRows !== null ? (
                    <>
                      <span aria-hidden="true">·</span>
                      <span className="tabular">
                        {integer(run.nRows)} rows
                        {run.nColumns !== null ? ` × ${integer(run.nColumns)} cols` : ""}
                      </span>
                    </>
                  ) : null}
                </span>
              </Cell>

              <Cell label="Task">
                <span className="block truncate text-xs text-ink-2">
                  {run.taskType ? taskLabel(run.taskType) : EMPTY}
                </span>
                {run.targetColumn ? (
                  <span className="mt-0.5 block truncate font-mono text-[11px] text-ink-3">
                    target: {run.targetColumn}
                  </span>
                ) : null}
              </Cell>

              <Cell label="Best score">
                <span className="tabular block truncate text-sm font-semibold text-ink">
                  {metricValue(run.bestScore)}
                </span>
                <span className="mt-0.5 block truncate text-[11px] text-ink-3">
                  {run.primaryMetric ? metricLabel(run.primaryMetric) : EMPTY}
                  {run.bestFamily ? ` · ${familyLabel(run.bestFamily)}` : ""}
                </span>
              </Cell>

              <Cell label="Duration">
                <span className="tabular block text-xs text-ink-2">
                  {/* A run in flight has duration_seconds 0 until it finishes, so
                      the elapsed time is derived from started_at instead. */}
                  {duration(
                    run.durationSeconds && run.durationSeconds > 0
                      ? run.durationSeconds
                      : elapsedSeconds(run.startedAt, run.finishedAt, now),
                  )}
                </span>
                {run.nExperiments !== null ? (
                  <span className="mt-0.5 block text-[11px] text-ink-3">
                    {integer(run.nExperiments)} experiments
                  </span>
                ) : null}
              </Cell>

              <Cell label="Status" className="lg:text-right">
                <span className="inline-flex flex-wrap items-center justify-end gap-1.5">
                  <Badge
                    tone={runStatusTone(run.status)}
                    pulse={isRunActive(run.status)}
                    title={run.error ?? undefined}
                  >
                    {runStatusLabel(run.status)}
                  </Badge>
                  {run.grade ? (
                    <Badge tone={gradeTone(run.grade)}>Grade {run.grade}</Badge>
                  ) : null}
                </span>
                {run.error ? (
                  <span className="mt-1 block truncate text-[11px] text-ink-3" title={run.error}>
                    {run.error}
                  </span>
                ) : null}
              </Cell>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Aggregate tiles over the loaded page of runs — counts, not estimates. */
export function RunListStats({ runs }: { runs: RunCard[] }) {
  if (!runs.length) return null;
  const completed = runs.filter((run) => run.status === "completed").length;
  const active = runs.filter((run) => isRunActive(run.status)).length;
  const failed = runs.filter((run) => run.status === "failed").length;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Chip>{runs.length} loaded</Chip>
      {active ? <Badge tone="accent">{active} active</Badge> : null}
      {completed ? <Badge tone="good">{completed} completed</Badge> : null}
      {failed ? <Badge tone="critical">{failed} failed</Badge> : null}
    </div>
  );
}
