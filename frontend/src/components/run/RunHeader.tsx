"use client";

import Link from "next/link";
import { useCallback, useState } from "react";

import { Badge, Chip } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Hero, StatTile } from "@/components/ui/Metrics";
import { ErrorBanner, Spinner } from "@/components/ui/States";
import { asApiError, cancelRun } from "@/lib/api";
import {
  compactNumber,
  duration,
  elapsedSeconds,
  EMPTY,
  integer,
  metricLabel,
  metricValue,
  timestamp,
} from "@/lib/format";
import { familyLabel, gradeTone, isRunActive, runStatusLabel, runStatusTone, taskLabel } from "@/lib/labels";
import { rankExperiments } from "@/lib/metrics";
import { useNow } from "@/hooks/useNow";
import type { RunSummary } from "@/types/api";

/**
 * Run header: identity, status, and the one number the run is judged on.
 *
 * The hero is the best primary-metric score, or an explicit "no scored
 * experiment yet" — never a zero standing in for an absent measurement.
 */
export function RunHeader({
  summary,
  refreshing,
  onRefresh,
}: {
  summary: RunSummary;
  refreshing: boolean;
  onRefresh: () => void;
}) {
  const active = isRunActive(summary.status);
  const now = useNow(1000, active);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  const { best, direction } = rankExperiments(summary.experiments);
  const metric =
    summary.experiments?.primary_metric ||
    best?.primary_metric ||
    summary.problem?.primary_metric ||
    summary.config.primary_metric_override ||
    "";

  const elapsed =
    summary.duration_seconds > 0
      ? summary.duration_seconds
      : elapsedSeconds(summary.started_at, summary.finished_at, now);

  const cancel = useCallback(async () => {
    setCancelError(null);
    setCancelling(true);
    try {
      await cancelRun(summary.run_id);
      onRefresh();
    } catch (raw) {
      setCancelError(asApiError(raw).detail);
    } finally {
      setCancelling(false);
    }
  }, [summary.run_id, onRefresh]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs text-ink-3">
        <Link href="/" className="rounded-md hover:text-ink">
          ← All runs
        </Link>
        <span aria-hidden="true">/</span>
        <span className="font-mono text-ink-2">{summary.run_id}</span>
        <Chip>{summary.project}</Chip>
        {summary.config.source.uri ? (
          <Chip mono title={summary.config.source.uri}>
            {summary.config.source.kind}
          </Chip>
        ) : null}
      </div>

      <div className="rounded-xl border border-hairline bg-surface px-4 py-4 sm:px-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <h1 className="text-base font-semibold tracking-tight text-ink">
              {taskLabel(summary.problem?.task_type ?? summary.config.task_type_override)}
            </h1>
            <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
              <span>
                Target:{" "}
                <span className="font-mono text-ink-2">
                  {summary.problem?.target_column ?? summary.config.target_column ?? "none"}
                </span>
              </span>
              <span aria-hidden="true">·</span>
              <span>Started {timestamp(summary.started_at)}</span>
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {refreshing ? <Spinner label="Refreshing run" /> : null}
            <Badge tone={runStatusTone(summary.status)} pulse={active}>
              {runStatusLabel(summary.status)}
            </Badge>
            {summary.evaluation ? (
              <Badge tone={gradeTone(summary.evaluation.overall_grade)}>
                Grade {summary.evaluation.overall_grade}
              </Badge>
            ) : null}
            {summary.replans > 0 ? <Chip>{summary.replans} replans</Chip> : null}
            <Button size="sm" onClick={onRefresh} disabled={refreshing}>
              Refresh
            </Button>
            {summary.report ? (
              <Link
                href={`/runs/${summary.run_id}/report`}
                className="inline-flex items-center rounded-lg border border-transparent bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-strong"
              >
                View report
              </Link>
            ) : null}
            {active ? (
              <Button
                size="sm"
                variant="danger"
                onClick={() => void cancel()}
                disabled={cancelling}
              >
                {cancelling ? "Cancelling…" : "Cancel run"}
              </Button>
            ) : null}
          </div>
        </div>

        <div className="mt-4 border-t border-hairline pt-4">
          <Hero
            label={metric ? `Best ${metricLabel(metric)}` : "Best score"}
            value={metricValue(best?.primary_score ?? null)}
            note={
              best
                ? `${familyLabel(best.family)}${best.tuned ? " (tuned)" : ""} · ${
                    direction ? "higher is better" : "lower is better"
                  }`
                : "No scored experiment yet"
            }
          />
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
          <StatTile
            label="Rows × columns"
            value={
              summary.ingestion
                ? `${compactNumber(summary.ingestion.n_rows)} × ${integer(summary.ingestion.n_columns)}`
                : EMPTY
            }
            note={summary.ingestion?.truncated ? "row cap applied" : undefined}
          />
          <StatTile
            label="Experiments"
            value={integer(summary.experiments?.results.length ?? 0)}
            note={
              summary.experiments
                ? `${summary.experiments.results.filter((row) => row.failed).length} failed`
                : undefined
            }
          />
          <StatTile
            label={active ? "Elapsed" : "Duration"}
            value={duration(elapsed)}
            note={`budget ${duration(summary.config.time_budget_seconds)}`}
          />
          <StatTile
            label="Steps complete"
            value={`${summary.steps.filter((step) => step.status === "completed").length}/${summary.steps.length}`}
          />
          <StatTile
            label="Warnings"
            value={integer(summary.warnings.length)}
            note={summary.warnings.length ? "see the warnings panel" : "none recorded"}
          />
        </div>
      </div>

      {summary.error ? (
        <ErrorBanner title="The run failed" detail={summary.error} onRetry={onRefresh} retryLabel="Reload" />
      ) : null}

      {cancelError ? <ErrorBanner title="Cancel request failed" detail={cancelError} /> : null}
    </div>
  );
}
