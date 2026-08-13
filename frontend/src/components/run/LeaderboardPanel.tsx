"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { Panel } from "@/components/ui/Panel";
import { DataBar } from "@/components/ui/Metrics";
import { Disclosure } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { bytes, duration, integer, metricLabel, metricValue } from "@/lib/format";
import { familyLabel } from "@/lib/labels";
import { cvSpread, rankExperiments, relativeBarFraction } from "@/lib/metrics";
import type { ExperimentLog, ExperimentResult } from "@/types/api";

/**
 * Model leaderboard.
 *
 * Sorted by the primary metric in the direction the backend reports, with the
 * baseline explicitly marked — a leaderboard without its baseline cannot answer
 * "is any of this better than guessing?", which is the first question worth
 * asking. Bar length is relative to the range of scores on this board (a full
 * 0-1 axis makes every ROC AUC look identical), so every row also carries its
 * literal score.
 */

function ResultRow({
  result,
  rank,
  isBest,
  fraction,
  metric,
}: {
  result: ExperimentResult;
  rank: number;
  isBest: boolean;
  fraction: number;
  metric: string;
}) {
  const spread = cvSpread(result);
  return (
    <li
      className={`rounded-lg px-2.5 py-2 ${
        isBest ? "bg-accent-wash ring-1 ring-accent/40" : ""
      }`}
    >
      <DataBar
        name={
          <>
            <span className="tabular w-4 shrink-0 text-ink-3">{rank}</span>
            <span className="truncate font-medium">{familyLabel(result.family)}</span>
            {result.is_baseline ? <Badge tone="neutral">Baseline</Badge> : null}
            {result.tuned ? <Badge tone="accent">Tuned</Badge> : null}
            {isBest ? <Badge tone="good">Best</Badge> : null}
          </>
        }
        nameTitle={result.label || result.experiment_id}
        fraction={fraction}
        tone={result.is_baseline ? "neutral" : "accent"}
        valueText={metricValue(result.primary_score)}
      />
      <p className="tabular mt-0.5 flex flex-wrap gap-x-3 text-[11px] text-ink-3">
        <span>{metricLabel(metric)}</span>
        {spread !== null ? <span>± {metricValue(spread)} across folds</span> : null}
        {result.train_seconds > 0 ? <span>fit {duration(result.train_seconds)}</span> : null}
        {result.n_features_in > 0 ? <span>{integer(result.n_features_in)} features</span> : null}
        {result.model_size_bytes > 0 ? <span>{bytes(result.model_size_bytes)}</span> : null}
      </p>

      {result.metrics.length || result.params.length || result.cv_scores.length ? (
        <Disclosure summary="Metrics and parameters">
          <div className="space-y-2">
            {result.metrics.length ? (
              <div className="flex flex-wrap gap-1.5">
                {result.metrics.map((metricValueEntry, index) => (
                  <Chip key={`${metricValueEntry.name}-${index}`}>
                    {metricLabel(metricValueEntry.name)} {metricValue(metricValueEntry.value)}
                    {metricValueEntry.std !== null ? ` ± ${metricValue(metricValueEntry.std)}` : ""}
                  </Chip>
                ))}
              </div>
            ) : null}
            {result.cv_scores.length ? (
              <p className="tabular text-[11px] text-ink-2">
                <span className="text-ink-3">Folds: </span>
                {result.cv_scores.map((score) => metricValue(score)).join(", ")}
              </p>
            ) : null}
            {result.params.length ? (
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5">
                {result.params.map((param, index) => (
                  <div key={`${param.key}-${index}`} className="col-span-2 grid grid-cols-subgrid">
                    <dt className="font-mono text-[11px] text-ink-3">{param.key}</dt>
                    <dd className="break-words font-mono text-[11px] text-ink-2">{param.value}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
          </div>
        </Disclosure>
      ) : null}
    </li>
  );
}

export function LeaderboardPanel({ experiments }: { experiments: ExperimentLog | null }) {
  if (!experiments || !experiments.results.length) {
    return (
      <Panel title="Model leaderboard" subtitle="Every candidate that was trained and scored">
        <EmptyState
          title="No experiments yet"
          hint="Candidates are trained by the executor once the feature frame and splits exist."
        />
      </Panel>
    );
  }

  const { ranked, failed, best, direction } = rankExperiments(experiments);
  const metric = experiments.primary_metric || ranked[0]?.primary_metric || "";
  const scores = ranked.map((row) => row.primary_score as number);

  return (
    <Panel
      title="Model leaderboard"
      subtitle={`Sorted by ${metricLabel(metric)} · ${direction ? "higher" : "lower"} is better`}
      aside={
        <span className="flex flex-wrap items-center justify-end gap-1.5">
          <Chip>{experiments.results.length} trained</Chip>
          {failed.length ? <Badge tone="critical">{failed.length} failed</Badge> : null}
        </span>
      }
      bodyClassName="space-y-3"
    >
      <ul className="space-y-1">
        {ranked.map((result, index) => (
          <ResultRow
            key={result.experiment_id}
            result={result}
            rank={index + 1}
            isBest={best?.experiment_id === result.experiment_id}
            fraction={relativeBarFraction(result.primary_score as number, scores, direction)}
            metric={result.primary_metric || metric}
          />
        ))}
      </ul>

      {experiments.leaderboard_notes ? (
        <p className="prose-agent text-xs leading-relaxed text-ink-2">
          {experiments.leaderboard_notes}
        </p>
      ) : null}

      {failed.length ? (
        <Disclosure summary="Families that did not produce a score" count={failed.length}>
          <ul className="space-y-1.5">
            {failed.map((result) => (
              <li key={result.experiment_id} className="border-l-2 border-critical pl-3">
                <p className="text-xs font-medium text-ink">{familyLabel(result.family)}</p>
                <p className="prose-agent mt-0.5 break-words text-[11px] leading-relaxed text-ink-2">
                  {result.error ?? "No score was recorded for this candidate."}
                </p>
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : null}
    </Panel>
  );
}
