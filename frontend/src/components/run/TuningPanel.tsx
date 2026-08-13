"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { KeyValue, Panel } from "@/components/ui/Panel";
import { Disclosure, Rationale } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { duration, metricValue, signed } from "@/lib/format";
import { familyLabel, tuningMethodLabel } from "@/lib/labels";
import type { TuningDecision, TuningResult } from "@/types/api";

/**
 * Hyperparameter search: the decision, then the outcome.
 *
 * "Not worthwhile" is a real, reportable answer here — the agent is asked to
 * weigh compute against expected gain, so a run that skips tuning with a reason
 * is a success, not a gap.
 */

/** Trial-by-trial scores. One series, so no legend: the label above names it. */
function TrialSparkline({ scores, direction }: { scores: number[]; direction: boolean }) {
  const finite = scores.filter((value) => Number.isFinite(value));
  if (finite.length < 2) return null;

  const width = 240;
  const height = 44;
  const pad = 5;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;

  const x = (index: number) => pad + (index / (finite.length - 1)) * (width - pad * 2);
  const y = (value: number) => height - pad - ((value - min) / span) * (height - pad * 2);

  const points = finite.map((value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`);
  const bestValue = direction ? max : min;
  const bestIndex = finite.indexOf(bestValue);
  const lastIndex = finite.length - 1;

  return (
    <figure className="mt-1">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height={height}
        role="img"
        aria-label={`Scores across ${finite.length} trials, from ${metricValue(finite[0] ?? null)} to ${metricValue(finite[lastIndex] ?? null)}, best ${metricValue(bestValue)}`}
        className="max-w-[240px]"
      >
        <polyline
          points={points.join(" ")}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <circle
          cx={x(bestIndex)}
          cy={y(bestValue)}
          r={4}
          fill="var(--accent)"
          stroke="var(--surface)"
          strokeWidth={2}
        />
      </svg>
      <figcaption className="tabular mt-0.5 text-[11px] text-ink-3">
        {finite.length} trials · best {metricValue(bestValue)} at trial {bestIndex + 1}
      </figcaption>
    </figure>
  );
}

export function TuningPanel({
  decision,
  result,
  direction,
}: {
  decision: TuningDecision | null;
  result: TuningResult | null;
  /** True when a higher primary-metric score is better. */
  direction: boolean;
}) {
  if (!decision && !result) {
    return (
      <Panel title="Hyperparameter tuning" subtitle="Cost/benefit decision, then the search">
        <EmptyState
          title="Not considered yet"
          hint="The Tuning agent weighs search cost against the expected gain after the first experiments land."
        />
      </Panel>
    );
  }

  const ran = result?.ran === true;

  return (
    <Panel
      title="Hyperparameter tuning"
      subtitle={
        decision
          ? `${tuningMethodLabel(decision.method)}${decision.target_family ? ` on ${familyLabel(decision.target_family)}` : ""}`
          : undefined
      }
      aside={
        ran ? (
          <Badge tone="good">Search ran</Badge>
        ) : decision && !decision.worthwhile ? (
          <Badge tone="neutral">Skipped by design</Badge>
        ) : (
          <Badge tone="neutral">Not run</Badge>
        )
      }
      bodyClassName="space-y-3"
    >
      {decision ? (
        <>
          <Rationale label={decision.worthwhile ? "Why tune" : "Why not tune"}>
            {decision.rationale}
          </Rationale>
          {decision.method_rationale ? (
            <Rationale label="Why this method">{decision.method_rationale}</Rationale>
          ) : null}
          <dl className="grid gap-3 sm:grid-cols-3">
            <KeyValue label="Trial budget">{decision.n_trials}</KeyValue>
            <KeyValue label="Timeout">{duration(decision.timeout_seconds)}</KeyValue>
            <KeyValue label="Early stopping">{decision.early_stopping ? "yes" : "no"}</KeyValue>
          </dl>
          {decision.expected_gain ? (
            <p className="text-xs leading-relaxed text-ink-2">
              <span className="text-ink-3">Expected gain: </span>
              {decision.expected_gain}
            </p>
          ) : null}
        </>
      ) : null}

      {result ? (
        <div className="rounded-lg border border-hairline bg-surface-2 px-3.5 py-3">
          {ran ? (
            <>
              <dl className="grid gap-3 sm:grid-cols-4">
                <KeyValue label="Best score">{metricValue(result.best_score)}</KeyValue>
                <KeyValue label="Baseline">{metricValue(result.baseline_score)}</KeyValue>
                <KeyValue label="Improvement">{signed(result.improvement)}</KeyValue>
                <KeyValue label="Trials / time">
                  {result.n_trials_completed} · {duration(result.seconds)}
                </KeyValue>
              </dl>
              <TrialSparkline scores={result.trial_scores} direction={direction} />
              {result.best_params.length ? (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {result.best_params.map((param, index) => (
                    <Chip key={`${param.key}-${index}`} mono>
                      {param.key}={param.value}
                    </Chip>
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <p className="text-xs leading-relaxed text-ink-2">
              {result.skipped_reason ??
                result.error ??
                "No search was executed for this run."}
            </p>
          )}
          {result.error && ran ? (
            <p className="mt-2 border-l-2 border-critical pl-3 text-xs leading-relaxed text-ink-2">
              {result.error}
            </p>
          ) : null}
        </div>
      ) : null}

      {decision?.search_space.length ? (
        <Disclosure summary="Search space" count={decision.search_space.length}>
          <ul className="space-y-1.5">
            {decision.search_space.map((entry) => (
              <li key={entry.name} className="text-[11px] leading-relaxed text-ink-2">
                <span className="font-mono text-ink">{entry.name}</span>{" "}
                <Chip>{entry.kind}</Chip>{" "}
                <span className="tabular">
                  {entry.choices.length
                    ? entry.choices.join(" | ")
                    : `${metricValue(entry.low)} → ${metricValue(entry.high)}`}
                </span>
                {entry.rationale ? (
                  <span className="block text-ink-3">{entry.rationale}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : null}
    </Panel>
  );
}
