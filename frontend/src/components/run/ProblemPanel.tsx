"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { KeyValue, Panel } from "@/components/ui/Panel";
import { BulletList, Rationale } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { EMPTY, metricLabel } from "@/lib/format";
import { taskLabel } from "@/lib/labels";
import type { ProblemDefinition } from "@/types/api";
import type { Tone } from "@/lib/labels";

const CONFIDENCE_TONE: Record<ProblemDefinition["confidence"], Tone> = {
  high: "good",
  medium: "warning",
  low: "serious",
};

export function ProblemPanel({ problem }: { problem: ProblemDefinition | null }) {
  if (!problem) {
    return (
      <Panel title="Problem definition" subtitle="What is being predicted, and how it is scored">
        <EmptyState
          title="Not defined yet"
          hint="The Problem agent picks the task type and the metric from the profile and the target's shape."
        />
      </Panel>
    );
  }

  return (
    <Panel
      title="Problem definition"
      subtitle="What is being predicted, and how success is measured"
      aside={
        <Badge tone={CONFIDENCE_TONE[problem.confidence] ?? "neutral"}>
          {problem.confidence} confidence
        </Badge>
      }
      bodyClassName="space-y-3"
    >
      <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <KeyValue label="Task">{taskLabel(problem.task_type)}</KeyValue>
        <KeyValue label="Target" mono>
          {problem.target_column ?? "none (unsupervised)"}
        </KeyValue>
        <KeyValue label="Primary metric">{metricLabel(problem.primary_metric)}</KeyValue>
        <KeyValue label="Positive class" mono>
          {problem.positive_class ?? EMPTY}
        </KeyValue>
      </dl>

      {problem.secondary_metrics.length ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] uppercase tracking-wide text-ink-3">Also tracking</span>
          {problem.secondary_metrics.map((metric) => (
            <Chip key={metric}>{metricLabel(metric)}</Chip>
          ))}
        </div>
      ) : null}

      {problem.temporal_column || problem.group_column || problem.horizon !== null ? (
        <dl className="grid gap-3 sm:grid-cols-3">
          {problem.temporal_column ? (
            <KeyValue label="Temporal column" mono>
              {problem.temporal_column}
            </KeyValue>
          ) : null}
          {problem.group_column ? (
            <KeyValue label="Group key" mono>
              {problem.group_column}
            </KeyValue>
          ) : null}
          {problem.horizon !== null ? (
            <KeyValue label="Forecast horizon">{problem.horizon} periods</KeyValue>
          ) : null}
        </dl>
      ) : null}

      <Rationale label="Why this task">{problem.rationale}</Rationale>
      <Rationale label="Why this metric">{problem.metric_rationale}</Rationale>

      <div>
        <h3 className="text-xs font-semibold text-ink">Business objective</h3>
        <p className="prose-agent mt-1 text-xs leading-relaxed text-ink-2">
          {problem.business_objective}
        </p>
      </div>

      {problem.alternatives_considered.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Alternatives considered</h3>
          <BulletList items={problem.alternatives_considered} className="mt-1.5" />
        </div>
      ) : null}

      {problem.constraints.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Constraints</h3>
          <BulletList items={problem.constraints} className="mt-1.5" />
        </div>
      ) : null}
    </Panel>
  );
}
