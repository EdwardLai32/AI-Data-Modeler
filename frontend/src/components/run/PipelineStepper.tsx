"use client";

import { Badge, AgentTag, Chip } from "@/components/ui/Badge";
import { Panel } from "@/components/ui/Panel";
import { EmptyState } from "@/components/ui/States";
import { Rationale } from "@/components/ui/Rationale";
import { duration, elapsedSeconds } from "@/lib/format";
import { agentShort, stepStatusLabel, stepStatusTone, TONE_DOT } from "@/lib/labels";
import { useNow } from "@/hooks/useNow";
import type { ExecutionPlan, PlanStep, StepRecord, StepStatus } from "@/types/api";

/**
 * The plan, with live status per step.
 *
 * Two sources are merged: the plan the Planning agent wrote (which carries the
 * objective and the per-step rationale) and the orchestrator's step records
 * (which carry status and timing). Records with no matching plan step are still
 * shown — pre-plan work like ingestion and profiling happens before a plan
 * exists, and dropping it would make the pipeline look like it started late.
 */

interface Row {
  stepId: string;
  order: number | null;
  title: string;
  agent: string;
  status: StepStatus;
  planStep: PlanStep | null;
  record: StepRecord | null;
}

function mergeRows(plan: ExecutionPlan | null, records: StepRecord[]): Row[] {
  const byId = new Map(records.map((record) => [record.step_id, record]));
  const rows: Row[] = [];
  const planned = new Set<string>();

  const orderedPlan = [...(plan?.steps ?? [])].sort((a, b) => a.order - b.order);
  for (const step of orderedPlan) {
    planned.add(step.step_id);
    const record = byId.get(step.step_id) ?? null;
    rows.push({
      stepId: step.step_id,
      order: step.order,
      title: step.title || step.step_id,
      agent: agentShort(step.agent),
      status: record?.status ?? "pending",
      planStep: step,
      record,
    });
  }

  const extras = records.filter((record) => !planned.has(record.step_id));
  for (const record of extras) {
    rows.push({
      stepId: record.step_id,
      order: null,
      title: record.title || record.step_id,
      agent: agentShort(record.agent),
      status: record.status,
      planStep: null,
      record,
    });
  }

  // Unplanned records ran before the plan existed, so they belong at the top.
  return [...rows.filter((row) => row.order === null), ...rows.filter((row) => row.order !== null)];
}

function rowElapsed(row: Row, now: number): number | null {
  const record = row.record;
  if (!record) return null;
  if (record.duration_seconds > 0) return record.duration_seconds;
  if (record.status === "running" || record.status === "awaiting_approval") {
    return elapsedSeconds(record.started_at, null, now);
  }
  return elapsedSeconds(record.started_at, record.finished_at, now);
}

export function PipelineStepper({
  plan,
  records,
  active,
}: {
  plan: ExecutionPlan | null;
  records: StepRecord[];
  active: boolean;
}) {
  const now = useNow(1000, active);
  const rows = mergeRows(plan, records);

  const completed = rows.filter((row) => row.status === "completed").length;
  const total = rows.length;

  return (
    <Panel
      title="Pipeline"
      subtitle={
        plan
          ? `Plan revision ${plan.revision}${plan.revision_reason ? " · replanned" : ""}`
          : "Waiting for the Planning agent"
      }
      aside={
        total ? (
          <Chip>
            {completed}/{total} steps complete
          </Chip>
        ) : null
      }
      bodyClassName="py-3"
    >
      {!total ? (
        <EmptyState
          title="No steps yet"
          hint="The Planning agent writes the step list after the dataset has been profiled."
        />
      ) : (
        <ol className="relative">
          {rows.map((row, index) => {
            const tone = stepStatusTone(row.status);
            const elapsed = rowElapsed(row, now);
            const isLast = index === rows.length - 1;
            return (
              <li key={`${row.stepId}-${index}`} className="relative flex gap-3 pb-3">
                {/* rail */}
                {!isLast ? (
                  <span
                    aria-hidden="true"
                    className="absolute left-[7px] top-4 h-full w-px bg-hairline-strong"
                  />
                ) : null}
                <span
                  aria-hidden="true"
                  className={`relative z-10 mt-1 size-3.5 shrink-0 rounded-full border-2 border-surface ${TONE_DOT[tone]} ${
                    row.status === "running" ? "animate-pulse" : ""
                  } ${row.status === "pending" ? "opacity-50" : ""}`}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span className="text-sm font-medium text-ink">
                      {row.order !== null ? (
                        <span className="tabular mr-1.5 text-ink-3">{row.order}.</span>
                      ) : null}
                      {row.title}
                    </span>
                    <Badge tone={tone} pulse={row.status === "running"}>
                      {stepStatusLabel(row.status)}
                    </Badge>
                    <AgentTag>{row.agent}</AgentTag>
                    {row.planStep?.destructive ? <Badge tone="warning">Destructive</Badge> : null}
                    {row.planStep?.optional ? <Chip>Optional</Chip> : null}
                    {row.record && row.record.attempts > 1 ? (
                      <Chip>{row.record.attempts} attempts</Chip>
                    ) : null}
                    <span className="tabular ml-auto shrink-0 text-[11px] text-ink-3">
                      {elapsed !== null
                        ? duration(elapsed)
                        : row.planStep
                          ? `est. ${duration(row.planStep.estimated_seconds)}`
                          : ""}
                    </span>
                  </div>

                  {row.planStep?.objective ? (
                    <p className="prose-agent mt-1 text-xs leading-relaxed text-ink-2">
                      {row.planStep.objective}
                    </p>
                  ) : null}

                  {row.record?.summary ? (
                    <p className="prose-agent mt-1 text-xs leading-relaxed text-ink-3">
                      {row.record.summary}
                    </p>
                  ) : null}

                  {row.planStep?.rationale ? (
                    <Rationale>{row.planStep.rationale}</Rationale>
                  ) : null}

                  {row.record?.error ? (
                    <p className="mt-1.5 flex gap-2 border-l-2 border-critical pl-3 text-xs leading-relaxed text-ink-2">
                      <span className="font-medium text-ink">Error:</span>
                      <span className="prose-agent break-words">{row.record.error}</span>
                    </p>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}
