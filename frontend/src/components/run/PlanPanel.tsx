"use client";

import { Chip } from "@/components/ui/Badge";
import { Panel } from "@/components/ui/Panel";
import { BulletList, Disclosure, Rationale } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import type { ExecutionPlan } from "@/types/api";

/**
 * The plan's strategy.
 *
 * Per-step rationale lives on the Pipeline stepper next to each step's live
 * status, where it is read in context; this panel carries the whole-plan
 * argument — what was adapted for this dataset, what could go wrong, and what
 * happens if evaluation rejects the result.
 */
export function PlanPanel({
  plan,
  history,
}: {
  plan: ExecutionPlan | null;
  history: ExecutionPlan[];
}) {
  if (!plan) {
    return (
      <Panel title="Execution plan" subtitle="Written by the Planning agent for this dataset">
        <EmptyState
          title="Not planned yet"
          hint="The Planning agent writes the strategy after the dataset understanding and problem definition are in."
        />
      </Panel>
    );
  }

  const superseded = history.filter((revision) => revision.revision !== plan.revision);

  return (
    <Panel
      title="Execution plan"
      subtitle={`${plan.steps.length} steps · revision ${plan.revision}`}
      aside={superseded.length ? <Chip>{superseded.length} superseded</Chip> : null}
      bodyClassName="space-y-3"
    >
      <p className="prose-agent text-xs leading-relaxed text-ink-2">{plan.summary}</p>

      {plan.revision_reason ? (
        <Rationale label={`Why revision ${plan.revision}`}>{plan.revision_reason}</Rationale>
      ) : null}

      {plan.dataset_specific_adaptations.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">
            How this differs from a generic pipeline
          </h3>
          <BulletList items={plan.dataset_specific_adaptations} className="mt-1.5" />
        </div>
      ) : null}

      {plan.risks.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Plan risks</h3>
          <BulletList items={plan.risks} className="mt-1.5" />
        </div>
      ) : null}

      {plan.fallback_strategy ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">If evaluation rejects the model</h3>
          <p className="prose-agent mt-1 text-xs leading-relaxed text-ink-2">
            {plan.fallback_strategy}
          </p>
        </div>
      ) : null}

      {superseded.length ? (
        <Disclosure summary="Superseded plan revisions" count={superseded.length}>
          <ul className="space-y-2">
            {superseded.map((revision, index) => (
              <li key={`${revision.revision}-${index}`} className="border-l-2 border-baseline pl-3">
                <p className="text-[11px] font-semibold text-ink">
                  Revision {revision.revision} · {revision.steps.length} steps
                </p>
                <p className="prose-agent mt-0.5 text-xs leading-relaxed text-ink-2">
                  {revision.summary}
                </p>
                {revision.revision_reason ? (
                  <p className="mt-0.5 text-[11px] text-ink-3">
                    Replaced because: {revision.revision_reason}
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : null}
    </Panel>
  );
}
