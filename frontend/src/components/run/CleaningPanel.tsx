"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { Divided, Panel } from "@/components/ui/Panel";
import { BulletList, Disclosure, NoteLine, Rationale } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { humanise } from "@/lib/format";
import { severityTone } from "@/lib/labels";
import type { CleaningPlan } from "@/types/api";

/**
 * Cleaning decisions, each with the evidence that justified it.
 *
 * Destructive actions are marked, and the deliberately-skipped list is shown
 * beside the applied one — "we considered dropping this and chose not to" is as
 * much a part of the audit trail as the transformations that ran.
 */
export function CleaningPanel({ cleaning }: { cleaning: CleaningPlan | null }) {
  if (!cleaning) {
    return (
      <Panel title="Cleaning decisions" subtitle="Each transformation with its evidence">
        <EmptyState
          title="Nothing decided yet"
          hint="The Cleaning agent proposes transformations from the profile; the executor applies them with pandas."
        />
      </Panel>
    );
  }

  const destructive = cleaning.decisions.filter((decision) => decision.destructive).length;

  return (
    <Panel
      title="Cleaning decisions"
      subtitle={`${cleaning.decisions.length} transformation${cleaning.decisions.length === 1 ? "" : "s"}`}
      aside={destructive ? <Badge tone="warning">{destructive} destructive</Badge> : null}
      bodyClassName="space-y-3"
    >
      {cleaning.summary ? (
        <p className="prose-agent text-xs leading-relaxed text-ink-2">{cleaning.summary}</p>
      ) : null}

      {cleaning.columns_to_drop.length ? (
        <div className="rounded-lg border border-hairline border-l-2 border-l-warning bg-surface-2 px-3 py-2">
          <p className="text-xs font-semibold text-ink">
            Columns dropped ({cleaning.columns_to_drop.length})
          </p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {cleaning.columns_to_drop.map((column) => (
              <Chip key={column} mono>
                {column}
              </Chip>
            ))}
          </div>
          {cleaning.drop_rationale.length ? (
            <BulletList items={cleaning.drop_rationale} className="mt-2" />
          ) : null}
        </div>
      ) : null}

      {cleaning.decisions.length ? (
        <Divided>
          {cleaning.decisions.map((decision, index) => (
            <div key={`${decision.action}-${index}`} className="py-2.5 first:pt-0 last:pb-0">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs font-semibold text-ink">{humanise(decision.action)}</span>
                {decision.strategy ? <Chip>{humanise(decision.strategy)}</Chip> : null}
                {decision.destructive ? <Badge tone="warning">Destructive</Badge> : null}
                <Badge tone={severityTone(decision.severity_if_skipped)}>
                  {humanise(decision.severity_if_skipped)} if skipped
                </Badge>
              </div>

              {decision.columns.length ? (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {decision.columns.slice(0, 12).map((column) => (
                    <Chip key={column} mono>
                      {column}
                    </Chip>
                  ))}
                  {decision.columns.length > 12 ? (
                    <Chip>+{decision.columns.length - 12} more</Chip>
                  ) : null}
                </div>
              ) : (
                <p className="mt-1 text-[11px] text-ink-3">Applies to the whole table.</p>
              )}

              <Rationale>{decision.rationale}</Rationale>

              {decision.expected_impact ? (
                <div className="mt-1">
                  <NoteLine label="Expected impact">{decision.expected_impact}</NoteLine>
                </div>
              ) : null}

              {decision.parameters.length ? (
                <Disclosure summary="Parameters" count={decision.parameters.length}>
                  <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5">
                    {decision.parameters.map((param, paramIndex) => (
                      <div
                        key={`${param.key}-${paramIndex}`}
                        className="col-span-2 grid grid-cols-subgrid"
                      >
                        <dt className="font-mono text-[11px] text-ink-3">{param.key}</dt>
                        <dd className="break-words font-mono text-[11px] text-ink-2">
                          {param.value}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </Disclosure>
              ) : null}
            </div>
          ))}
        </Divided>
      ) : (
        <p className="text-xs text-ink-3">
          The agent proposed no transformations — the data was already fit to model.
        </p>
      )}

      {cleaning.skipped_considerations.length ? (
        <Disclosure
          summary="Deliberately not applied"
          count={cleaning.skipped_considerations.length}
        >
          <BulletList items={cleaning.skipped_considerations} />
        </Disclosure>
      ) : null}
    </Panel>
  );
}
