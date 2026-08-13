"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { Panel, KeyValue } from "@/components/ui/Panel";
import { BulletList, Disclosure, Rationale } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { humanise } from "@/lib/format";
import type { Tone } from "@/lib/labels";
import type { DatasetUnderstanding } from "@/types/api";

/** Readiness is a judgement, so it gets a status hue plus the word itself. */
const READINESS_TONE: Record<DatasetUnderstanding["data_readiness"], Tone> = {
  ready: "good",
  needs_cleaning: "warning",
  needs_major_work: "serious",
  unusable: "critical",
};

const POTENTIAL_TONE: Record<string, Tone> = {
  high: "good",
  medium: "accent",
  low: "warning",
  none: "neutral",
};

export function UnderstandingPanel({
  understanding,
}: {
  understanding: DatasetUnderstanding | null;
}) {
  if (!understanding) {
    return (
      <Panel title="Dataset understanding" subtitle="The Dataset agent's read on the profile">
        <EmptyState
          title="Not written yet"
          hint="The Dataset agent narrates the profile once the statistics are in."
        />
      </Panel>
    );
  }

  return (
    <Panel
      title="Dataset understanding"
      subtitle="The Dataset agent's read on the measured profile"
      aside={
        <Badge tone={READINESS_TONE[understanding.data_readiness] ?? "neutral"}>
          {humanise(understanding.data_readiness)}
        </Badge>
      }
      bodyClassName="space-y-3"
    >
      <p className="prose-agent text-sm font-medium leading-relaxed text-ink">
        {understanding.headline}
      </p>

      <dl className="grid gap-3 sm:grid-cols-2">
        <KeyValue label="Likely domain" wrap>{understanding.likely_domain}</KeyValue>
        <KeyValue label="One row is" wrap>{understanding.grain}</KeyValue>
      </dl>

      <Rationale label="Readiness">{understanding.readiness_rationale}</Rationale>

      {understanding.suggested_target_columns.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Suggested targets, best first</h3>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {understanding.suggested_target_columns.map((column, index) => (
              <Chip key={column} mono title={index === 0 ? "Top candidate" : undefined}>
                {index + 1}. {column}
              </Chip>
            ))}
          </div>
        </div>
      ) : null}

      {understanding.key_findings.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Key findings</h3>
          <BulletList items={understanding.key_findings} className="mt-1.5" />
        </div>
      ) : null}

      {understanding.risks.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Risks</h3>
          <BulletList items={understanding.risks} className="mt-1.5" />
        </div>
      ) : null}

      {understanding.narrative ? (
        <Disclosure summary="Full exploratory narrative">
          <p className="prose-agent whitespace-pre-wrap text-xs leading-relaxed text-ink-2">
            {understanding.narrative}
          </p>
        </Disclosure>
      ) : null}

      {understanding.column_assessments.length ? (
        <Disclosure
          summary="Per-column assessment"
          count={understanding.column_assessments.length}
        >
          <ul className="space-y-2">
            {understanding.column_assessments.map((assessment) => (
              <li key={assessment.name} className="border-l-2 border-baseline pl-3">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="font-mono text-[11px] text-ink">{assessment.name}</span>
                  <Chip>{humanise(assessment.role)}</Chip>
                  <Badge tone={POTENTIAL_TONE[assessment.predictive_potential] ?? "neutral"}>
                    {assessment.predictive_potential} potential
                  </Badge>
                </div>
                <p className="prose-agent mt-1 text-xs leading-relaxed text-ink-2">
                  {assessment.notes}
                </p>
                {assessment.concerns.length ? (
                  <p className="mt-0.5 text-[11px] text-ink-3">
                    Concerns: {assessment.concerns.join("; ")}
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
