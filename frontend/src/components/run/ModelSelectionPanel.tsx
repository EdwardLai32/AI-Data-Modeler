"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { Divided, KeyValue, Panel } from "@/components/ui/Panel";
import { Disclosure, Rationale } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { humanise } from "@/lib/format";
import { familyLabel } from "@/lib/labels";
import type { Tone } from "@/lib/labels";
import type { ModelSelection } from "@/types/api";

const SUITABILITY_TONE: Record<string, Tone> = {
  excellent: "good",
  good: "accent",
  fair: "warning",
  poor: "serious",
};

/** Why these model families, and why not the others. */
export function ModelSelectionPanel({ selection }: { selection: ModelSelection | null }) {
  if (!selection) {
    return (
      <Panel title="Model selection" subtitle="Candidates chosen for this dataset's shape">
        <EmptyState
          title="Not selected yet"
          hint="The Model Selection agent ranks families against the dataset's size, sparsity, and signal."
        />
      </Panel>
    );
  }

  const ordered = [...selection.candidates].sort((a, b) => a.rank - b.rank);

  return (
    <Panel
      title="Model selection"
      subtitle={`${selection.candidates.length} candidate${selection.candidates.length === 1 ? "" : "s"}`}
      bodyClassName="space-y-3"
    >
      {selection.summary ? (
        <p className="prose-agent text-xs leading-relaxed text-ink-2">{selection.summary}</p>
      ) : null}

      <dl className="grid gap-3 sm:grid-cols-2">
        <KeyValue label="Validation strategy" wrap>{selection.validation_strategy}</KeyValue>
        {selection.validation_rationale ? (
          <KeyValue label="Why that strategy" wrap>{selection.validation_rationale}</KeyValue>
        ) : null}
      </dl>

      <Divided>
        {ordered.map((candidate) => (
          <div key={`${candidate.family}-${candidate.rank}`} className="py-2.5 first:pt-0 last:pb-0">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="tabular text-ink-3">{candidate.rank}</span>
              <span className="text-xs font-semibold text-ink">
                {familyLabel(candidate.family)}
              </span>
              <Badge tone={SUITABILITY_TONE[candidate.suitability] ?? "neutral"}>
                {candidate.suitability} fit
              </Badge>
              {candidate.is_baseline ? <Chip>Baseline</Chip> : null}
              {candidate.tune_priority !== "none" ? (
                <Chip>tune: {candidate.tune_priority}</Chip>
              ) : null}
            </div>

            <Rationale>{candidate.rationale}</Rationale>

            {candidate.expected_strengths.length || candidate.expected_weaknesses.length ? (
              <div className="mt-1.5 grid gap-2 sm:grid-cols-2">
                {candidate.expected_strengths.length ? (
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-wide text-ink-3">
                      Expected strengths
                    </p>
                    <ul className="mt-0.5 space-y-0.5">
                      {candidate.expected_strengths.map((item, index) => (
                        <li key={index} className="text-[11px] leading-relaxed text-ink-2">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {candidate.expected_weaknesses.length ? (
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-wide text-ink-3">
                      Expected weaknesses
                    </p>
                    <ul className="mt-0.5 space-y-0.5">
                      {candidate.expected_weaknesses.map((item, index) => (
                        <li key={index} className="text-[11px] leading-relaxed text-ink-2">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            ) : null}

            {candidate.initial_params.length ? (
              <Disclosure summary="Initial parameters" count={candidate.initial_params.length}>
                <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5">
                  {candidate.initial_params.map((param, index) => (
                    <div key={`${param.key}-${index}`} className="col-span-2 grid grid-cols-subgrid">
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

      {selection.reasoning ? (
        <Disclosure summary="Comparative reasoning across candidates">
          <p className="prose-agent whitespace-pre-wrap text-xs leading-relaxed text-ink-2">
            {selection.reasoning}
          </p>
        </Disclosure>
      ) : null}

      {selection.excluded_families.length ? (
        <Disclosure summary="Families ruled out" count={selection.excluded_families.length}>
          <ul className="space-y-1">
            {selection.excluded_families.map((family, index) => (
              <li key={`${family}-${index}`} className="text-xs leading-relaxed text-ink-2">
                <span className="font-medium text-ink">{humanise(family)}</span>
                {selection.exclusion_rationale[index]
                  ? ` — ${selection.exclusion_rationale[index]}`
                  : ""}
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : null}
    </Panel>
  );
}
