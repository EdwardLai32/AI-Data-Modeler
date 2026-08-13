"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { Divided, KeyValue, Panel } from "@/components/ui/Panel";
import { Disclosure, NoteLine, Rationale } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { humanise, integer, signed } from "@/lib/format";
import type { Tone } from "@/lib/labels";
import type { FeaturePlan } from "@/types/api";

const PRIORITY_TONE: Record<string, Tone> = {
  high: "accent",
  medium: "neutral",
  low: "neutral",
};

/**
 * Engineered features.
 *
 * Each operation carries the mechanism it is betting on (`hypothesis`) and the
 * leakage or overfitting exposure it creates (`risk`). Both are shown, because a
 * feature that improves validation score by leaking the target is the single
 * most expensive mistake this pipeline can make.
 */
export function FeaturesPanel({
  features,
  nFeaturesIn,
}: {
  features: FeaturePlan | null;
  /**
   * Column count the trained models actually saw, from
   * `ExperimentResult.n_features_in`. The plan's delta is an expectation; this
   * is the measured outcome, so both are shown.
   */
  nFeaturesIn?: number | null;
}) {
  if (!features) {
    return (
      <Panel title="Engineered features" subtitle="Operations, hypotheses, and leakage risk">
        <EmptyState
          title="Nothing engineered yet"
          hint="The Feature agent proposes operations; the executor builds the feature frame with sklearn and pandas."
        />
      </Panel>
    );
  }

  return (
    <Panel
      title="Engineered features"
      subtitle={`${features.decisions.length} operation${features.decisions.length === 1 ? "" : "s"}`}
      aside={
        features.expected_feature_count_delta !== 0 ? (
          <Chip title="Expected change in column count">
            {signed(features.expected_feature_count_delta, 0)} columns expected
          </Chip>
        ) : null
      }
      bodyClassName="space-y-3"
    >
      {features.summary ? (
        <p className="prose-agent text-xs leading-relaxed text-ink-2">{features.summary}</p>
      ) : null}

      {features.dimensionality_strategy || features.selection_strategy ? (
        <dl className="grid gap-3 sm:grid-cols-2">
          {features.dimensionality_strategy ? (
            <KeyValue label="Dimensionality strategy" wrap>
              {features.dimensionality_strategy}
            </KeyValue>
          ) : null}
          {features.selection_strategy ? (
            <KeyValue label="Selection strategy" wrap>{features.selection_strategy}</KeyValue>
          ) : null}
        </dl>
      ) : null}

      {features.decisions.length ? (
        <Divided>
          {features.decisions.map((decision, index) => (
            <div key={`${decision.op}-${index}`} className="py-2.5 first:pt-0 last:pb-0">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs font-semibold text-ink">{humanise(decision.op)}</span>
                <Badge tone={PRIORITY_TONE[decision.priority] ?? "neutral"}>
                  {decision.priority} priority
                </Badge>
                {decision.output_name_hint ? (
                  <Chip mono title="Output name hint">
                    → {decision.output_name_hint}
                  </Chip>
                ) : null}
              </div>

              {decision.input_columns.length ? (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {decision.input_columns.slice(0, 12).map((column) => (
                    <Chip key={column} mono>
                      {column}
                    </Chip>
                  ))}
                  {decision.input_columns.length > 12 ? (
                    <Chip>+{decision.input_columns.length - 12} more</Chip>
                  ) : null}
                </div>
              ) : null}

              <Rationale>{decision.rationale}</Rationale>

              <div className="mt-1 space-y-0.5">
                {decision.hypothesis ? (
                  <NoteLine label="Mechanism">{decision.hypothesis}</NoteLine>
                ) : null}
                {decision.risk ? <NoteLine label="Risk">{decision.risk}</NoteLine> : null}
              </div>

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
          No feature operations were proposed for this dataset.
        </p>
      )}

      {typeof nFeaturesIn === "number" && nFeaturesIn > 0 ? (
        <p className="text-[11px] text-ink-3">
          Models were fitted on {integer(nFeaturesIn)} feature columns.
        </p>
      ) : null}
    </Panel>
  );
}
