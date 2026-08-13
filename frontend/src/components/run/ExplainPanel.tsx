"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { Panel } from "@/components/ui/Panel";
import { DataBar } from "@/components/ui/Metrics";
import { BulletList, Disclosure } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { percent } from "@/lib/format";
import type { ExplainabilityReport, FeatureAttribution } from "@/types/api";

/**
 * Feature attribution.
 *
 * Importances arrive normalised to sum to 1, so they are shown as shares of
 * total importance rather than raw SHAP magnitudes — "31% of the signal" is a
 * sentence a business reader can use, and it is exactly what the number means.
 * Direction is a word next to the bar, never a colour on its own.
 */

const DIRECTION_LABEL: Record<FeatureAttribution["direction"], string> = {
  increases: "↑ raises prediction",
  decreases: "↓ lowers prediction",
  mixed: "mixed effect",
  unknown: "",
};

function AttributionList({ items }: { items: FeatureAttribution[] }) {
  const max = items.reduce((best, item) => Math.max(best, Math.abs(item.importance)), 0);
  return (
    <div>
      {items.map((item, index) => (
        <DataBar
          key={`${item.feature}-${index}`}
          name={<span className="truncate font-mono text-[11px]">{item.feature}</span>}
          nameTitle={`${item.feature} (${item.method})`}
          fraction={max ? Math.abs(item.importance) / max : 0}
          valueText={percent(item.importance)}
          aside={
            DIRECTION_LABEL[item.direction] ? (
              <span className="text-[11px] text-ink-3">{DIRECTION_LABEL[item.direction]}</span>
            ) : null
          }
        />
      ))}
    </div>
  );
}

export function ExplainPanel({ report }: { report: ExplainabilityReport | null }) {
  if (!report) {
    return (
      <Panel title="Feature importance" subtitle="What the winning model actually keys on">
        <EmptyState
          title="Not computed yet"
          hint="Attribution runs against the fitted model after the leaderboard settles."
        />
      </Panel>
    );
  }

  const global = report.global_attributions;
  const permutation = report.permutation_importance;
  const method = global[0]?.method ?? permutation[0]?.method ?? "";

  return (
    <Panel
      title="Feature importance"
      subtitle={
        global.length
          ? `${global.length} features attributed${method ? ` via ${method}` : ""}`
          : "What the winning model keys on"
      }
      aside={
        <Badge tone={report.shap_available ? "good" : "neutral"}>
          {report.shap_available ? "SHAP available" : "SHAP unavailable"}
        </Badge>
      }
      bodyClassName="space-y-3"
    >
      {global.length ? (
        <AttributionList items={global.slice(0, 15)} />
      ) : permutation.length ? (
        <AttributionList items={permutation.slice(0, 15)} />
      ) : (
        <p className="text-xs text-ink-3">
          No attributions were produced for this model.
          {report.method_notes ? ` ${report.method_notes}` : ""}
        </p>
      )}

      {report.plain_language_explanations.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">In plain language</h3>
          <BulletList items={report.plain_language_explanations} className="mt-1.5" />
        </div>
      ) : null}

      {report.narrative ? (
        <Disclosure summary="Explainability narrative">
          <p className="prose-agent whitespace-pre-wrap text-xs leading-relaxed text-ink-2">
            {report.narrative}
          </p>
        </Disclosure>
      ) : null}

      {global.length && permutation.length ? (
        <Disclosure summary="Permutation importance" count={permutation.length}>
          <AttributionList items={permutation.slice(0, 15)} />
        </Disclosure>
      ) : null}

      {report.counterfactuals.length ? (
        <Disclosure summary="Counterfactuals" count={report.counterfactuals.length}>
          <ul className="space-y-2">
            {report.counterfactuals.map((item, index) => (
              <li key={index} className="border-l-2 border-baseline pl-3">
                <p className="text-xs leading-relaxed text-ink-2">{item.description}</p>
                <p className="tabular mt-0.5 text-[11px] text-ink-3">
                  {item.original_prediction} → {item.new_prediction}
                </p>
                {item.changed_features.length ? (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {item.changed_features.map((param, paramIndex) => (
                      <Chip key={`${param.key}-${paramIndex}`} mono>
                        {param.key}={param.value}
                      </Chip>
                    ))}
                  </div>
                ) : null}
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : null}

      {report.method_notes ? (
        <p className="text-[11px] leading-relaxed text-ink-3">{report.method_notes}</p>
      ) : null}
    </Panel>
  );
}
