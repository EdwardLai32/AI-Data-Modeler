"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { Panel } from "@/components/ui/Panel";
import { BulletList, Disclosure } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { humanise } from "@/lib/format";
import type { Tone } from "@/lib/labels";
import type { InsightReport } from "@/types/api";

const CONFIDENCE_TONE: Record<string, Tone> = {
  high: "good",
  medium: "warning",
  low: "serious",
};

/**
 * Business insights.
 *
 * Every insight is required by the schema to name its supporting evidence, so
 * the evidence line is rendered as prominently as the headline — an actionable
 * claim with no metric behind it is exactly what this product is meant not to
 * produce.
 */
export function InsightsPanel({ insights }: { insights: InsightReport | null }) {
  if (!insights) {
    return (
      <Panel title="Business insights" subtitle="What a decision-maker should do about this">
        <EmptyState
          title="Not written yet"
          hint="The Insight agent translates the model's behaviour into actions once evaluation passes."
        />
      </Panel>
    );
  }

  return (
    <Panel
      title="Business insights"
      subtitle={`${insights.insights.length} insight${insights.insights.length === 1 ? "" : "s"}`}
      bodyClassName="space-y-3"
    >
      {insights.executive_summary ? (
        <p className="prose-agent text-sm leading-relaxed text-ink">
          {insights.executive_summary}
        </p>
      ) : null}

      <ul className="space-y-2.5">
        {insights.insights.map((insight, index) => (
          <li
            key={`${index}-${insight.headline.slice(0, 24)}`}
            className="rounded-lg border border-hairline bg-surface-2 px-3.5 py-3"
          >
            <div className="flex flex-wrap items-start justify-between gap-2">
              <p className="prose-agent min-w-0 text-xs font-semibold leading-relaxed text-ink">
                {insight.headline}
              </p>
              <span className="flex shrink-0 flex-wrap items-center gap-1.5">
                <Badge tone={CONFIDENCE_TONE[insight.confidence] ?? "neutral"}>
                  {insight.confidence} confidence
                </Badge>
                <Chip>{humanise(insight.audience)}</Chip>
              </span>
            </div>

            <p className="prose-agent mt-1.5 text-xs leading-relaxed text-ink-2">
              {insight.detail}
            </p>

            <dl className="mt-2 space-y-1">
              <div>
                <dt className="inline text-[10px] font-semibold uppercase tracking-wide text-ink-3">
                  Evidence:{" "}
                </dt>
                <dd className="inline text-xs leading-relaxed text-ink-2">
                  {insight.supporting_evidence}
                </dd>
              </div>
              <div>
                <dt className="inline text-[10px] font-semibold uppercase tracking-wide text-ink-3">
                  Do next:{" "}
                </dt>
                <dd className="inline text-xs leading-relaxed text-ink-2">
                  {insight.recommended_action}
                </dd>
              </div>
              {insight.expected_value ? (
                <div>
                  <dt className="inline text-[10px] font-semibold uppercase tracking-wide text-ink-3">
                    Expected value:{" "}
                  </dt>
                  <dd className="inline text-xs leading-relaxed text-ink-2">
                    {insight.expected_value}
                  </dd>
                </div>
              ) : null}
            </dl>
          </li>
        ))}
      </ul>

      {insights.key_drivers_plain_language.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Key drivers, in plain language</h3>
          <BulletList items={insights.key_drivers_plain_language} className="mt-1.5" />
        </div>
      ) : null}

      {insights.caveats.length ? (
        <Disclosure summary="Caveats" count={insights.caveats.length}>
          <BulletList items={insights.caveats} />
        </Disclosure>
      ) : null}

      {insights.suggested_next_experiments.length ? (
        <Disclosure
          summary="Suggested next experiments"
          count={insights.suggested_next_experiments.length}
        >
          <BulletList items={insights.suggested_next_experiments} />
        </Disclosure>
      ) : null}
    </Panel>
  );
}
