"use client";

import { Panel } from "@/components/ui/Panel";
import { Meter, StatTile } from "@/components/ui/Metrics";
import { compactNumber, duration, elapsedSeconds, integer, percent, usd } from "@/lib/format";
import { useNow } from "@/hooks/useNow";
import type { Tone } from "@/lib/labels";
import type { RunSummary } from "@/types/api";

/**
 * Token and cost meter.
 *
 * Every figure is a measured total from `RunSummary.usage`, which the LLM client
 * accumulates from real API responses — nothing here is estimated from message
 * lengths. The two bars are the only genuinely bounded ratios available: elapsed
 * against the run's own time budget, and cache reads against total prompt
 * tokens (the run's caching effectiveness, which is what actually drives the
 * bill down on a long multi-agent run).
 */

function budgetTone(fraction: number): Tone {
  if (fraction >= 1) return "critical";
  if (fraction >= 0.8) return "warning";
  return "accent";
}

export function UsageMeter({
  summary,
  active,
}: {
  summary: RunSummary;
  active: boolean;
}) {
  const now = useNow(1000, active);
  const usage = summary.usage;

  const elapsed =
    summary.duration_seconds > 0
      ? summary.duration_seconds
      : (elapsedSeconds(summary.started_at, summary.finished_at, now) ?? 0);
  const budget = summary.config.time_budget_seconds;
  const budgetFraction = budget > 0 ? elapsed / budget : null;

  const promptTokens = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens;
  const cacheFraction = promptTokens > 0 ? usage.cache_read_tokens / promptTokens : null;

  return (
    <Panel
      title="Tokens & cost"
      subtitle="Provider-reported tokens; cost estimates use configured model prices"
      bodyClassName="space-y-4"
    >
      <div className="grid grid-cols-2 gap-2.5">
        <StatTile
          label="Estimated run cost"
          value={usage.pricing_configured ? usd(usage.cost_usd) : "Prices not configured"}
          note={`${integer(usage.llm_calls)} LLM call${usage.llm_calls === 1 ? "" : "s"}`}
        />
        <StatTile
          label="Output tokens"
          value={compactNumber(usage.output_tokens)}
          note={`${integer(usage.output_tokens)} exact`}
        />
        <StatTile
          label="Prompt tokens"
          value={compactNumber(promptTokens)}
          note={`${integer(usage.input_tokens)} fresh · ${integer(usage.cache_read_tokens)} cached`}
        />
        <StatTile
          label="Cache writes"
          value={compactNumber(usage.cache_write_tokens)}
          note="Provider-specific cache billing"
        />
      </div>

      <Meter
        label="Time budget used"
        fraction={budgetFraction}
        valueText={`${duration(elapsed)} / ${duration(budget)}`}
        tone={budgetFraction === null ? "accent" : budgetTone(budgetFraction)}
        hint={
          budgetFraction !== null && budgetFraction >= 1
            ? "Deadline reached. No new stages start; active training is stopped."
            : undefined
        }
      />

      <Meter
        label="Prompt cache hit rate"
        fraction={cacheFraction}
        valueText={cacheFraction === null ? "no prompt tokens yet" : percent(cacheFraction)}
        hint={
          cacheFraction === null
            ? "Appears once the first agent call returns."
            : "Cache billing depends on your provider and selected model."
        }
      />
    </Panel>
  );
}
