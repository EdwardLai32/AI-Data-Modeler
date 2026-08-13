"use client";

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Badge, Chip } from "@/components/ui/Badge";
import { KeyValue, Panel } from "@/components/ui/Panel";
import { BulletList } from "@/components/ui/Rationale";
import { compactNumber, humanise, metricValue } from "@/lib/format";
import type { FinalReport } from "@/types/api";

/**
 * The report, rendered from `FinalReport` rather than from a file.
 *
 * Rendering the structured object keeps the on-screen report in sync with the
 * API even when the markdown/HTML artifacts were not written (a run can disable
 * a format), and it lets sections keep real heading semantics for screen
 * readers.
 */
export function ReportView({ report }: { report: FinalReport }) {
  const sections = [...report.sections].sort((a, b) => a.order - b.order);
  const deployment = report.deployment;

  return (
    <article className="space-y-5">
      <header className="rounded-xl border border-hairline bg-surface px-5 py-5">
        <h1 className="text-xl font-semibold tracking-tight text-ink">{report.title}</h1>
        {report.subtitle ? (
          <p className="mt-1 text-sm text-ink-2">{report.subtitle}</p>
        ) : null}
        <div className="markdown-body mt-4 border-t border-hairline pt-4">
          <Markdown remarkPlugins={[remarkGfm]}>{report.executive_summary}</Markdown>
        </div>
      </header>

      {sections.map((section) => (
        <section
          key={`${section.order}-${section.heading}`}
          className="rounded-xl border border-hairline bg-surface px-5 py-4"
        >
          <h2 className="text-sm font-semibold tracking-tight text-ink">
            <span className="tabular mr-2 text-ink-3">{section.order}</span>
            {section.heading}
          </h2>
          <div className="markdown-body mt-3">
            <Markdown remarkPlugins={[remarkGfm]}>{section.body_markdown}</Markdown>
          </div>
          {section.chart_refs.length ? (
            <p className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-hairline pt-3">
              <span className="text-[11px] uppercase tracking-wide text-ink-3">Charts</span>
              {section.chart_refs.map((ref) => (
                <Chip key={ref} mono>
                  {ref}
                </Chip>
              ))}
            </p>
          ) : null}
        </section>
      ))}

      <Panel
        title="Deployment recommendation"
        aside={<Badge tone="accent">{humanise(deployment.pattern)}</Badge>}
        bodyClassName="space-y-3"
      >
        <p className="prose-agent text-xs leading-relaxed text-ink-2">{deployment.rationale}</p>

        <dl className="grid gap-3 sm:grid-cols-3">
          <KeyValue label="Estimated latency">
            {deployment.estimated_latency_ms !== null
              ? `${metricValue(deployment.estimated_latency_ms)} ms`
              : "—"}
          </KeyValue>
          <KeyValue label="Estimated throughput">
            {deployment.estimated_throughput_rps !== null
              ? `${compactNumber(deployment.estimated_throughput_rps)} req/s`
              : "—"}
          </KeyValue>
          <KeyValue label="Model size">
            {deployment.model_size_mb !== null
              ? `${metricValue(deployment.model_size_mb)} MB`
              : "—"}
          </KeyValue>
        </dl>

        {deployment.infrastructure_notes ? (
          <KeyValue label="Infrastructure" wrap>
            {deployment.infrastructure_notes}
          </KeyValue>
        ) : null}

        <dl className="grid gap-3 sm:grid-cols-2">
          {deployment.retraining_cadence ? (
            <KeyValue label="Retraining cadence" wrap>
              {deployment.retraining_cadence}
            </KeyValue>
          ) : null}
          {deployment.rollout_strategy ? (
            <KeyValue label="Rollout" wrap>
              {deployment.rollout_strategy}
            </KeyValue>
          ) : null}
        </dl>

        {deployment.monitoring_plan.length ? (
          <div>
            <h3 className="text-xs font-semibold text-ink">Monitoring plan</h3>
            <BulletList items={deployment.monitoring_plan} className="mt-1.5" />
          </div>
        ) : null}

        {deployment.risks.length ? (
          <div>
            <h3 className="text-xs font-semibold text-ink">Deployment risks</h3>
            <BulletList items={deployment.risks} className="mt-1.5" />
          </div>
        ) : null}
      </Panel>

      {report.appendix_notes.length ? (
        <Panel title="Appendix">
          <BulletList items={report.appendix_notes} />
        </Panel>
      ) : null}
    </article>
  );
}

/** Deterministic markdown assembled from the structured report. */
export function reportToMarkdown(report: FinalReport): string {
  const lines: string[] = [`# ${report.title}`];
  if (report.subtitle) lines.push(`*${report.subtitle}*`);
  lines.push("", "## Executive summary", "", report.executive_summary);
  for (const section of [...report.sections].sort((a, b) => a.order - b.order)) {
    lines.push("", `## ${section.heading}`, "", section.body_markdown);
    if (section.chart_refs.length) {
      lines.push("", `Charts: ${section.chart_refs.join(", ")}`);
    }
  }
  const deployment = report.deployment;
  lines.push("", "## Deployment recommendation", "", `Pattern: ${deployment.pattern}`, "", deployment.rationale);
  if (deployment.monitoring_plan.length) {
    lines.push("", "### Monitoring plan", "");
    for (const item of deployment.monitoring_plan) lines.push(`- ${item}`);
  }
  if (report.appendix_notes.length) {
    lines.push("", "## Appendix", "");
    for (const note of report.appendix_notes) lines.push(`- ${note}`);
  }
  return lines.join("\n");
}
