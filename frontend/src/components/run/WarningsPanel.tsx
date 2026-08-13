"use client";

import { Badge } from "@/components/ui/Badge";
import { Panel } from "@/components/ui/Panel";
import { BulletList } from "@/components/ui/Rationale";
import type { IngestionResult } from "@/types/api";

/**
 * Degraded-capability notices.
 *
 * The orchestrator's rule is that a missing optional dependency or a failed
 * model family warns and continues. Those warnings are the only record that
 * something ran with reduced capability, so they get their own panel instead of
 * scrolling past in the log.
 */
export function WarningsPanel({
  warnings,
  ingestion,
}: {
  warnings: string[];
  ingestion: IngestionResult | null;
}) {
  const validationWarnings = ingestion?.validation_warnings ?? [];
  const validationErrors = ingestion?.validation_errors ?? [];
  const fromIngestion = validationWarnings.length + validationErrors.length;
  if (!warnings.length && !fromIngestion) return null;

  return (
    <Panel
      title="Warnings"
      subtitle="Where the run continued with reduced capability"
      aside={
        // Counted separately on purpose: the run header's tile reports
        // `RunSummary.warnings` only, and a single merged total would look like
        // the two disagreed.
        <Badge tone={validationErrors.length ? "critical" : "serious"}>
          {warnings.length} run
          {fromIngestion ? ` · ${fromIngestion} ingestion` : ""}
        </Badge>
      }
      bodyClassName="space-y-3"
    >
      {validationErrors.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Ingestion errors</h3>
          <BulletList items={validationErrors} className="mt-1.5" />
        </div>
      ) : null}

      {validationWarnings.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Ingestion warnings</h3>
          <BulletList items={validationWarnings} className="mt-1.5" />
        </div>
      ) : null}

      {warnings.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Run warnings</h3>
          <BulletList items={warnings} className="mt-1.5" />
        </div>
      ) : null}
    </Panel>
  );
}
