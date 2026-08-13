"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

import { ApprovalHistory, ApprovalModal } from "@/components/run/ApprovalModal";
import { AskPanel } from "@/components/run/AskPanel";
import { ChartsPanel } from "@/components/run/ChartsPanel";
import { CleaningPanel } from "@/components/run/CleaningPanel";
import { EvaluationPanel } from "@/components/run/EvaluationPanel";
import { EventFeed } from "@/components/run/EventFeed";
import { ExplainPanel } from "@/components/run/ExplainPanel";
import { FeaturesPanel } from "@/components/run/FeaturesPanel";
import { InsightsPanel } from "@/components/run/InsightsPanel";
import { LeaderboardPanel } from "@/components/run/LeaderboardPanel";
import { ModelSelectionPanel } from "@/components/run/ModelSelectionPanel";
import { PipelineStepper } from "@/components/run/PipelineStepper";
import { PlanPanel } from "@/components/run/PlanPanel";
import { ProblemPanel } from "@/components/run/ProblemPanel";
import { ProfilePanel } from "@/components/run/ProfilePanel";
import { RunHeader } from "@/components/run/RunHeader";
import { TuningPanel } from "@/components/run/TuningPanel";
import { UnderstandingPanel } from "@/components/run/UnderstandingPanel";
import { UsageMeter } from "@/components/run/UsageMeter";
import { WarningsPanel } from "@/components/run/WarningsPanel";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { ApiFailure, EmptyState, PanelSkeleton, Skeleton } from "@/components/ui/States";
import { useRunStream } from "@/hooks/useRunStream";
import { useRunSummary } from "@/hooks/useRunSummary";
import { isRunActive } from "@/lib/labels";
import { rankExperiments } from "@/lib/metrics";
import type { EventKind } from "@/types/api";

/**
 * The live run view.
 *
 * Lives here rather than in the route file on purpose: Tailwind's source scan
 * does not reach `app/runs/[runId]/**` (the bracketed directory name of a
 * dynamic route reads as a glob character class), so any utility class written
 * inside a route folder is silently dropped from the stylesheet. Route files in
 * this app stay class-free shells and every styled element lives under
 * `src/components`.
 *
 * Two data paths, deliberately kept separate: the event stream is the narration
 * (append-only, ordered, never lost) and the run summary is the state (whole
 * objects, refetched). Events trigger a coalesced summary refresh, which is what
 * makes panels fill in as work completes without polling the whole blackboard
 * every second.
 */

/** Event kinds that mean some panel's underlying object has changed. */
const STATE_CHANGING: ReadonlySet<EventKind> = new Set<EventKind>([
  "run_started",
  "run_completed",
  "run_failed",
  "run_cancelled",
  "step_started",
  "step_completed",
  "step_failed",
  "step_skipped",
  "step_retried",
  "agent_decision",
  "artifact_written",
  "approval_requested",
  "approval_resolved",
  "replan_triggered",
  "metric_recorded",
]);

export function RunView() {
  const params = useParams<{ runId: string | string[] }>();
  const raw = params?.runId;
  const runId = Array.isArray(raw) ? (raw[0] ?? "") : (raw ?? "");

  const { summary, error, loading, refreshing, requestRefresh, refreshNow } = useRunSummary(runId);
  const [dismissedApproval, setDismissedApproval] = useState<string | null>(null);

  const onEvent = useCallback(
    (event: { kind: EventKind }) => {
      if (STATE_CHANGING.has(event.kind)) requestRefresh();
    },
    [requestRefresh],
  );

  const stream = useRunStream(runId, { enabled: Boolean(runId), onEvent });

  // A terminal event lands before the final summary is written; pick it up.
  useEffect(() => {
    if (stream.finished) refreshNow();
  }, [stream.finished, refreshNow]);

  const active = summary ? isRunActive(summary.status) : true;
  const pendingApproval = useMemo(
    () => summary?.approvals.find((approval) => approval.decision === "pending") ?? null,
    [summary],
  );
  const direction = useMemo(
    () => rankExperiments(summary?.experiments ?? null).direction,
    [summary?.experiments],
  );
  const bestFeatureCount = useMemo(() => {
    const best = rankExperiments(summary?.experiments ?? null).best;
    return best?.n_features_in ?? null;
  }, [summary?.experiments]);

  if (!runId) {
    return (
      <EmptyState title="No run id in the URL" hint="Open a run from the dashboard.">
        <Link
          href="/"
          className="rounded-lg border border-hairline bg-surface-2 px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface"
        >
          Back to runs
        </Link>
      </EmptyState>
    );
  }

  if (loading && !summary) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-6 w-64" />
        <Skeleton className="h-40 w-full" />
        <div className="grid gap-4 xl:grid-cols-[minmax(360px,420px)_minmax(0,1fr)]">
          <Panel title="Pipeline">
            <PanelSkeleton rows={6} />
          </Panel>
          <Panel title="Loading run">
            <PanelSkeleton rows={8} />
          </Panel>
        </div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="space-y-4">
        <Link href="/" className="text-xs text-ink-3 hover:text-ink">
          ← All runs
        </Link>
        {error ? (
          error.status === 404 ? (
            <EmptyState
              title={`No run with id ${runId}`}
              hint="It may have been started in a different workspace, or the database was reset."
            >
              <Link
                href="/"
                className="rounded-lg border border-hairline bg-surface-2 px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface"
              >
                Back to runs
              </Link>
            </EmptyState>
          ) : (
            <ApiFailure error={error} context="Loading the run" onRetry={refreshNow} />
          )
        ) : (
          <EmptyState title="Nothing to show for this run yet" hint="Try reloading in a moment.">
            <Button size="sm" onClick={refreshNow}>
              Reload
            </Button>
          </EmptyState>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <RunHeader summary={summary} refreshing={refreshing} onRefresh={refreshNow} />

      {error ? (
        <ApiFailure error={error} context="Refreshing the run" onRetry={refreshNow} />
      ) : null}

      {pendingApproval && dismissedApproval === pendingApproval.request_id ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-hairline border-l-2 border-l-warning bg-surface px-4 py-3">
          <p className="text-xs text-ink-2">
            This run is suspended awaiting your approval on{" "}
            <span className="font-mono text-ink">{pendingApproval.step_id}</span>.
          </p>
          <Button size="sm" variant="primary" onClick={() => setDismissedApproval(null)}>
            Review the request
          </Button>
        </div>
      ) : null}

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(360px,420px)_minmax(0,1fr)]">
        {/* Live column — narration and cost */}
        <div className="space-y-5 xl:sticky xl:top-16">
          <PipelineStepper plan={summary.plan} records={summary.steps} active={active} />
          <EventFeed
            events={stream.events}
            state={stream.state}
            transport={stream.transport}
            attempts={stream.attempts}
            error={stream.error}
            onReconnect={stream.reconnect}
          />
          <UsageMeter summary={summary} active={active} />
        </div>

        {/* Results column — the blackboard, filling in as steps complete */}
        <div className="space-y-5">
          <WarningsPanel warnings={summary.warnings} ingestion={summary.ingestion} />
          <ProfilePanel profile={summary.profile} />
          <UnderstandingPanel understanding={summary.understanding} />
          <ProblemPanel problem={summary.problem} />
          <PlanPanel plan={summary.plan} history={summary.plan_history} />
          <CleaningPanel cleaning={summary.cleaning} />
          <FeaturesPanel features={summary.features} nFeaturesIn={bestFeatureCount} />
          <ModelSelectionPanel selection={summary.model_selection} />
          <LeaderboardPanel experiments={summary.experiments} />
          <TuningPanel
            decision={summary.tuning_decision}
            result={summary.tuning}
            direction={direction}
          />
          <ExplainPanel report={summary.explainability} />
          <EvaluationPanel evaluation={summary.evaluation} />
          <InsightsPanel insights={summary.insights} />
          <ChartsPanel
            runId={runId}
            bundle={summary.visualizations}
            plan={summary.visualization_plan}
          />
          <AskPanel runId={runId} />
          <ApprovalHistory approvals={summary.approvals} />
        </div>
      </div>

      {pendingApproval && dismissedApproval !== pendingApproval.request_id ? (
        <ApprovalModal
          runId={runId}
          request={pendingApproval}
          onClose={() => setDismissedApproval(pendingApproval.request_id)}
          onResolved={refreshNow}
        />
      ) : null}
    </div>
  );
}
