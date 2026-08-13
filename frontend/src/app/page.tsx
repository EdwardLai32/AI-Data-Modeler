"use client";

import { useMemo, useState } from "react";

import { NewAnalysisPanel } from "@/components/dashboard/NewAnalysisPanel";
import { RunList, RunListStats } from "@/components/dashboard/RunList";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { TextInput } from "@/components/ui/Field";
import {
  ApiFailure,
  BackendDown,
  EmptyState,
  ErrorBanner,
  Spinner,
} from "@/components/ui/States";
import { useHealth } from "@/hooks/useHealth";
import { useRuns } from "@/hooks/useRuns";
import { API_BASE } from "@/lib/api";

/**
 * Dashboard: run history on the left, launcher on the right.
 *
 * Reachability is resolved before anything else. If the API is down the page
 * says so, with the command that fixes it, rather than rendering an empty list
 * that reads as "you have no runs".
 */
export default function DashboardPage() {
  const health = useHealth();
  const { runs, error, loading, refreshing, refresh } = useRuns({ limit: 100 });
  const [filter, setFilter] = useState("");

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return runs;
    return runs.filter((run) =>
      [run.runId, run.project, run.targetColumn ?? "", run.taskType ?? "", run.status]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [runs, filter]);

  const backendDown = health.state === "unreachable" || error?.unreachable === true;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-ink">Runs</h1>
          <p className="mt-0.5 text-xs text-ink-3">
            Every analysis this workspace has executed, newest first.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {health.state === "up" ? (
            <Badge tone="good" title={`Connected to ${API_BASE}`}>
              API connected
              {health.health?.version ? ` · v${health.health.version}` : ""}
            </Badge>
          ) : null}
          {health.state === "checking" ? <Spinner label="Checking the API" /> : null}
          {refreshing ? <Spinner label="Refreshing runs" /> : null}
          <Button size="sm" onClick={refresh} disabled={refreshing}>
            Refresh
          </Button>
        </div>
      </div>

      {backendDown ? (
        <BackendDown
          detail={health.error ?? error?.detail}
          onRetry={() => {
            health.recheck();
            refresh();
          }}
        />
      ) : null}

      {health.state === "error" ? (
        <ErrorBanner
          title="The API is running but its health check failed"
          detail={health.error ?? undefined}
          onRetry={health.recheck}
        />
      ) : null}

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(360px,420px)]">
        <Panel
          title="Run history"
          aside={<RunListStats runs={runs} />}
          bodyClassName="pt-2"
          subtitle={health.health?.workspace ? `Workspace: ${health.health.workspace}` : undefined}
        >
          <div className="mb-3">
            <label htmlFor="run-filter" className="sr-only">
              Filter loaded runs
            </label>
            <TextInput
              id="run-filter"
              type="search"
              value={filter}
              placeholder="Filter by run id, project, target, task, or status"
              onChange={(event) => setFilter(event.target.value)}
            />
          </div>

          {error && !backendDown ? (
            <div className="mb-3">
              <ApiFailure error={error} context="Loading runs" onRetry={refresh} />
            </div>
          ) : null}

          {backendDown && !runs.length ? (
            // "No runs yet" would be a claim about the database; with the API
            // down the honest statement is that nothing could be read.
            <EmptyState
              title="Run history could not be loaded"
              hint="The API is unreachable, so this list is unknown rather than empty."
            />
          ) : filter && !visible.length && runs.length ? (
            <p className="rounded-lg border border-dashed border-hairline-strong px-4 py-6 text-center text-sm text-ink-2">
              No loaded run matches “{filter}”.
            </p>
          ) : (
            <RunList runs={visible} loading={loading} />
          )}
        </Panel>

        <NewAnalysisPanel onStarted={refresh} />
      </div>
    </div>
  );
}
