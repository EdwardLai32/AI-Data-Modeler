"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";

import { ReportView, reportToMarkdown } from "@/components/report/ReportView";
import { Badge, Chip } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { ApiFailure, EmptyState, PanelSkeleton, Skeleton, Spinner } from "@/components/ui/States";
import { artifactUrl, asApiError, getArtifactText } from "@/lib/api";
import { basename, timestamp } from "@/lib/format";
import { gradeTone, isRunActive, runStatusLabel, runStatusTone } from "@/lib/labels";
import { useRunSummary } from "@/hooks/useRunSummary";
import type { RunSummary } from "@/types/api";

/**
 * The rendered report, with a format switcher.
 *
 * Lives outside the route folder so Tailwind's source scan sees its classes;
 * see the note in `RunView`.
 *
 * "Rendered" is built from the `FinalReport` object the Report agent returned,
 * so it is always available. The other formats are the files the writer actually
 * wrote — each tab appears only when its path is present on `report_bundle`,
 * because offering a PDF tab that 404s is worse than not offering it.
 */

type Format = "rendered" | "markdown" | "html" | "json";

const LABELS: Record<Format, string> = {
  rendered: "Rendered",
  markdown: "Markdown",
  html: "Standalone HTML",
  json: "JSON",
};

function availableFormats(summary: RunSummary): Format[] {
  const formats: Format[] = ["rendered"];
  if (summary.report_bundle?.markdown_path || summary.report) formats.push("markdown");
  if (summary.report_bundle?.html_path) formats.push("html");
  if (summary.report || summary.report_bundle?.json_path) formats.push("json");
  return formats;
}

/** Raw text view, fetched from the written artifact when one exists. */
function RawMarkdown({ summary }: { summary: RunSummary }) {
  const path = summary.report_bundle?.markdown_path ?? null;
  const derived = useMemo(
    () => (summary.report ? reportToMarkdown(summary.report) : ""),
    [summary.report],
  );
  const [text, setText] = useState<string | null>(path ? null : derived);
  const [note, setNote] = useState<string | null>(
    path ? null : "Assembled from the report object; no markdown file was written for this run.",
  );

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    const controller = new AbortController();
    getArtifactText(summary.run_id, path, controller.signal)
      .then((body) => {
        if (!cancelled) setText(body);
      })
      .catch((raw) => {
        if (cancelled) return;
        if (raw instanceof DOMException && raw.name === "AbortError") return;
        setText(derived);
        setNote(
          `Could not read ${basename(path)} (${asApiError(raw).detail}); showing markdown assembled from the report object instead.`,
        );
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [path, summary.run_id, derived]);

  if (text === null) {
    return (
      <Panel title="Markdown">
        <div className="flex items-center gap-2">
          <Spinner label="Loading markdown" />
          <span className="text-xs text-ink-3">Reading {basename(path)}…</span>
        </div>
      </Panel>
    );
  }

  return (
    <Panel
      title="Markdown source"
      subtitle={note ?? (path ? basename(path) : undefined)}
      aside={
        path ? (
          <a
            href={artifactUrl(summary.run_id, path)}
            target="_blank"
            rel="noreferrer"
            className="text-[11px] font-medium text-ink-2 underline decoration-hairline-strong underline-offset-2 hover:text-ink"
          >
            Download ↗
          </a>
        ) : null
      }
    >
      <pre className="max-h-[70vh] overflow-auto whitespace-pre-wrap rounded-lg border border-hairline bg-surface-2 px-3.5 py-3 font-mono text-[11px] leading-relaxed text-ink-2">
        {text}
      </pre>
    </Panel>
  );
}

export function ReportPageView() {
  const params = useParams<{ runId: string | string[] }>();
  const raw = params?.runId;
  const runId = Array.isArray(raw) ? (raw[0] ?? "") : (raw ?? "");

  // A finished run's report never changes, so there is nothing to poll for.
  const { summary, error, loading, refreshing, refreshNow } = useRunSummary(runId, { pollMs: 0 });
  const [format, setFormat] = useState<Format>("rendered");

  const formats = summary ? availableFormats(summary) : [];
  const ensureValid = useCallback(
    (next: Format) => {
      if (formats.includes(next)) setFormat(next);
    },
    [formats],
  );

  if (loading && !summary) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-6 w-64" />
        <Panel title="Loading report">
          <PanelSkeleton rows={10} />
        </Panel>
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
          <ApiFailure error={error} context="Loading the run" onRetry={refreshNow} />
        ) : (
          <EmptyState title={`No run with id ${runId}`} />
        )}
      </div>
    );
  }

  const bundle = summary.report_bundle;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2 text-xs text-ink-3">
        <Link href="/" className="rounded-md hover:text-ink">
          ← All runs
        </Link>
        <span aria-hidden="true">/</span>
        <Link href={`/runs/${runId}`} className="rounded-md font-mono text-ink-2 hover:text-ink">
          {runId}
        </Link>
        <span aria-hidden="true">/</span>
        <span>Report</span>
      </div>

      <div className="flex flex-wrap items-end justify-between gap-3 rounded-xl border border-hairline bg-surface px-4 py-3.5 sm:px-5">
        <div className="min-w-0">
          <h1 className="text-base font-semibold tracking-tight text-ink">
            {summary.report?.title ?? "Report"}
          </h1>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-3">
            <Badge tone={runStatusTone(summary.status)} pulse={isRunActive(summary.status)}>
              {runStatusLabel(summary.status)}
            </Badge>
            {summary.evaluation ? (
              <Badge tone={gradeTone(summary.evaluation.overall_grade)}>
                Grade {summary.evaluation.overall_grade}
              </Badge>
            ) : null}
            <span>
              {summary.finished_at
                ? `Finished ${timestamp(summary.finished_at)}`
                : "Still running — the report updates as the run completes"}
            </span>
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {refreshing ? <Spinner label="Refreshing" /> : null}
          <Button size="sm" onClick={refreshNow} disabled={refreshing}>
            Refresh
          </Button>
          {bundle?.pdf_path ? (
            <a
              href={artifactUrl(runId, bundle.pdf_path)}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-hairline bg-surface-2 px-2.5 py-1 text-xs font-medium text-ink hover:bg-surface"
            >
              PDF ↗
            </a>
          ) : null}
          {bundle?.pptx_path ? (
            <a
              href={artifactUrl(runId, bundle.pptx_path)}
              target="_blank"
              rel="noreferrer"
              className="rounded-lg border border-hairline bg-surface-2 px-2.5 py-1 text-xs font-medium text-ink hover:bg-surface"
            >
              PPTX ↗
            </a>
          ) : null}
        </div>
      </div>

      {!summary.report && !bundle ? (
        <EmptyState
          title="No report yet"
          hint={
            isRunActive(summary.status)
              ? "The Report agent writes it at the end of the run. This page updates when you refresh."
              : "This run ended before the reporting step produced anything."
          }
        >
          <Link
            href={`/runs/${runId}`}
            className="rounded-lg border border-hairline bg-surface-2 px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface"
          >
            Back to the live view
          </Link>
        </EmptyState>
      ) : (
        <>
          <div
            role="group"
            aria-label="Report format"
            className="flex flex-wrap items-center gap-1"
          >
            {formats.map((option) => (
              <button
                key={option}
                type="button"
                aria-pressed={format === option}
                onClick={() => ensureValid(option)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                  format === option
                    ? "border-accent bg-accent-wash text-ink"
                    : "border-hairline bg-surface-2 text-ink-2 hover:text-ink"
                }`}
              >
                {LABELS[option]}
              </button>
            ))}
            {bundle?.warnings.length ? (
              <Chip title={bundle.warnings.join(" · ")}>
                {bundle.warnings.length} writer warning
                {bundle.warnings.length === 1 ? "" : "s"}
              </Chip>
            ) : null}
          </div>

          {bundle?.warnings.length ? (
            <ul className="space-y-1">
              {bundle.warnings.map((warning, index) => (
                <li key={index} className="text-[11px] leading-relaxed text-ink-3">
                  {warning}
                </li>
              ))}
            </ul>
          ) : null}

          {format === "rendered" ? (
            summary.report ? (
              <ReportView report={summary.report} />
            ) : (
              <EmptyState
                title="No structured report object"
                hint="Only written files are available for this run; switch to Standalone HTML or Markdown."
              />
            )
          ) : null}

          {format === "markdown" ? <RawMarkdown summary={summary} /> : null}

          {format === "html" && bundle?.html_path ? (
            <Panel
              title="Standalone HTML report"
              subtitle={basename(bundle.html_path)}
              aside={
                <a
                  href={artifactUrl(runId, bundle.html_path)}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[11px] font-medium text-ink-2 underline decoration-hairline-strong underline-offset-2 hover:text-ink"
                >
                  Open in a new tab ↗
                </a>
              }
              bodyClassName="p-2"
            >
              <iframe
                src={artifactUrl(runId, bundle.html_path)}
                title="Standalone HTML report"
                className="h-[80vh] w-full rounded-lg border border-hairline bg-white"
              />
            </Panel>
          ) : null}

          {format === "json" ? (
            <Panel
              title="Report JSON"
              subtitle="The FinalReport object exactly as the API returned it"
              aside={
                bundle?.json_path ? (
                  <a
                    href={artifactUrl(runId, bundle.json_path)}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[11px] font-medium text-ink-2 underline decoration-hairline-strong underline-offset-2 hover:text-ink"
                  >
                    Download ↗
                  </a>
                ) : null
              }
            >
              <pre className="max-h-[70vh] overflow-auto rounded-lg border border-hairline bg-surface-2 px-3.5 py-3 font-mono text-[11px] leading-relaxed text-ink-2">
                {JSON.stringify(summary.report, null, 2)}
              </pre>
            </Panel>
          ) : null}
        </>
      )}
    </div>
  );
}
