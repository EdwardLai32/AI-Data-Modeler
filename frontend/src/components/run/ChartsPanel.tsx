"use client";

import { useCallback, useEffect, useState } from "react";

import { Badge, Chip } from "@/components/ui/Badge";
import { Panel } from "@/components/ui/Panel";
import { Disclosure } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { PlotlyFigureView } from "@/components/run/PlotlyFigureView";
import { API_BASE, artifactUrl, listArtifacts } from "@/lib/api";
import { basename } from "@/lib/format";
import { chartKindLabel } from "@/lib/labels";
import type { ArtifactRef, ChartArtifact, VisualizationBundle, VisualizationPlan } from "@/types/api";

/**
 * Embedded charts.
 *
 * Preference order per chart: the plotly JSON (interactive, themed to match the
 * page), then the standalone HTML in an iframe, then the PNG. The fallback chain
 * matters because chart rendering degrades independently — kaleido may be absent
 * for PNGs, or a single figure may have failed while the rest succeeded, and
 * neither should blank the panel.
 */

type Mode = "json" | "html" | "png" | "none";

function initialMode(artifact: ChartArtifact): Mode {
  if (artifact.json_path) return "json";
  if (artifact.html_path) return "html";
  if (artifact.png_path) return "png";
  return "none";
}

function downgrade(current: Mode, artifact: ChartArtifact): Mode {
  if (current === "json" && artifact.html_path) return "html";
  if ((current === "json" || current === "html") && artifact.png_path) return "png";
  return "none";
}

function ChartCard({ runId, artifact }: { runId: string; artifact: ChartArtifact }) {
  const [mode, setMode] = useState<Mode>(() => initialMode(artifact));
  const fallback = useCallback(() => {
    setMode((current) => downgrade(current, artifact));
  }, [artifact]);

  const openUrl =
    (artifact.html_path && artifactUrl(runId, artifact.html_path)) ||
    (artifact.png_path && artifactUrl(runId, artifact.png_path)) ||
    (artifact.json_path && artifactUrl(runId, artifact.json_path)) ||
    null;

  return (
    <figure className="min-w-0 rounded-lg border border-hairline bg-surface">
      <figcaption className="flex flex-wrap items-start justify-between gap-2 border-b border-hairline px-3 py-2">
        <div className="min-w-0">
          <p className="truncate text-xs font-semibold text-ink">{artifact.spec.title}</p>
          <p className="mt-0.5 text-[11px] text-ink-3">{chartKindLabel(artifact.spec.kind)}</p>
        </div>
        <span className="flex shrink-0 items-center gap-1.5">
          {!artifact.rendered ? <Badge tone="serious">Not rendered</Badge> : null}
          {openUrl ? (
            <a
              href={openUrl}
              target="_blank"
              rel="noreferrer"
              className="rounded-md px-1.5 py-0.5 text-[11px] font-medium text-ink-2 underline decoration-hairline-strong underline-offset-2 hover:text-ink"
            >
              Open ↗
            </a>
          ) : null}
        </span>
      </figcaption>

      <div className="px-2 py-2">
        {artifact.error ? (
          <p className="px-1 py-3 text-[11px] leading-relaxed text-ink-2">
            The chart executor reported: {artifact.error}
          </p>
        ) : mode === "json" && artifact.json_path ? (
          <PlotlyFigureView
            runId={runId}
            jsonPath={artifact.json_path}
            title={artifact.spec.title}
            onUnavailable={fallback}
          />
        ) : mode === "html" && artifact.html_path ? (
          <iframe
            src={artifactUrl(runId, artifact.html_path)}
            title={artifact.spec.title}
            loading="lazy"
            className="h-[320px] w-full rounded-md border-0 bg-surface"
          />
        ) : mode === "png" && artifact.png_path ? (
          // eslint-disable-next-line @next/next/no-img-element -- served by the API, not Next
          <img
            src={artifactUrl(runId, artifact.png_path)}
            alt={artifact.spec.title}
            className="w-full rounded-md"
          />
        ) : (
          <p className="px-1 py-3 text-[11px] leading-relaxed text-ink-3">
            No renderable file for this chart. The artifact endpoint may not be serving
            {" "}
            {basename(artifact.json_path ?? artifact.html_path ?? artifact.png_path)}.
          </p>
        )}
      </div>

      {artifact.caption || artifact.spec.rationale ? (
        <div className="border-t border-hairline px-3 py-2">
          {artifact.caption ? (
            <p className="prose-agent text-[11px] leading-relaxed text-ink-2">
              {artifact.caption}
            </p>
          ) : null}
          {artifact.spec.rationale ? (
            <p className="prose-agent mt-1 text-[11px] leading-relaxed text-ink-3">
              Answers: {artifact.spec.rationale}
            </p>
          ) : null}
        </div>
      ) : null}
    </figure>
  );
}

export function ChartsPanel({
  runId,
  bundle,
  plan,
}: {
  runId: string;
  bundle: VisualizationBundle | null;
  plan: VisualizationPlan | null;
}) {
  const [extra, setExtra] = useState<ArtifactRef[]>([]);
  const [artifactsProbed, setArtifactsProbed] = useState(false);

  // The artifact index is optional; a 404 here is not an error worth showing.
  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    listArtifacts(runId, controller.signal)
      .then((rows) => {
        if (!cancelled) setExtra(rows);
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setArtifactsProbed(true);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [runId]);

  const artifacts = bundle?.artifacts ?? [];
  const rendered = artifacts.filter((artifact) => artifact.rendered || artifact.json_path || artifact.html_path || artifact.png_path);
  const plannedOnly = (plan?.charts ?? []).filter(
    (spec) => !artifacts.some((artifact) => artifact.spec.title === spec.title),
  );

  return (
    <Panel
      title="Charts"
      subtitle={bundle?.narrative || plan?.dashboard_narrative || undefined}
      aside={
        <span className="flex flex-wrap items-center justify-end gap-1.5">
          {rendered.length ? <Chip>{rendered.length} rendered</Chip> : null}
          {bundle?.dashboard_path ? (
            <a
              href={artifactUrl(runId, bundle.dashboard_path)}
              target="_blank"
              rel="noreferrer"
              className="rounded-md px-1.5 py-0.5 text-[11px] font-medium text-ink-2 underline decoration-hairline-strong underline-offset-2 hover:text-ink"
            >
              Full dashboard ↗
            </a>
          ) : null}
        </span>
      }
      bodyClassName="space-y-3"
    >
      {!rendered.length ? (
        <EmptyState
          title={plannedOnly.length ? "Charts planned, none rendered yet" : "No charts yet"}
          hint={
            plannedOnly.length
              ? "The Visualization agent has chosen the charts; the renderer fills them in as the run proceeds."
              : "The Visualization agent picks charts once there are results worth plotting."
          }
        />
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {rendered.map((artifact, index) => (
            <ChartCard key={`${artifact.spec.title}-${index}`} runId={runId} artifact={artifact} />
          ))}
        </div>
      )}

      {plannedOnly.length ? (
        <Disclosure summary="Planned but not yet rendered" count={plannedOnly.length}>
          <ul className="space-y-1.5">
            {plannedOnly.map((spec, index) => (
              <li key={`${spec.title}-${index}`} className="text-[11px] leading-relaxed text-ink-2">
                <span className="font-medium text-ink">{spec.title}</span>{" "}
                <Chip>{chartKindLabel(spec.kind)}</Chip>
                {spec.rationale ? <span className="block text-ink-3">{spec.rationale}</span> : null}
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : null}

      {artifactsProbed && extra.length ? (
        <Disclosure summary="All files written by this run" count={extra.length}>
          <ul className="space-y-1">
            {extra.map((item, index) => {
              // The index already returns a server-relative URL; only the origin
              // is missing. `artifactUrl` must not be used on it — it would
              // treat the URL as a filesystem path and double the prefix.
              const href = item.url
                ? item.url.startsWith("http")
                  ? item.url
                  : `${API_BASE}${item.url}`
                : artifactUrl(runId, item.relative_path);
              return (
                <li key={`${item.relative_path}-${index}`} className="truncate text-[11px]">
                  <a
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    className="font-mono text-ink-2 underline decoration-hairline-strong underline-offset-2 hover:text-ink"
                  >
                    {item.relative_path}
                  </a>
                  {item.kind ? <span className="ml-2 text-ink-3">{item.kind}</span> : null}
                </li>
              );
            })}
          </ul>
        </Disclosure>
      ) : null}
    </Panel>
  );
}
