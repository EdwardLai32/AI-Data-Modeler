"use client";

import { useEffect, useRef, useState } from "react";

import { asApiError, getFigure } from "@/lib/api";
import { Spinner } from "@/components/ui/States";

/**
 * Render a plotly figure written by the charts executor.
 *
 * The figure's `data` is passed through untouched — the traces, values, and
 * colours are the backend's, and re-deriving them here would break the
 * guarantee that every number on screen came from the pipeline. Only the chart
 * *chrome* is themed (surface, ink, gridlines) so the embedded chart does not
 * sit on a white rectangle in dark mode.
 *
 * plotly.js is ~3 MB, so it is imported dynamically: pages with no charts never
 * download it.
 */

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

/** Plotly's title can be a string or `{text: …}` depending on the writer. */
function titleText(layout: Record<string, unknown>): string {
  const title = layout.title;
  if (typeof title === "string") return title;
  const text = asRecord(title).text;
  return typeof text === "string" ? text : "";
}

/**
 * Merge the page's chrome into the figure's own layout.
 *
 * Every key is merged one level deep rather than replaced: a shallow spread
 * would drop the axis titles and tick formats the chart executor set, which are
 * data, not decoration. Only colours, fonts, and margins are ours.
 *
 * `cardTitle` lets a figure title that merely repeats the card heading be
 * suppressed — the heading is already there, and two copies waste the plot area.
 */
function themedLayout(base: Record<string, unknown>, cardTitle: string): Record<string, unknown> {
  const styles = getComputedStyle(document.documentElement);
  const read = (token: string, fallback: string) =>
    styles.getPropertyValue(token).trim() || fallback;

  const ink = read("--ink-2", "#52514e");
  const muted = read("--ink-3", "#898781");
  const grid = read("--grid", "#e1e0d9");
  const baseline = read("--baseline", "#c3c2b7");

  const axis = (existing: unknown) => ({
    ...asRecord(existing),
    gridcolor: grid,
    zerolinecolor: baseline,
    linecolor: baseline,
    // Let plotly size the margin around real tick text; a fixed margin clips
    // long category names such as one-hot encoded feature columns.
    automargin: true,
    tickfont: { ...asRecord(asRecord(existing).tickfont), color: muted },
    title: {
      ...asRecord(asRecord(existing).title),
      font: { ...asRecord(asRecord(asRecord(existing).title).font), color: muted },
    },
  });

  const figureTitle = titleText(base).trim();
  const keepTitle = Boolean(figureTitle) && figureTitle !== cardTitle.trim();

  return {
    ...base,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      ...asRecord(base.font),
      color: ink,
      family: 'system-ui, -apple-system, "Segoe UI", sans-serif',
      size: 11,
    },
    title: keepTitle
      ? { ...asRecord(base.title), text: figureTitle, font: { color: ink, size: 13 } }
      : { text: "" },
    margin: { l: 8, r: 12, t: keepTitle ? 34 : 8, b: 8 },
    xaxis: axis(base.xaxis),
    yaxis: axis(base.yaxis),
    legend: {
      ...asRecord(base.legend),
      font: { ...asRecord(asRecord(base.legend).font), color: ink },
    },
    hoverlabel: {
      ...asRecord(base.hoverlabel),
      bgcolor: read("--surface", "#fcfcfb"),
      bordercolor: baseline,
      font: { color: read("--ink", "#0b0b0b") },
    },
  };
}

export function PlotlyFigureView({
  runId,
  jsonPath,
  title,
  height = 320,
  onUnavailable,
}: {
  runId: string;
  jsonPath: string;
  title: string;
  height?: number;
  /** Called when the JSON cannot be loaded, so the parent can fall back. */
  onUnavailable?: (reason: string) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const element = container.current;
    let plotly: typeof import("plotly.js-dist-min").default | null = null;
    let observer: MutationObserver | null = null;
    let figureLayout: Record<string, unknown> = {};
    let onResize: (() => void) | null = null;

    const draw = async () => {
      if (!element) return;
      try {
        const [figure, mod] = await Promise.all([
          getFigure(runId, jsonPath, controller.signal),
          import("plotly.js-dist-min"),
        ]);
        if (cancelled) return;
        plotly = mod.default;

        figureLayout = asRecord(figure.layout);
        await plotly.newPlot(element, figure.data, themedLayout(figureLayout, title), {
          responsive: true,
          displaylogo: false,
          displayModeBar: false,
          // Charts are read, not edited; scroll should scroll the page.
          scrollZoom: false,
        });
        if (cancelled) return;
        setState("ready");

        onResize = () => {
          if (plotly && element) plotly.Plots.resize(element);
        };
        window.addEventListener("resize", onResize);

        // Re-theme when the light/dark class flips.
        observer = new MutationObserver(() => {
          if (plotly && element) void plotly.relayout(element, themedLayout(figureLayout, title));
        });
        observer.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ["class"],
        });
      } catch (raw) {
        if (raw instanceof DOMException && raw.name === "AbortError") return;
        if (cancelled) return;
        const failure = asApiError(raw);
        setError(failure.detail);
        setState("error");
        onUnavailable?.(failure.detail);
      }
    };

    void draw();

    return () => {
      cancelled = true;
      controller.abort();
      if (onResize) window.removeEventListener("resize", onResize);
      observer?.disconnect();
      if (plotly && element) plotly.purge(element);
    };
  }, [runId, jsonPath, title, onUnavailable]);

  return (
    <div>
      <div
        ref={container}
        role="img"
        aria-label={title}
        style={{ height }}
        className="w-full"
      />
      {state === "loading" ? (
        <div className="flex items-center gap-2 px-1 py-2">
          <Spinner label={`Loading ${title}`} />
          <span className="text-[11px] text-ink-3">Loading chart…</span>
        </div>
      ) : null}
      {state === "error" ? (
        <p className="px-1 py-2 text-[11px] leading-relaxed text-ink-3">
          Chart JSON unavailable{error ? `: ${error}` : "."}
        </p>
      ) : null}
    </div>
  );
}
