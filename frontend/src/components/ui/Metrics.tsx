import type { ReactNode } from "react";

import type { Tone } from "@/lib/labels";

/**
 * Figures and bars.
 *
 * Shared rules, applied here once: bars are thin, grow from a single baseline,
 * and round only the data end; the number is always written out beside the mark
 * rather than left to be read off the bar's length; and values wear text
 * colours, never the mark's colour.
 */

const FILL: Record<Tone, string> = {
  neutral: "bg-baseline",
  accent: "bg-accent",
  good: "bg-good",
  warning: "bg-warning",
  serious: "bg-serious",
  critical: "bg-critical",
};

/** A labelled number. `note` carries units, denominators, or a caveat. */
export function StatTile({
  label,
  value,
  note,
  children,
}: {
  label: string;
  value: ReactNode;
  note?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-hairline bg-surface px-3.5 py-3">
      <p className="text-[11px] font-medium uppercase tracking-wide text-ink-3">{label}</p>
      <p className="mt-1 truncate text-xl font-semibold text-ink">{value}</p>
      {note ? <p className="mt-0.5 truncate text-xs text-ink-3">{note}</p> : null}
      {children}
    </div>
  );
}

/**
 * The one number a view leads with. Exactly one per page.
 */
export function Hero({
  label,
  value,
  note,
  aside,
}: {
  label: string;
  value: ReactNode;
  note?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <p className="text-[11px] font-medium uppercase tracking-wide text-ink-3">{label}</p>
        <p className="mt-1 text-[clamp(2.75rem,6vw,3.5rem)] font-semibold leading-none tracking-tight text-ink">
          {value}
        </p>
        {note ? <p className="mt-1.5 text-xs text-ink-2">{note}</p> : null}
      </div>
      {aside ? <div className="shrink-0">{aside}</div> : null}
    </div>
  );
}

/**
 * A bounded ratio.
 *
 * The unfilled track is a lighter step of the fill's own ramp, so the state
 * reads across the whole bar. `tone` carries severity when the ratio itself is
 * a warning (a time budget running out, for instance).
 */
export function Meter({
  label,
  fraction,
  valueText,
  tone = "accent",
  hint,
}: {
  label: string;
  /** 0-1. Values outside the range are clamped for drawing, not for the text. */
  fraction: number | null;
  valueText: string;
  tone?: Tone;
  hint?: string;
}) {
  const clamped = fraction === null ? 0 : Math.min(1, Math.max(0, fraction));
  const percent = Math.round(clamped * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="truncate text-xs text-ink-2">{label}</span>
        <span className="tabular shrink-0 text-xs font-medium text-ink">{valueText}</span>
      </div>
      <div
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={fraction === null ? undefined : percent}
        aria-valuetext={valueText}
        className="mt-1.5 h-2 w-full overflow-hidden rounded-sm bg-accent-track"
      >
        {fraction === null ? null : (
          <div
            className={`h-full rounded-r-[4px] ${FILL[tone]}`}
            style={{ width: `${Math.max(clamped > 0 ? 2 : 0, percent)}%` }}
          />
        )}
      </div>
      {hint ? <p className="mt-1 text-[11px] text-ink-3">{hint}</p> : null}
    </div>
  );
}

/**
 * One row of a horizontal bar chart: name, mark, value.
 *
 * The value is a direct label on every row here because these lists are short
 * (top-N features, a leaderboard) and the exact number is the point.
 */
export function DataBar({
  name,
  fraction,
  valueText,
  tone = "accent",
  aside,
  nameTitle,
}: {
  name: ReactNode;
  fraction: number;
  valueText: string;
  tone?: Tone;
  aside?: ReactNode;
  nameTitle?: string;
}) {
  const clamped = Math.min(1, Math.max(0, fraction));
  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="flex min-w-0 items-center gap-1.5 truncate text-xs text-ink" title={nameTitle}>
          {name}
        </span>
        <span className="flex shrink-0 items-center gap-2">
          {aside}
          <span className="tabular text-xs font-medium text-ink">{valueText}</span>
        </span>
      </div>
      <div className="mt-1 h-2 w-full rounded-sm bg-surface-2">
        <div
          className={`h-full rounded-r-[4px] ${FILL[tone]}`}
          style={{ width: `${Math.max(2, clamped * 100)}%` }}
        />
      </div>
    </div>
  );
}
