import type { ReactNode } from "react";

import { TONE_DOT, type Tone } from "@/lib/labels";

/**
 * Status badge: a reserved status hue as a dot, plus the label as text.
 *
 * The colour never carries the meaning on its own — the word is always there —
 * which is what keeps it readable under colour-vision deficiency, in forced
 * colours, and in print.
 */
export function Badge({
  tone = "neutral",
  children,
  pulse = false,
  title,
}: {
  tone?: Tone;
  children: ReactNode;
  /** Animate the dot, for a genuinely in-progress state. */
  pulse?: boolean;
  title?: string;
}) {
  return (
    <span
      title={title}
      className="inline-flex max-w-full items-center gap-1.5 rounded-full border border-hairline bg-surface-2 px-2 py-0.5 text-[11px] font-medium text-ink-2"
    >
      <span
        aria-hidden="true"
        className={`size-1.5 shrink-0 rounded-full ${TONE_DOT[tone]} ${
          pulse ? "animate-pulse" : ""
        }`}
      />
      <span className="truncate">{children}</span>
    </span>
  );
}

/** A neutral pill for an enum value, a column name, or a count. */
export function Chip({
  children,
  mono = false,
  title,
}: {
  children: ReactNode;
  mono?: boolean;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex max-w-full items-center rounded-md border border-hairline bg-surface-2 px-1.5 py-0.5 text-[11px] text-ink-2 ${
        mono ? "font-mono" : ""
      }`}
    >
      <span className="truncate">{children}</span>
    </span>
  );
}

/** Agent attribution: who made this decision. */
export function AgentTag({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-hairline bg-surface px-1.5 py-0.5 text-[11px] font-medium text-ink-2">
      <span aria-hidden="true" className="size-1.5 rounded-full bg-accent" />
      {children}
    </span>
  );
}
