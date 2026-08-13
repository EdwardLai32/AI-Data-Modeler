import type { ReactNode } from "react";

/**
 * The reason behind a decision.
 *
 * Rationale is the product, not a footnote, so it is rendered as prose with a
 * quote rule rather than hidden in a tooltip. `<details>` is used for the
 * secondary fields (hypothesis, risk, parameters) because it is keyboard- and
 * screen-reader-navigable with no JavaScript.
 */
export function Rationale({
  children,
  label = "Why",
}: {
  children: ReactNode;
  label?: string;
}) {
  return (
    <div className="mt-1.5 border-l-2 border-baseline pl-3">
      <p className="text-[10px] font-semibold uppercase tracking-wide text-ink-3">{label}</p>
      <p className="prose-agent mt-0.5 text-xs leading-relaxed text-ink-2">{children}</p>
    </div>
  );
}

export function Disclosure({
  summary,
  children,
  count,
}: {
  summary: string;
  children: ReactNode;
  count?: number;
}) {
  return (
    <details className="group mt-2">
      <summary className="inline-flex cursor-pointer list-none items-center gap-1.5 rounded-md px-1.5 py-1 text-[11px] font-medium text-ink-2 hover:bg-surface-2 hover:text-ink">
        <span aria-hidden="true" className="transition-transform group-open:rotate-90">
          ▸
        </span>
        {summary}
        {typeof count === "number" ? <span className="text-ink-3">({count})</span> : null}
      </summary>
      <div className="mt-1.5 pl-4">{children}</div>
    </details>
  );
}

/** A short labelled note: hypothesis, risk, expected impact. */
export function NoteLine({ label, children }: { label: string; children: ReactNode }) {
  return (
    <p className="prose-agent text-xs leading-relaxed text-ink-2">
      <span className="font-medium text-ink-3">{label}: </span>
      {children}
    </p>
  );
}

/** Bulleted list of agent-authored strings. */
export function BulletList({
  items,
  className = "",
}: {
  items: string[];
  className?: string;
}) {
  if (!items.length) return null;
  return (
    <ul className={`space-y-1 ${className}`}>
      {items.map((item, index) => (
        <li key={`${index}-${item.slice(0, 24)}`} className="flex gap-2 text-xs leading-relaxed text-ink-2">
          <span aria-hidden="true" className="mt-1.5 size-1 shrink-0 rounded-full bg-baseline" />
          <span className="prose-agent">{item}</span>
        </li>
      ))}
    </ul>
  );
}
