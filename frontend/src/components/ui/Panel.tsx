import type { ReactNode } from "react";

/**
 * The one card shell every panel uses.
 *
 * `title` renders as a real heading so the run page is navigable by screen
 * reader headings; `aside` is for a status chip or a count, never for controls
 * that need their own label.
 */
export function Panel({
  title,
  subtitle,
  aside,
  children,
  className = "",
  bodyClassName = "",
  as: Heading = "h2",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  as?: "h2" | "h3";
}) {
  return (
    <section
      className={`rounded-xl border border-hairline bg-surface shadow-[0_1px_2px_rgba(0,0,0,0.04)] ${className}`}
    >
      {(title || aside) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-hairline px-4 py-3 sm:px-5">
          <div className="min-w-0">
            {title ? (
              <Heading className="text-sm font-semibold tracking-tight text-ink">{title}</Heading>
            ) : null}
            {subtitle ? (
              <p className="mt-0.5 text-xs leading-relaxed text-ink-3">{subtitle}</p>
            ) : null}
          </div>
          {aside ? <div className="shrink-0">{aside}</div> : null}
        </header>
      )}
      <div className={`px-4 py-4 sm:px-5 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

/** A labelled fact. Used in dense two-column detail grids. */
export function KeyValue({
  label,
  children,
  mono = false,
  wrap = false,
}: {
  label: string;
  children: ReactNode;
  mono?: boolean;
  /** Let the value run to several lines. Off by default so grids stay aligned. */
  wrap?: boolean;
}) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] font-medium uppercase tracking-wide text-ink-3">{label}</dt>
      <dd
        className={`mt-0.5 text-sm text-ink ${
          wrap ? "prose-agent text-xs leading-relaxed text-ink-2" : "truncate"
        } ${mono ? "font-mono text-[13px]" : ""}`}
        title={!wrap && typeof children === "string" ? children : undefined}
      >
        {children}
      </dd>
    </div>
  );
}

/** A hairline-separated list, for stacked decision rows. */
export function Divided({ children }: { children: ReactNode }) {
  return <div className="divide-y divide-hairline">{children}</div>;
}
