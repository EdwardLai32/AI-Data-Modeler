"use client";

import type { ReactNode } from "react";

import { API_BASE, ApiError, SERVE_COMMAND } from "@/lib/api";
import { Button } from "@/components/ui/Button";

/** Indeterminate activity. Announced once, not on every frame. */
export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-ink-3">
      <span
        aria-hidden="true"
        className="size-3.5 animate-spin rounded-full border-2 border-baseline border-t-accent"
      />
      <span className="sr-only">{label}</span>
    </span>
  );
}

/** Layout-stable placeholder for content that is genuinely on its way. */
export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded-md bg-surface-2 ${className}`}
    />
  );
}

export function PanelSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={index} className={index === 0 ? "h-4 w-2/5" : "h-3.5 w-full"} />
      ))}
    </div>
  );
}

/**
 * Nothing here yet — and why.
 *
 * Every empty state says what will fill it, so a panel that is waiting on the
 * pipeline never looks like a panel that is broken.
 */
export function EmptyState({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children?: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-dashed border-hairline-strong px-4 py-6 text-center">
      <p className="text-sm font-medium text-ink-2">{title}</p>
      {hint ? <p className="mx-auto mt-1 max-w-prose text-xs text-ink-3">{hint}</p> : null}
      {children ? <div className="mt-3 flex justify-center">{children}</div> : null}
    </div>
  );
}

/** A recoverable failure, with the retry that recovers it. */
export function ErrorBanner({
  title,
  detail,
  onRetry,
  retryLabel = "Try again",
}: {
  title: string;
  detail?: string;
  onRetry?: () => void;
  retryLabel?: string;
}) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-hairline border-l-2 border-l-critical bg-surface-2 px-4 py-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="flex items-center gap-2 text-sm font-medium text-ink">
            <span aria-hidden="true" className="size-1.5 rounded-full bg-critical" />
            {title}
          </p>
          {detail ? (
            <p className="mt-1 max-w-prose break-words text-xs leading-relaxed text-ink-2">
              {detail}
            </p>
          ) : null}
        </div>
        {onRetry ? (
          <Button size="sm" onClick={onRetry}>
            {retryLabel}
          </Button>
        ) : null}
      </div>
    </div>
  );
}

/**
 * The backend is not running.
 *
 * This is the single most likely failure on a fresh checkout, so it gets a real
 * explanation and the exact command rather than a spinner that never resolves.
 */
export function BackendDown({
  detail,
  onRetry,
}: {
  detail?: string;
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="rounded-xl border border-hairline border-l-2 border-l-warning bg-surface px-5 py-4"
    >
      <p className="flex items-center gap-2 text-sm font-semibold text-ink">
        <span aria-hidden="true" className="size-1.5 rounded-full bg-warning" />
        The AI Data Modeler API is not responding
      </p>
      <p className="mt-2 max-w-prose text-xs leading-relaxed text-ink-2">
        Nothing on this page can load until the backend is up. Start it from the repository
        root:
      </p>
      <pre className="mt-2 overflow-x-auto rounded-lg border border-hairline bg-surface-2 px-3 py-2 font-mono text-xs text-ink">
        {SERVE_COMMAND}
      </pre>
      <dl className="mt-3 grid gap-1 text-xs text-ink-3 sm:grid-cols-[auto_1fr] sm:gap-x-3">
        <dt className="font-medium">Expected at</dt>
        <dd className="font-mono break-all text-ink-2">{API_BASE}</dd>
        <dt className="font-medium">Override with</dt>
        <dd className="font-mono break-all text-ink-2">NEXT_PUBLIC_API_URL</dd>
        {detail ? (
          <>
            <dt className="font-medium">Browser reported</dt>
            <dd className="break-words text-ink-2">{detail}</dd>
          </>
        ) : null}
      </dl>
      {onRetry ? (
        <div className="mt-3">
          <Button size="sm" variant="primary" onClick={onRetry}>
            Retry connection
          </Button>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Pick the right failure surface for an `ApiError`.
 *
 * A transport failure and a 500 look nothing alike to the person reading them,
 * so they are never collapsed into one message.
 */
export function ApiFailure({
  error,
  onRetry,
  context,
}: {
  error: ApiError;
  onRetry?: () => void;
  context?: string;
}) {
  if (error.unreachable) return <BackendDown detail={error.detail} onRetry={onRetry} />;
  return (
    <ErrorBanner
      title={
        context
          ? `${context} failed${error.status ? ` (${error.status})` : ""}`
          : `Request failed${error.status ? ` (${error.status})` : ""}`
      }
      detail={error.detail}
      onRetry={onRetry}
    />
  );
}
