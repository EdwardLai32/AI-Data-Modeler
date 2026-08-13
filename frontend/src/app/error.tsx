"use client";

import { useEffect } from "react";

/**
 * Last-resort boundary.
 *
 * Network and API failures are handled inside the panels that own them; this
 * only catches a genuine rendering bug, and it says so rather than pretending
 * the backend is at fault.
 */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Unhandled rendering error in the AI Data Modeler dashboard", error);
  }, [error]);

  return (
    <div
      role="alert"
      className="mx-auto max-w-lg rounded-xl border border-hairline border-l-2 border-l-critical bg-surface px-5 py-5"
    >
      <p className="text-sm font-semibold text-ink">The dashboard hit a rendering error</p>
      <p className="mt-1 text-xs leading-relaxed text-ink-2">
        This is a bug in the frontend, not a backend failure. The message below is the
        browser&rsquo;s; the full stack is in the browser console.
      </p>
      <pre className="mt-3 overflow-x-auto rounded-lg border border-hairline bg-surface-2 px-3 py-2 font-mono text-[11px] text-ink-2">
        {error.message}
        {error.digest ? `\ndigest: ${error.digest}` : ""}
      </pre>
      <button
        type="button"
        onClick={reset}
        className="mt-4 inline-flex items-center rounded-lg border border-transparent bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-strong"
      >
        Try rendering again
      </button>
    </div>
  );
}
