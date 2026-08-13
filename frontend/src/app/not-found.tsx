import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-md rounded-xl border border-hairline bg-surface px-5 py-6 text-center">
      <p className="text-sm font-semibold text-ink">That page does not exist</p>
      <p className="mt-1 text-xs leading-relaxed text-ink-2">
        The dashboard has three views: the run list, a run&rsquo;s live view, and that run&rsquo;s
        report.
      </p>
      <Link
        href="/"
        className="mt-4 inline-flex items-center rounded-lg border border-transparent bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-strong"
      >
        Back to runs
      </Link>
    </div>
  );
}
