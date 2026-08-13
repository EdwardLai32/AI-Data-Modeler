import type { Metadata, Viewport } from "next";
import Link from "next/link";

import "./globals.css";
import { THEME_BOOTSTRAP_SCRIPT, ThemeToggle } from "@/components/ThemeToggle";

export const metadata: Metadata = {
  title: {
    default: "AutoML Architect",
    template: "%s · AutoML Architect",
  },
  description:
    "Live view of an autonomous multi-agent data-science run: every plan step, every agent decision, and the reasoning behind it.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Applies the stored theme before first paint. Without this the page
            flashes light before hydration on a dark-mode machine. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP_SCRIPT }} />
      </head>
      <body className="min-h-dvh bg-plane text-ink antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:border focus:border-hairline focus:bg-surface focus:px-3 focus:py-2 focus:text-sm"
        >
          Skip to content
        </a>
        <header className="sticky top-0 z-30 border-b border-hairline bg-surface/85 backdrop-blur">
          <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-4 py-2.5 sm:px-6">
            <Link
              href="/"
              className="group flex min-w-0 items-center gap-2.5 rounded-md py-0.5"
              aria-label="AutoML Architect home"
            >
              <span
                aria-hidden="true"
                className="grid size-7 shrink-0 place-items-center rounded-md bg-accent text-[13px] font-bold text-white"
              >
                A
              </span>
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold tracking-tight text-ink">
                  AutoML Architect
                </span>
                <span className="block truncate text-[11px] text-ink-3">
                  Agents decide · Python computes
                </span>
              </span>
            </Link>
            <div className="flex items-center gap-2">
              <Link
                href="/"
                className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-ink-2 hover:bg-surface-2 hover:text-ink"
              >
                Runs
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>
        <main id="main" className="mx-auto max-w-[1600px] px-4 py-5 sm:px-6 sm:py-6">
          {children}
        </main>
      </body>
    </html>
  );
}
