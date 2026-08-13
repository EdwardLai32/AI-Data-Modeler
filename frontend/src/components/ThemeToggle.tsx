"use client";

import { useEffect, useState } from "react";

/**
 * Light / dark / system theme switch.
 *
 * Dark mode is a chosen theme, not an inversion: the tokens in `globals.css`
 * define their own dark steps. The choice is persisted and applied by an inline
 * script in `layout.tsx` before first paint, so there is no flash of the wrong
 * theme; this component only keeps the class in sync afterwards.
 */

export type ThemeChoice = "light" | "dark" | "system";

export const THEME_STORAGE_KEY = "amla-theme";

/** Inline, blocking, and deliberately tiny: it runs before the first paint. */
export const THEME_BOOTSTRAP_SCRIPT = `(function(){try{var c=localStorage.getItem("${THEME_STORAGE_KEY}")||"system";var d=c==="dark"||(c!=="light"&&window.matchMedia("(prefers-color-scheme: dark)").matches);var r=document.documentElement;r.classList.toggle("dark",d);r.style.colorScheme=d?"dark":"light";}catch(e){}})();`;

function apply(choice: ThemeChoice): void {
  const dark =
    choice === "dark" ||
    (choice === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.style.colorScheme = dark ? "dark" : "light";
}

const OPTIONS: Array<{ value: ThemeChoice; label: string; glyph: string }> = [
  { value: "light", label: "Light", glyph: "☀" },
  { value: "dark", label: "Dark", glyph: "☾" },
  { value: "system", label: "System", glyph: "◐" },
];

export function ThemeToggle() {
  const [choice, setChoice] = useState<ThemeChoice>("system");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY) as ThemeChoice | null;
    const initial: ThemeChoice =
      stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
    setChoice(initial);
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    window.localStorage.setItem(THEME_STORAGE_KEY, choice);
    apply(choice);
    if (choice !== "system") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => apply("system");
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [choice, ready]);

  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className="inline-flex items-center gap-0.5 rounded-lg border border-hairline bg-surface-2 p-0.5"
    >
      {OPTIONS.map((option) => {
        const active = ready && choice === option.value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={`${option.label} theme`}
            onClick={() => setChoice(option.value)}
            className={`rounded-md px-2 py-1 text-xs transition-colors ${
              active ? "bg-surface text-ink shadow-[0_1px_2px_rgba(0,0,0,0.06)]" : "text-ink-3 hover:text-ink"
            }`}
          >
            <span aria-hidden="true">{option.glyph}</span>
            <span className="sr-only">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}
