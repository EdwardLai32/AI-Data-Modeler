"use client";

import { useEffect, useState } from "react";

/**
 * A ticking clock, for elapsed-time readouts on in-flight work.
 *
 * Pass `active: false` once the run finishes so a completed page stops
 * re-rendering every second.
 */
export function useNow(intervalMs = 1000, active = true): number {
  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs, active]);

  return now;
}
