"use client";

/**
 * Backend reachability probe.
 *
 * Distinguishing "the API is not running" from "the API returned an error" is
 * worth a dedicated hook: the first is the overwhelmingly common first-run
 * experience and deserves the `amla serve` hint, not a generic failure state.
 * While unreachable the probe keeps retrying, so the page recovers on its own
 * the moment the server comes up.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { asApiError, getHealth } from "@/lib/api";
import type { HealthResponse } from "@/types/api";

export type BackendState = "checking" | "up" | "error" | "unreachable";

export interface UseHealthResult {
  state: BackendState;
  health: HealthResponse | null;
  error: string | null;
  recheck: () => void;
}

const RETRY_MS = 5000;

export function useHealth(): UseHealthResult {
  const [state, setState] = useState<BackendState>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const recheck = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const probe = async () => {
      try {
        const response = await getHealth(controller.signal);
        if (cancelled) return;
        setHealth(response);
        setError(null);
        setState("up");
      } catch (raw) {
        if (raw instanceof DOMException && raw.name === "AbortError") return;
        if (cancelled) return;
        const failure = asApiError(raw);
        setHealth(null);
        setError(failure.detail);
        setState(failure.unreachable ? "unreachable" : "error");
        // Only an unreachable API is worth retrying blind; a 500 needs a fix.
        if (failure.unreachable) {
          timer.current = setTimeout(() => {
            if (!cancelled) void probe();
          }, RETRY_MS);
        }
      }
    };

    void probe();

    return () => {
      cancelled = true;
      controller.abort();
      if (timer.current !== null) clearTimeout(timer.current);
      timer.current = null;
    };
  }, [nonce]);

  return { state, health, error, recheck };
}
