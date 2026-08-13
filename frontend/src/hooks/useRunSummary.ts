"use client";

/**
 * The run blackboard, kept current while a run is in flight.
 *
 * The event stream says *that* something happened; this says *what the state
 * now is*. Refreshes are therefore event-driven (an `agent_decision` means a
 * new panel can be filled in) with a slow poll underneath as a safety net for
 * anything the stream did not announce. Refresh requests coalesce, so a burst
 * of ten events costs one request.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, asApiError, getRun } from "@/lib/api";
import { TERMINAL_RUN_STATUSES } from "@/types/api";
import type { RunSummary } from "@/types/api";

export interface UseRunSummaryResult {
  summary: RunSummary | null;
  error: ApiError | null;
  /** True only for the first load, so the UI can show a skeleton once. */
  loading: boolean;
  /** True while a background refresh is in flight. */
  refreshing: boolean;
  /** Coalesced refresh request. Safe to call on every incoming event. */
  requestRefresh: () => void;
  /** Immediate refresh, for explicit user action. */
  refreshNow: () => void;
}

const DEBOUNCE_MS = 700;

export function useRunSummary(
  runId: string | null,
  options: { pollMs?: number } = {},
): UseRunSummaryResult {
  const { pollMs = 6000 } = options;

  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState<boolean>(Boolean(runId));
  const [refreshing, setRefreshing] = useState(false);

  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlight = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  const finished = useRef(false);

  const load = useCallback(async () => {
    if (!runId || !mounted.current) return;
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setRefreshing(true);
    try {
      const next = await getRun(runId, controller.signal);
      if (!mounted.current || controller.signal.aborted) return;
      setSummary(next);
      setError(null);
      finished.current = TERMINAL_RUN_STATUSES.has(next.status);
    } catch (raw) {
      if (raw instanceof DOMException && raw.name === "AbortError") return;
      if (!mounted.current) return;
      setError(asApiError(raw));
    } finally {
      if (mounted.current && !controller.signal.aborted) {
        setRefreshing(false);
        setLoading(false);
      }
    }
  }, [runId]);

  const requestRefresh = useCallback(() => {
    if (debounce.current !== null) return;
    debounce.current = setTimeout(() => {
      debounce.current = null;
      void load();
    }, DEBOUNCE_MS);
  }, [load]);

  const refreshNow = useCallback(() => {
    if (debounce.current !== null) {
      clearTimeout(debounce.current);
      debounce.current = null;
    }
    void load();
  }, [load]);

  useEffect(() => {
    mounted.current = true;
    finished.current = false;
    setSummary(null);
    setError(null);
    setLoading(Boolean(runId));
    void load();

    return () => {
      mounted.current = false;
      inFlight.current?.abort();
      if (debounce.current !== null) clearTimeout(debounce.current);
      debounce.current = null;
    };
  }, [runId, load]);

  // Safety-net poll. Stops itself once the run reaches a terminal status so a
  // finished run costs nothing to keep open in a tab.
  useEffect(() => {
    if (!runId || pollMs <= 0) return;
    const interval = setInterval(() => {
      if (!finished.current) void load();
    }, pollMs);
    return () => clearInterval(interval);
  }, [runId, pollMs, load]);

  return { summary, error, loading, refreshing, requestRefresh, refreshNow };
}
