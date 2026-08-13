"use client";

/**
 * The dashboard run list.
 *
 * Polls only while at least one run is still moving; a page full of finished
 * runs settles into zero network traffic.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, asApiError, listRuns } from "@/lib/api";
import { deriveRunCard, type RunCard } from "@/lib/runCard";
import { isRunActive } from "@/lib/labels";

export interface UseRunsResult {
  runs: RunCard[];
  error: ApiError | null;
  loading: boolean;
  refreshing: boolean;
  refresh: () => void;
}

const ACTIVE_POLL_MS = 4000;

export function useRuns(options: { project?: string; limit?: number } = {}): UseRunsResult {
  const { project, limit = 50 } = options;

  const [runs, setRuns] = useState<RunCard[]>([]);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [nonce, setNonce] = useState(0);

  const anyActive = useRef(false);
  const mounted = useRef(true);

  const refresh = useCallback(() => setNonce((value) => value + 1), []);

  const load = useCallback(
    async (signal?: AbortSignal) => {
      setRefreshing(true);
      try {
        const rows = await listRuns({ project, limit, signal });
        if (!mounted.current || signal?.aborted) return;
        const cards = rows.map(deriveRunCard);
        setRuns(cards);
        setError(null);
        anyActive.current = cards.some((card) => isRunActive(card.status));
      } catch (raw) {
        if (raw instanceof DOMException && raw.name === "AbortError") return;
        if (!mounted.current) return;
        setError(asApiError(raw));
      } finally {
        if (mounted.current && !signal?.aborted) {
          setRefreshing(false);
          setLoading(false);
        }
      }
    },
    [project, limit],
  );

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    void load(controller.signal);
    return () => {
      mounted.current = false;
      controller.abort();
    };
  }, [load, nonce]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (anyActive.current) void load();
    }, ACTIVE_POLL_MS);
    return () => clearInterval(interval);
  }, [load]);

  return { runs, error, loading, refreshing, refresh };
}
