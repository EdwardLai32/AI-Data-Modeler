"use client";

/**
 * EventSource lifecycle for one run's event log.
 *
 * The contract this hook keeps is *no event is ever lost*, which needs three
 * things working together:
 *
 * 1.  **Backfill before subscribing.** A run that started before the page
 *     loaded already has history; `GET /api/runs/{id}/events?after=0` replays it.
 * 2.  **A sequence cursor, not a timestamp.** `RunEvent.sequence` is a
 *     monotonic per-run counter assigned under the bus lock, so it is the only
 *     safe resume point. Reconnects pass it as `last_sequence`, and the gap is
 *     *also* closed over REST before the socket reopens — belt and braces,
 *     because the browser may have missed events while offline.
 * 3.  **Idempotent ingest.** Sequences already seen are dropped, so a replay
 *     that overlaps the live stream cannot duplicate rows in the feed.
 *
 * If SSE never connects at all (endpoint absent, proxy stripping the stream)
 * the hook degrades to REST polling rather than showing an empty feed, and
 * reports which transport is live so the UI can say so.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { asApiError, getEvents, streamUrl } from "@/lib/api";
import { EVENT_KINDS, TERMINAL_EVENT_KINDS } from "@/types/api";
import type { RunEvent } from "@/types/api";

export type StreamState =
  | "idle"
  | "backfilling"
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed"
  | "unreachable";

export type StreamTransport = "none" | "stream" | "poll";

export interface UseRunStreamOptions {
  /** Set false to hold the connection closed (e.g. while the run id is unknown). */
  enabled?: boolean;
  /** Ring-buffer cap on retained events. The cursor still advances past them. */
  maxEvents?: number;
  /** Called once per newly seen event, in sequence order. */
  onEvent?: (event: RunEvent) => void;
}

export interface UseRunStreamResult {
  events: RunEvent[];
  lastSequence: number;
  state: StreamState;
  transport: StreamTransport;
  error: string | null;
  /** Consecutive failed connection attempts; resets to 0 on a successful open. */
  attempts: number;
  /** True once a terminal run event has arrived — the stream is closed for good. */
  finished: boolean;
  /** Force a fresh backfill + reconnect from the current cursor. */
  reconnect: () => void;
}

const MAX_BACKOFF_MS = 15_000;
const BASE_BACKOFF_MS = 1_000;
const POLL_INTERVAL_MS = 3_000;
/** Give SSE this many tries before falling back to polling. */
const FALLBACK_AFTER_ATTEMPTS = 3;

/** SSE frames the backend may send that are not run events. */
const KEEPALIVE_EVENTS = ["ping", "keepalive", "heartbeat", "comment"];

function isRunEvent(value: unknown): value is RunEvent {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<RunEvent>;
  return typeof candidate.sequence === "number" && typeof candidate.kind === "string";
}

export function useRunStream(
  runId: string | null,
  options: UseRunStreamOptions = {},
): UseRunStreamResult {
  const { enabled = true, maxEvents = 4000, onEvent } = options;

  const [events, setEvents] = useState<RunEvent[]>([]);
  const [lastSequence, setLastSequence] = useState(0);
  const [state, setState] = useState<StreamState>("idle");
  const [transport, setTransport] = useState<StreamTransport>("none");
  const [error, setError] = useState<string | null>(null);
  const [attempts, setAttempts] = useState(0);
  const [finished, setFinished] = useState(false);
  const [nonce, setNonce] = useState(0);

  // Kept in a ref so a changing callback identity does not tear down the socket.
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const reconnect = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    if (!runId || !enabled) {
      setState("idle");
      setTransport("none");
      return;
    }

    let cancelled = false;
    let source: EventSource | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let cursor = 0;
    let failures = 0;
    let openedOnce = false;
    let terminal = false;
    const seen = new Set<number>();

    setEvents([]);
    setLastSequence(0);
    setError(null);
    setAttempts(0);
    setFinished(false);

    const clearTimer = () => {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
    };

    const closeSource = () => {
      if (source) {
        source.onopen = null;
        source.onerror = null;
        source.onmessage = null;
        source.close();
        source = null;
      }
    };

    const ingest = (incoming: RunEvent[]) => {
      if (cancelled || !incoming.length) return;
      const fresh = incoming
        .filter((event) => !seen.has(event.sequence))
        .sort((a, b) => a.sequence - b.sequence);
      if (!fresh.length) return;

      for (const event of fresh) {
        seen.add(event.sequence);
        if (event.sequence > cursor) cursor = event.sequence;
        if (TERMINAL_EVENT_KINDS.has(event.kind)) terminal = true;
      }

      setEvents((previous) => {
        const merged = [...previous, ...fresh];
        return merged.length > maxEvents ? merged.slice(merged.length - maxEvents) : merged;
      });
      setLastSequence(cursor);
      for (const event of fresh) onEventRef.current?.(event);

      if (terminal) {
        clearTimer();
        closeSource();
        setFinished(true);
        setState("closed");
        setTransport("none");
      }
    };

    /** Pull anything the socket missed. Returns false when the API is down. */
    const closeGap = async (): Promise<boolean> => {
      try {
        const missed = await getEvents(runId, cursor);
        ingest(missed);
        return true;
      } catch (raw) {
        if (cancelled) return false;
        const failure = asApiError(raw);
        setError(failure.unreachable ? failure.message : failure.detail);
        return !failure.unreachable;
      }
    };

    const poll = async () => {
      if (cancelled || terminal) return;
      setTransport("poll");
      setState("open");
      const reachable = await closeGap();
      if (cancelled || terminal) return;
      if (!reachable) setState("unreachable");
      timer = setTimeout(() => void poll(), POLL_INTERVAL_MS);
    };

    const scheduleRetry = () => {
      if (cancelled || terminal) return;
      failures += 1;
      setAttempts(failures);

      if (!openedOnce && failures >= FALLBACK_AFTER_ATTEMPTS) {
        // SSE is not going to work here. Polling still shows the whole log.
        setError(
          (previous) =>
            previous ??
            "Live stream unavailable; falling back to polling the events endpoint.",
        );
        void poll();
        return;
      }

      setState("reconnecting");
      const delay = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * 2 ** (failures - 1));
      timer = setTimeout(() => {
        void (async () => {
          const reachable = await closeGap();
          if (cancelled || terminal) return;
          if (!reachable) {
            setState("unreachable");
            scheduleRetry();
            return;
          }
          connect();
        })();
      }, delay);
    };

    const connect = () => {
      if (cancelled || terminal) return;
      closeSource();
      setState(openedOnce ? "reconnecting" : "connecting");
      setTransport("stream");

      let stream: EventSource;
      try {
        stream = new EventSource(streamUrl(runId, cursor));
      } catch {
        setError(`Cannot open the event stream for ${runId}.`);
        scheduleRetry();
        return;
      }
      source = stream;

      const handleFrame = (raw: MessageEvent<string>) => {
        if (!raw.data) return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(raw.data) as unknown;
        } catch {
          return; // a keepalive comment or a non-JSON frame
        }
        if (Array.isArray(parsed)) {
          ingest(parsed.filter(isRunEvent));
        } else if (isRunEvent(parsed)) {
          ingest([parsed]);
        }
      };

      stream.onopen = () => {
        if (cancelled) return;
        openedOnce = true;
        failures = 0;
        setAttempts(0);
        setError(null);
        setState("open");
        setTransport("stream");
      };

      stream.onmessage = handleFrame;

      // sse-starlette can name the SSE `event:` field after the event kind. Both
      // conventions are handled so the feed works either way.
      for (const kind of [...EVENT_KINDS, "run_event"]) {
        stream.addEventListener(kind, handleFrame as EventListener);
      }
      for (const kind of KEEPALIVE_EVENTS) {
        stream.addEventListener(kind, () => {
          if (!cancelled) setState("open");
        });
      }

      stream.onerror = () => {
        if (cancelled || terminal) return;
        closeSource();
        scheduleRetry();
      };
    };

    const start = async () => {
      setState("backfilling");
      const reachable = await closeGap();
      if (cancelled || terminal) return;
      if (!reachable) {
        setState("unreachable");
        scheduleRetry();
        return;
      }
      connect();
    };

    void start();

    return () => {
      cancelled = true;
      clearTimer();
      closeSource();
    };
  }, [runId, enabled, maxEvents, nonce]);

  return {
    events,
    lastSequence,
    state,
    transport,
    error,
    attempts,
    finished,
    reconnect,
  };
}
