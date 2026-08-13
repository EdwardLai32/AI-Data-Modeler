"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { AgentTag, Badge, Chip } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { Disclosure } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { clockTime, duration, integer, usd } from "@/lib/format";
import { agentLabel, eventLabel, eventTone, isReasoningEvent, TONE_RULE } from "@/lib/labels";
import type { StreamState, StreamTransport } from "@/hooks/useRunStream";
import type { RunEvent } from "@/types/api";

/**
 * The live event feed.
 *
 * Agent reasoning is rendered as prose with the agent's name attached, while
 * machine logs stay a dense mono line — the visual difference is the point of
 * the panel, because "why" and "what" are read differently.
 *
 * Auto-scroll follows the tail but yields the moment the reader scrolls up:
 * yanking someone back to the bottom while they are reading a decision is worse
 * than missing the newest line, so the pause is sticky until they ask for it
 * back.
 */

const BOTTOM_THRESHOLD_PX = 32;

type Filter = "all" | "reasoning" | "problems";

const FILTERS: Array<{ value: Filter; label: string; hint: string }> = [
  { value: "all", label: "Everything", hint: "Every event in sequence" },
  { value: "reasoning", label: "Reasoning", hint: "Agent thinking and decisions only" },
  { value: "problems", label: "Problems", hint: "Warnings, failures, and replans" },
];

const PROBLEM_KINDS = new Set([
  "warning",
  "step_failed",
  "run_failed",
  "step_retried",
  "replan_triggered",
]);

function connectionLabel(
  state: StreamState,
  transport: StreamTransport,
  attempts: number,
): { text: string; tone: "neutral" | "accent" | "good" | "warning" | "critical"; pulse: boolean } {
  switch (state) {
    case "backfilling":
      return { text: "Loading history", tone: "accent", pulse: true };
    case "connecting":
      return { text: "Connecting", tone: "accent", pulse: true };
    case "open":
      return transport === "poll"
        ? { text: "Polling", tone: "warning", pulse: true }
        : { text: "Live", tone: "good", pulse: true };
    case "reconnecting":
      return {
        text: attempts > 1 ? `Reconnecting (attempt ${attempts})` : "Reconnecting",
        tone: "warning",
        pulse: true,
      };
    case "unreachable":
      return { text: "API unreachable", tone: "critical", pulse: false };
    case "closed":
      return { text: "Stream closed", tone: "neutral", pulse: false };
    default:
      return { text: "Idle", tone: "neutral", pulse: false };
  }
}

function EventRow({ event }: { event: RunEvent }) {
  const tone = eventTone(event.kind);
  const reasoning = isReasoningEvent(event.kind);
  const hasUsage =
    event.tokens_in !== null || event.tokens_out !== null || event.cost_usd !== null;

  if (reasoning) {
    return (
      <li className={`border-l-2 py-2 pl-3 ${TONE_RULE[tone]}`}>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <AgentTag>{agentLabel(event.agent)}</AgentTag>
          <span className="text-[11px] font-semibold uppercase tracking-wide text-ink-3">
            {eventLabel(event.kind)}
          </span>
          {event.step_id ? <Chip mono>{event.step_id}</Chip> : null}
          <span className="tabular ml-auto text-[11px] text-ink-3">{clockTime(event.at)}</span>
        </div>
        <p className="prose-agent mt-1 whitespace-pre-wrap text-[13px] leading-relaxed text-ink">
          {event.message}
        </p>
        {event.duration_seconds !== null || hasUsage ? (
          <p className="tabular mt-1 flex flex-wrap gap-x-3 text-[11px] text-ink-3">
            {event.duration_seconds !== null ? (
              <span>{duration(event.duration_seconds)}</span>
            ) : null}
            {event.tokens_in !== null ? <span>{integer(event.tokens_in)} in</span> : null}
            {event.tokens_out !== null ? <span>{integer(event.tokens_out)} out</span> : null}
            {event.cache_read_tokens ? (
              <span>{integer(event.cache_read_tokens)} cached</span>
            ) : null}
            {event.cost_usd !== null ? <span>{usd(event.cost_usd)}</span> : null}
          </p>
        ) : null}
        {event.payload.length ? (
          <Disclosure summary="Payload" count={event.payload.length}>
            <dl className="grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-3 gap-y-0.5">
              {event.payload.map((param, index) => (
                <div key={`${param.key}-${index}`} className="col-span-2 grid grid-cols-subgrid">
                  <dt className="font-mono text-[11px] text-ink-3">{param.key}</dt>
                  <dd className="break-words font-mono text-[11px] text-ink-2">{param.value}</dd>
                </div>
              ))}
            </dl>
          </Disclosure>
        ) : null}
      </li>
    );
  }

  return (
    <li className="flex items-baseline gap-2 py-1">
      <span className="tabular shrink-0 font-mono text-[11px] text-ink-3">
        {clockTime(event.at)}
      </span>
      <Badge tone={tone}>{eventLabel(event.kind)}</Badge>
      <span className="min-w-0 flex-1 break-words text-xs leading-relaxed text-ink-2">
        {event.message || "—"}
        {event.duration_seconds !== null ? (
          <span className="tabular ml-2 text-ink-3">{duration(event.duration_seconds)}</span>
        ) : null}
        {event.cost_usd !== null ? (
          <span className="tabular ml-2 text-ink-3">{usd(event.cost_usd)}</span>
        ) : null}
      </span>
    </li>
  );
}

export function EventFeed({
  events,
  state,
  transport,
  attempts,
  error,
  onReconnect,
}: {
  events: RunEvent[];
  state: StreamState;
  transport: StreamTransport;
  attempts: number;
  error: string | null;
  onReconnect: () => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [pinned, setPinned] = useState(true);
  const [seen, setSeen] = useState(0);
  const scroller = useRef<HTMLDivElement>(null);

  const visible = useMemo(() => {
    if (filter === "reasoning") return events.filter((event) => isReasoningEvent(event.kind));
    if (filter === "problems") return events.filter((event) => PROBLEM_KINDS.has(event.kind));
    return events;
  }, [events, filter]);

  useEffect(() => {
    const element = scroller.current;
    if (!element || !pinned) return;
    element.scrollTop = element.scrollHeight;
    setSeen(events.length);
  }, [events.length, filter, pinned]);

  const unseen = pinned ? 0 : Math.max(0, events.length - seen);
  const connection = connectionLabel(state, transport, attempts);

  return (
    <Panel
      title="Live event feed"
      subtitle="Agent reasoning, decisions, and orchestrator logs in sequence order"
      aside={
        <div className="flex flex-wrap items-center justify-end gap-1.5">
          <Badge tone={connection.tone} pulse={connection.pulse}>
            {connection.text}
          </Badge>
          <Chip title="Highest event sequence received">
            {events.length ? `seq ${events[events.length - 1]?.sequence ?? 0}` : "no events"}
          </Chip>
          {state === "unreachable" || state === "closed" ? (
            <Button size="sm" onClick={onReconnect}>
              Reconnect
            </Button>
          ) : null}
        </div>
      }
      bodyClassName="pt-3"
    >
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div role="group" aria-label="Filter events" className="flex flex-wrap gap-1">
          {FILTERS.map((option) => (
            <button
              key={option.value}
              type="button"
              aria-pressed={filter === option.value}
              title={option.hint}
              onClick={() => setFilter(option.value)}
              className={`rounded-md border px-2 py-1 text-[11px] font-medium transition-colors ${
                filter === option.value
                  ? "border-accent bg-accent-wash text-ink"
                  : "border-hairline bg-surface-2 text-ink-3 hover:text-ink"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
        {!pinned ? (
          <Button
            size="sm"
            variant="primary"
            onClick={() => {
              setPinned(true);
              const element = scroller.current;
              if (element) element.scrollTop = element.scrollHeight;
            }}
          >
            {unseen ? `Jump to latest (${unseen} new)` : "Jump to latest"}
          </Button>
        ) : null}
      </div>

      {error ? (
        <p className="mb-2 rounded-md border border-hairline border-l-2 border-l-warning bg-surface-2 px-3 py-2 text-[11px] leading-relaxed text-ink-2">
          {error}
        </p>
      ) : null}

      <div
        ref={scroller}
        onScroll={(event) => {
          const element = event.currentTarget;
          const atBottom =
            element.scrollHeight - element.scrollTop - element.clientHeight <=
            BOTTOM_THRESHOLD_PX;
          if (atBottom !== pinned) setPinned(atBottom);
        }}
        tabIndex={0}
        role="log"
        aria-live="polite"
        aria-atomic="false"
        aria-label="Run event feed"
        className="max-h-[32rem] min-h-[12rem] overflow-y-auto rounded-lg border border-hairline bg-surface-2/50 px-3 py-2"
      >
        {!visible.length ? (
          <div className="py-6">
            <EmptyState
              title={events.length ? "Nothing matches this filter" : "No events yet"}
              hint={
                events.length
                  ? "Switch back to Everything to see the full log."
                  : "Events appear the moment the orchestrator emits them."
              }
            />
          </div>
        ) : (
          <ul className="divide-y divide-hairline">
            {visible.map((event) => (
              <EventRow key={event.event_id || `seq-${event.sequence}`} event={event} />
            ))}
          </ul>
        )}
      </div>

      <p className="mt-2 text-[11px] text-ink-3">
        {pinned
          ? "Following the tail. Scroll up to pause."
          : "Auto-scroll paused while you read."}
      </p>
    </Panel>
  );
}
