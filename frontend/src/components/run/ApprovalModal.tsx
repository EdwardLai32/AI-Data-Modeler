"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Badge, Chip } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Field, TextArea, TextInput } from "@/components/ui/Field";
import { Panel } from "@/components/ui/Panel";
import { ApiFailure } from "@/components/ui/States";
import { ApiError, asApiError, decideApproval } from "@/lib/api";
import { humanise, integer, timestamp } from "@/lib/format";
import { agentLabel, severityTone } from "@/lib/labels";
import type { ApprovalRequest } from "@/types/api";

/**
 * Human approval for a destructive step.
 *
 * The run is genuinely suspended while this is open, so the dialog states
 * exactly what will be destroyed — which columns, how many rows — before asking.
 * Escape dismisses the dialog but not the decision: the run stays suspended and
 * a banner offers it back, because silently resuming would be the one
 * unrecoverable behaviour here.
 */

const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function ApprovalModal({
  runId,
  request,
  onClose,
  onResolved,
}: {
  runId: string;
  request: ApprovalRequest;
  onClose: () => void;
  onResolved: () => void;
}) {
  const dialog = useRef<HTMLDivElement>(null);
  const [note, setNote] = useState("");
  const [decidedBy, setDecidedBy] = useState("");
  const [pending, setPending] = useState<"approved" | "rejected" | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const first = dialog.current?.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus();
    return () => previous?.focus?.();
  }, []);

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const nodes = Array.from(dialog.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []);
      if (!nodes.length) return;
      const first = nodes[0] as HTMLElement;
      const last = nodes[nodes.length - 1] as HTMLElement;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    },
    [onClose],
  );

  const decide = useCallback(
    async (decision: "approved" | "rejected") => {
      setError(null);
      setPending(decision);
      try {
        await decideApproval(runId, request.request_id, {
          decision,
          note: note.trim() || undefined,
          decided_by: decidedBy.trim() || undefined,
        });
        onResolved();
        onClose();
      } catch (raw) {
        setError(asApiError(raw));
      } finally {
        setPending(null);
      }
    },
    [runId, request.request_id, note, decidedBy, onResolved, onClose],
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/45 px-4 py-8 backdrop-blur-sm"
      onKeyDown={onKeyDown}
    >
      <div
        ref={dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="approval-title"
        aria-describedby="approval-summary"
        className="w-full max-w-xl rounded-xl border border-hairline bg-surface shadow-lg"
      >
        <div className="flex items-start justify-between gap-3 border-b border-hairline px-5 py-3.5">
          <div className="min-w-0">
            <h2 id="approval-title" className="text-sm font-semibold text-ink">
              Approval required before a destructive step
            </h2>
            <p className="mt-0.5 text-[11px] text-ink-3">
              The run is paused at <span className="font-mono">{request.step_id}</span> ·{" "}
              {agentLabel(request.agent)} · requested {timestamp(request.created_at)}
            </p>
          </div>
          <Badge tone={severityTone(request.severity)}>{humanise(request.severity)}</Badge>
        </div>

        <div className="space-y-3 px-5 py-4">
          <p id="approval-summary" className="prose-agent text-sm leading-relaxed text-ink">
            {request.action_summary}
          </p>

          {request.details.length ? (
            <ul className="space-y-1">
              {request.details.map((detail, index) => (
                <li key={index} className="flex gap-2 text-xs leading-relaxed text-ink-2">
                  <span aria-hidden="true" className="mt-1.5 size-1 shrink-0 rounded-full bg-baseline" />
                  <span className="prose-agent">{detail}</span>
                </li>
              ))}
            </ul>
          ) : null}

          <div className="rounded-lg border border-hairline border-l-2 border-l-warning bg-surface-2 px-3.5 py-3">
            <p className="text-xs font-semibold text-ink">What this changes</p>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {request.affected_columns.length ? (
                request.affected_columns.map((column) => (
                  <Chip key={column} mono>
                    {column}
                  </Chip>
                ))
              ) : (
                <span className="text-[11px] text-ink-3">No specific columns listed.</span>
              )}
            </div>
            {request.affected_row_estimate > 0 ? (
              <p className="tabular mt-1.5 text-[11px] text-ink-2">
                Estimated rows affected: {integer(request.affected_row_estimate)}
              </p>
            ) : null}
          </div>

          <Field
            label="Note (recorded on the approval)"
            htmlFor="approval-note"
            hint="Optional. Stored with the decision in the run's audit trail."
          >
            <TextArea
              id="approval-note"
              rows={2}
              value={note}
              aria-describedby="approval-note-hint"
              onChange={(event) => setNote(event.target.value)}
            />
          </Field>

          <Field label="Decided by" htmlFor="approval-by">
            <TextInput
              id="approval-by"
              value={decidedBy}
              placeholder="optional — your name"
              onChange={(event) => setDecidedBy(event.target.value)}
            />
          </Field>

          {error ? <ApiFailure error={error} context="Submitting the decision" /> : null}
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2 border-t border-hairline px-5 py-3">
          <Button variant="ghost" onClick={onClose} disabled={pending !== null}>
            Decide later
          </Button>
          <Button
            variant="danger"
            onClick={() => void decide("rejected")}
            disabled={pending !== null}
          >
            {pending === "rejected" ? "Rejecting…" : "Reject and skip"}
          </Button>
          <Button
            variant="primary"
            onClick={() => void decide("approved")}
            disabled={pending !== null}
          >
            {pending === "approved" ? "Approving…" : "Approve and resume"}
          </Button>
        </div>
      </div>
    </div>
  );
}

/** The audit trail of decisions already taken on this run. */
export function ApprovalHistory({ approvals }: { approvals: ApprovalRequest[] }) {
  const resolved = approvals.filter((approval) => approval.decision !== "pending");
  if (!resolved.length) return null;

  return (
    <Panel title="Approval history" subtitle={`${resolved.length} decision${resolved.length === 1 ? "" : "s"}`}>
      <ul className="space-y-2">
        {resolved.map((approval) => (
          <li key={approval.request_id} className="border-l-2 border-baseline pl-3">
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge tone={approval.decision === "approved" ? "good" : "critical"}>
                {humanise(approval.decision)}
              </Badge>
              <Chip mono>{approval.step_id}</Chip>
              <span className="text-[11px] text-ink-3">
                {timestamp(approval.decided_at)}
                {approval.decided_by ? ` · ${approval.decided_by}` : ""}
              </span>
            </div>
            <p className="prose-agent mt-1 text-xs leading-relaxed text-ink-2">
              {approval.action_summary}
            </p>
            {approval.note ? (
              <p className="mt-0.5 text-[11px] italic text-ink-3">“{approval.note}”</p>
            ) : null}
          </li>
        ))}
      </ul>
    </Panel>
  );
}
