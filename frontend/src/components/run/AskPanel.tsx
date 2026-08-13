"use client";

import { useCallback, useState } from "react";

import { Badge, Chip } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { TextArea } from "@/components/ui/Field";
import { Panel } from "@/components/ui/Panel";
import { ApiFailure, Spinner } from "@/components/ui/States";
import { ApiError, asApiError, askRun } from "@/lib/api";
import type { Tone } from "@/lib/labels";
import type { QuestionAnswer } from "@/types/api";

/**
 * Natural-language questions about this run.
 *
 * The evidence list is rendered as prominently as the answer, because the
 * endpoint's contract is that answers are grounded in the run's own events,
 * metrics, and decisions. An answer with an empty evidence list is shown as
 * exactly that — unevidenced — rather than being dressed up.
 */

const CONFIDENCE_TONE: Record<string, Tone> = {
  high: "good",
  medium: "warning",
  low: "serious",
};

const STARTERS = [
  "Why was this model chosen over the alternatives?",
  "What is the biggest weakness of this result?",
  "Which features drive the prediction, and by how much?",
  "Was any data dropped, and why?",
];

export function AskPanel({ runId }: { runId: string }) {
  const [question, setQuestion] = useState("");
  const [answers, setAnswers] = useState<QuestionAnswer[]>([]);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const ask = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      setError(null);
      setAsking(true);
      try {
        const answer = await askRun(runId, trimmed);
        // Newest first: the answer just asked for should not require scrolling.
        setAnswers((previous) => [answer, ...previous]);
        setQuestion("");
      } catch (raw) {
        setError(asApiError(raw));
      } finally {
        setAsking(false);
      }
    },
    [runId],
  );

  return (
    <Panel
      title="Ask about this run"
      subtitle="Answers are grounded in this run's events, metrics, and recorded decisions"
      bodyClassName="space-y-3"
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void ask(question);
        }}
      >
        <label htmlFor="ask-input" className="block text-xs font-medium text-ink-2">
          Your question
        </label>
        <div className="mt-1.5">
          <TextArea
            id="ask-input"
            rows={2}
            value={question}
            disabled={asking}
            placeholder="Why did the tuned model beat the baseline?"
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends; Shift+Enter keeps the newline for multi-part questions.
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void ask(question);
              }
            }}
          />
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Button type="submit" variant="primary" size="sm" disabled={asking || !question.trim()}>
            {asking ? "Asking…" : "Ask"}
          </Button>
          {asking ? <Spinner label="Waiting for the answer" /> : null}
          <span className="text-[11px] text-ink-3">Enter to send · Shift+Enter for a new line</span>
        </div>
      </form>

      {!answers.length && !asking ? (
        <div className="flex flex-wrap gap-1.5">
          {STARTERS.map((starter) => (
            <button
              key={starter}
              type="button"
              onClick={() => setQuestion(starter)}
              className="rounded-md border border-hairline bg-surface-2 px-2 py-1 text-[11px] text-ink-2 hover:text-ink"
            >
              {starter}
            </button>
          ))}
        </div>
      ) : null}

      {error ? <ApiFailure error={error} context="Asking the question" /> : null}

      {answers.length ? (
        <ul className="space-y-3">
          {answers.map((answer, index) => (
            <li
              key={`${index}-${answer.question.slice(0, 24)}`}
              className="rounded-lg border border-hairline bg-surface-2 px-3.5 py-3"
            >
              <div className="flex flex-wrap items-start justify-between gap-2">
                <p className="prose-agent min-w-0 text-xs font-semibold text-ink">
                  {answer.question}
                </p>
                <Badge tone={CONFIDENCE_TONE[answer.confidence] ?? "neutral"}>
                  {answer.confidence} confidence
                </Badge>
              </div>

              <p className="prose-agent mt-2 whitespace-pre-wrap text-xs leading-relaxed text-ink">
                {answer.answer}
              </p>

              <div className="mt-2">
                <p className="text-[10px] font-semibold uppercase tracking-wide text-ink-3">
                  Evidence
                </p>
                {answer.evidence.length ? (
                  <ul className="mt-1 space-y-0.5">
                    {answer.evidence.map((item, evidenceIndex) => (
                      <li
                        key={evidenceIndex}
                        className="flex gap-2 text-[11px] leading-relaxed text-ink-2"
                      >
                        <span aria-hidden="true" className="mt-1.5 size-1 shrink-0 rounded-full bg-accent" />
                        <span className="prose-agent">{item}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-1 text-[11px] text-ink-3">
                    No evidence was cited for this answer — treat it as unverified.
                  </p>
                )}
              </div>

              {answer.caveats.length ? (
                <p className="mt-2 text-[11px] leading-relaxed text-ink-3">
                  Caveats: {answer.caveats.join(" · ")}
                </p>
              ) : null}

              {answer.suggested_followups.length ? (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {answer.suggested_followups.map((followup) => (
                    <button
                      key={followup}
                      type="button"
                      onClick={() => setQuestion(followup)}
                      className="rounded-md border border-hairline bg-surface px-2 py-1 text-[11px] text-ink-2 hover:text-ink"
                    >
                      {followup}
                    </button>
                  ))}
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}

      {answers.length ? (
        <p className="text-[11px] text-ink-3">
          <Chip>{answers.length} asked this session</Chip> Questions are not persisted between
          page loads.
        </p>
      ) : null}
    </Panel>
  );
}
