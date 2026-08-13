"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { KeyValue, Panel } from "@/components/ui/Panel";
import { StatTile } from "@/components/ui/Metrics";
import { BulletList, Disclosure, Rationale } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { humanise, integer, metricLabel, metricValue, percent, signed } from "@/lib/format";
import { gradeTone } from "@/lib/labels";
import type { Tone } from "@/lib/labels";
import type { EvaluationVerdict } from "@/types/api";

/**
 * The quality gate.
 *
 * The verdict is shown with the diagnosis that produced it: train/validation/
 * test scores side by side, the gap between them, calibration, and the
 * bootstrap intervals. A grade with no numbers under it is an opinion; this
 * panel exists so it is not one.
 */

const FIT_TONE: Record<EvaluationVerdict["bias_variance"]["verdict"], Tone> = {
  good_fit: "good",
  overfitting: "serious",
  underfitting: "warning",
  inconclusive: "neutral",
};

const DRIFT_TONE: Record<EvaluationVerdict["drift_risk"], Tone> = {
  low: "good",
  medium: "warning",
  high: "critical",
  unknown: "neutral",
};

const ACTION_TONE: Record<EvaluationVerdict["recommended_action"], Tone> = {
  accept: "good",
  retry_feature_engineering: "warning",
  retry_model_selection: "warning",
  retry_cleaning: "warning",
  collect_more_data: "serious",
  reject: "critical",
};

export function EvaluationPanel({ evaluation }: { evaluation: EvaluationVerdict | null }) {
  if (!evaluation) {
    return (
      <Panel title="Evaluation verdict" subtitle="Is this model fit to recommend?">
        <EmptyState
          title="Not evaluated yet"
          hint="The Evaluation agent grades the winning model against measured diagnostics before any report is written."
        />
      </Panel>
    );
  }

  const bv = evaluation.bias_variance;
  const calibration = evaluation.calibration;

  return (
    <Panel
      title="Evaluation verdict"
      subtitle="Grade, generalisation diagnosis, and the recommended next move"
      aside={
        <span className="flex flex-wrap items-center justify-end gap-1.5">
          <Badge tone={gradeTone(evaluation.overall_grade)}>
            Grade {evaluation.overall_grade}
          </Badge>
          <Badge tone={evaluation.acceptable ? "good" : "critical"}>
            {evaluation.acceptable ? "Fit to recommend" : "Not fit to recommend"}
          </Badge>
        </span>
      }
      bodyClassName="space-y-3"
    >
      <Rationale label="Verdict">{evaluation.verdict_rationale}</Rationale>

      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
        <StatTile label="Train" value={metricValue(bv.train_score)} />
        <StatTile label="Validation" value={metricValue(bv.validation_score)} />
        <StatTile label="Test" value={metricValue(bv.test_score)} />
        <StatTile label="Train − test gap" value={signed(bv.gap)} />
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={FIT_TONE[bv.verdict] ?? "neutral"}>{humanise(bv.verdict)}</Badge>
        <Badge tone={DRIFT_TONE[evaluation.drift_risk] ?? "neutral"}>
          {humanise(evaluation.drift_risk)} drift risk
        </Badge>
        <Badge tone={ACTION_TONE[evaluation.recommended_action] ?? "neutral"}>
          {humanise(evaluation.recommended_action)}
        </Badge>
        {calibration.applicable ? (
          <Chip title={calibration.verdict}>
            Brier {metricValue(calibration.brier_score)} · ECE{" "}
            {metricValue(calibration.expected_calibration_error)}
          </Chip>
        ) : null}
      </div>

      {bv.detail ? (
        <p className="prose-agent text-xs leading-relaxed text-ink-2">{bv.detail}</p>
      ) : null}

      {evaluation.action_rationale ? (
        <Rationale label="Why that action">{evaluation.action_rationale}</Rationale>
      ) : null}

      {evaluation.confidence_intervals.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Confidence intervals</h3>
          <ul className="mt-1.5 space-y-1">
            {evaluation.confidence_intervals.map((interval, index) => (
              <li
                key={`${interval.metric}-${index}`}
                className="tabular flex flex-wrap items-baseline gap-2 text-xs text-ink-2"
              >
                <span className="font-medium text-ink">{metricLabel(interval.metric)}</span>
                <span>{metricValue(interval.point_estimate)}</span>
                <span className="text-ink-3">
                  [{metricValue(interval.lower)}, {metricValue(interval.upper)}]
                </span>
                <Chip>
                  {percent(interval.level, 0)} {interval.method}
                </Chip>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {evaluation.fairness_slices.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Fairness slices</h3>
          <div className="mt-1.5 overflow-x-auto rounded-lg border border-hairline">
            <table className="w-full text-left text-[11px]">
              <thead className="bg-surface-2 text-ink-3">
                <tr>
                  <th className="px-2.5 py-1.5 font-medium">Attribute</th>
                  <th className="px-2.5 py-1.5 font-medium">Slice</th>
                  <th className="px-2.5 py-1.5 text-right font-medium">Rows</th>
                  <th className="px-2.5 py-1.5 text-right font-medium">Metric</th>
                  <th className="px-2.5 py-1.5 text-right font-medium">vs overall</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hairline">
                {evaluation.fairness_slices.map((slice, index) => (
                  <tr key={`${slice.attribute}-${slice.slice_value}-${index}`} className="text-ink-2">
                    <td className="px-2.5 py-1 font-mono">{slice.attribute}</td>
                    <td className="px-2.5 py-1">{slice.slice_value}</td>
                    <td className="tabular px-2.5 py-1 text-right">{integer(slice.n_rows)}</td>
                    <td className="tabular px-2.5 py-1 text-right">
                      {metricValue(slice.metric_value)}
                      <span className="ml-1 text-ink-3">{metricLabel(slice.metric_name)}</span>
                    </td>
                    <td className="tabular px-2.5 py-1 text-right">
                      {signed(slice.delta_vs_overall)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {evaluation.fairness_notes ? (
            <p className="prose-agent mt-1.5 text-xs leading-relaxed text-ink-2">
              {evaluation.fairness_notes}
            </p>
          ) : null}
        </div>
      ) : null}

      {evaluation.weaknesses.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Weaknesses</h3>
          <BulletList items={evaluation.weaknesses} className="mt-1.5" />
        </div>
      ) : null}

      {evaluation.specific_improvements.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Specific improvements</h3>
          <BulletList items={evaluation.specific_improvements} className="mt-1.5" />
        </div>
      ) : null}

      {evaluation.error_analysis.length ? (
        <Disclosure summary="Error analysis" count={evaluation.error_analysis.length}>
          <BulletList items={evaluation.error_analysis} />
        </Disclosure>
      ) : null}

      {evaluation.generalisation_notes ||
      evaluation.residual_notes ||
      evaluation.learning_curve_notes ||
      evaluation.drift_rationale ||
      calibration.verdict ? (
        <Disclosure summary="Diagnostic notes">
          <dl className="grid gap-2.5">
            {evaluation.generalisation_notes ? (
              <KeyValue label="Generalisation" wrap>{evaluation.generalisation_notes}</KeyValue>
            ) : null}
            {evaluation.residual_notes ? (
              <KeyValue label="Residuals" wrap>{evaluation.residual_notes}</KeyValue>
            ) : null}
            {evaluation.learning_curve_notes ? (
              <KeyValue label="Learning curve" wrap>{evaluation.learning_curve_notes}</KeyValue>
            ) : null}
            {evaluation.drift_rationale ? (
              <KeyValue label="Drift" wrap>{evaluation.drift_rationale}</KeyValue>
            ) : null}
            {calibration.verdict ? (
              <KeyValue label="Calibration" wrap>{calibration.verdict}</KeyValue>
            ) : null}
          </dl>
        </Disclosure>
      ) : null}
    </Panel>
  );
}
