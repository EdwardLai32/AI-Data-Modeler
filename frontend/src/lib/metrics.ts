/**
 * Leaderboard ordering.
 *
 * The backend is authoritative about direction: `ExperimentLog.higher_is_better`
 * comes from `execution/metrics.higher_is_better`. The name-based fallback here
 * only runs when that flag is absent (an older summary, or a projected list
 * row), and it errs toward "higher is better" because that is the majority case.
 */

import type { ExperimentLog, ExperimentResult } from "@/types/api";

/** Metrics where a smaller number is a better model. */
const LOWER_IS_BETTER = [
  "rmse",
  "mse",
  "mae",
  "mape",
  "smape",
  "msle",
  "rmsle",
  "medae",
  "median_absolute_error",
  "mean_absolute_error",
  "mean_squared_error",
  "mean_squared_log_error",
  "log_loss",
  "logloss",
  "cross_entropy",
  "brier",
  "brier_score",
  "hamming_loss",
  "zero_one_loss",
  "davies_bouldin",
  "error",
  "loss",
];

/** Fallback direction inference from a metric name. */
export function higherIsBetter(metric: string | null | undefined): boolean {
  if (!metric) return true;
  const name = metric.toLowerCase().replace(/^neg_/, "");
  if (metric.toLowerCase().startsWith("neg_")) return true; // sklearn's neg_* are maximised
  return !LOWER_IS_BETTER.some((bad) => name === bad || name.endsWith(`_${bad}`));
}

export function logDirection(log: ExperimentLog | null | undefined): boolean {
  if (!log) return true;
  if (typeof log.higher_is_better === "boolean") return log.higher_is_better;
  return higherIsBetter(log.primary_metric);
}

/**
 * Successful experiments ordered best-first by the primary metric.
 *
 * Failures are returned separately rather than dropped: a family that could not
 * fit is a real result the user needs to see, but it must not occupy a rank.
 */
export function rankExperiments(log: ExperimentLog | null | undefined): {
  ranked: ExperimentResult[];
  failed: ExperimentResult[];
  best: ExperimentResult | null;
  direction: boolean;
} {
  const direction = logDirection(log);
  const results = log?.results ?? [];
  const scored = results.filter(
    (row) => !row.failed && row.primary_score !== null && Number.isFinite(row.primary_score),
  );
  const failed = results.filter(
    (row) => row.failed || row.primary_score === null || !Number.isFinite(row.primary_score),
  );
  const ranked = [...scored].sort((a, b) => {
    const left = a.primary_score as number;
    const right = b.primary_score as number;
    return direction ? right - left : left - right;
  });

  // The backend names the winner; trust it when the id resolves, because it may
  // have applied a tie-break the frontend cannot see (cost, train time).
  const named = results.find((row) => row.experiment_id === log?.best_experiment_id);
  const best = named && !named.failed ? named : (ranked[0] ?? null);
  return { ranked, failed, best, direction };
}

/**
 * Bar length for a score, on a scale that starts at the worst ranked score
 * rather than at zero.
 *
 * Anchoring to zero makes every ROC AUC bar look ~90% full and hides the
 * differences that matter; anchoring to the observed range shows them. The bar
 * is therefore a *relative* comparison within this leaderboard, and each bar
 * carries its literal score as a direct label so the number is never implied by
 * length alone.
 */
export function relativeBarFraction(
  score: number,
  scores: number[],
  direction: boolean,
): number {
  const finite = scores.filter((value) => Number.isFinite(value));
  if (!finite.length) return 0;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (max === min) return 1;
  const fraction = direction ? (score - min) / (max - min) : (max - score) / (max - min);
  // A floor keeps the worst row visible as a mark rather than nothing at all.
  return 0.08 + 0.92 * Math.min(1, Math.max(0, fraction));
}

/** Standard deviation of the cross-validation fold scores, when present. */
export function cvSpread(result: ExperimentResult): number | null {
  const scores = result.cv_scores?.filter((value) => Number.isFinite(value)) ?? [];
  if (scores.length < 2) return null;
  const mean = scores.reduce((sum, value) => sum + value, 0) / scores.length;
  const variance =
    scores.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (scores.length - 1);
  return Math.sqrt(variance);
}
