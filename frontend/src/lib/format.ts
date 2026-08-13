/**
 * Display formatting.
 *
 * Every function here takes a value that came from the API and returns a
 * string. When the value is missing the result is an explicit placeholder
 * (`"—"`), never a zero or a guess — a blank metric and a metric of 0.0 mean
 * very different things to someone reading a leaderboard.
 */

export const EMPTY = "—";

/** Compact a count for a stat tile: 1,284 / 12.9K / 4.2M. */
export function compactNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  const abs = Math.abs(value);
  if (abs < 1000) return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (abs < 1_000_000) return `${trimZero(value / 1000)}K`;
  if (abs < 1_000_000_000) return `${trimZero(value / 1_000_000)}M`;
  return `${trimZero(value / 1_000_000_000)}B`;
}

function trimZero(value: number): string {
  const rendered = value.toFixed(1);
  return rendered.endsWith(".0") ? rendered.slice(0, -2) : rendered;
}

/** Thousands-separated integer, for table columns. */
export function integer(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  return Math.round(value).toLocaleString("en-US");
}

/**
 * A metric score. Scores live on wildly different scales (roc_auc 0-1, rmse in
 * target units), so significant digits are chosen from the magnitude rather
 * than fixed.
 */
export function metricValue(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs < 0.001) return value.toExponential(2);
  if (abs < 100) return trimTrailingZeros(value.toFixed(4));
  if (abs < 1_000_000) return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return value.toExponential(3);
}

/** 0.0400 -> 0.04, 12.0000 -> 12. Only safe on a fixed-decimal string. */
function trimTrailingZeros(fixed: string): string {
  if (!fixed.includes(".")) return fixed;
  return fixed.replace(/0+$/, "").replace(/\.$/, "");
}

/** A 0-1 fraction as a percentage. */
export function percent(
  fraction: number | null | undefined,
  digits = 1,
): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return EMPTY;
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Signed delta, for improvement figures. */
export function signed(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  const rendered = Number.isInteger(value)
    ? String(value)
    : Math.abs(value) < 1
      ? trimTrailingZeros(value.toFixed(digits))
      : value.toFixed(2);
  return value > 0 ? `+${rendered}` : rendered;
}

/** Seconds as a human duration: 940ms / 12.4s / 3m 09s / 1h 04m. */
export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return EMPTY;
  if (seconds < 0) return EMPTY;
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(Math.floor(seconds % 60)).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

/** Bytes as KB/MB/GB. */
export function bytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  if (value < 1024) return `${Math.round(value)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let scaled = value / 1024;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled < 10 ? scaled.toFixed(1) : Math.round(scaled)} ${units[unit]}`;
}

/** USD, with enough precision that a sub-cent run cost is still visible. */
export function usd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  if (value < 1000) return `$${value.toFixed(2)}`;
  return `$${value.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

/** Absolute local timestamp, for tooltips and detail rows. */
export function timestamp(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return EMPTY;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** Wall-clock only, for the dense event feed. */
export function clockTime(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return EMPTY;
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/** "4m ago" / "just now", relative to now. */
export function relativeTime(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return EMPTY;
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return EMPTY;
  const seconds = Math.round((now - then) / 1000);
  if (seconds < 0) return "just now";
  if (seconds < 45) return "just now";
  if (seconds < 90) return "1m ago";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  return timestamp(iso);
}

/** Elapsed seconds between two ISO timestamps, or from `start` until now. */
export function elapsedSeconds(
  start: string | null | undefined,
  end: string | null | undefined,
  now = Date.now(),
): number | null {
  if (!start) return null;
  const from = new Date(start).getTime();
  if (Number.isNaN(from)) return null;
  const to = end ? new Date(end).getTime() : now;
  if (Number.isNaN(to)) return null;
  return Math.max(0, (to - from) / 1000);
}

/** `snake_case` / `SCREAMING_CASE` to sentence case, for enum values. */
export function humanise(value: string | null | undefined): string {
  if (!value) return EMPTY;
  const spaced = value.replace(/[_-]+/g, " ").trim();
  if (!spaced) return EMPTY;
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** Uppercase acronym-preserving label for metric names (roc_auc -> ROC AUC). */
const METRIC_LABELS: Record<string, string> = {
  roc_auc: "ROC AUC",
  pr_auc: "PR AUC",
  average_precision: "Average precision",
  f1: "F1",
  f1_macro: "F1 (macro)",
  f1_weighted: "F1 (weighted)",
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced accuracy",
  precision: "Precision",
  recall: "Recall",
  log_loss: "Log loss",
  brier_score: "Brier score",
  matthews_corrcoef: "Matthews corr.",
  rmse: "RMSE",
  mse: "MSE",
  mae: "MAE",
  mape: "MAPE",
  smape: "SMAPE",
  medae: "Median AE",
  r2: "R²",
  explained_variance: "Explained variance",
  silhouette: "Silhouette",
  davies_bouldin: "Davies-Bouldin",
  calinski_harabasz: "Calinski-Harabasz",
};

export function metricLabel(name: string | null | undefined): string {
  if (!name) return EMPTY;
  return METRIC_LABELS[name] ?? humanise(name);
}

/** Truncate for a single-line cell, preserving whole words where possible. */
export function truncate(text: string, limit = 140): string {
  if (text.length <= limit) return text;
  const cut = text.slice(0, limit);
  const lastSpace = cut.lastIndexOf(" ");
  return `${(lastSpace > limit * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd()}…`;
}

/** Last path segment of a filesystem path, for artifact captions. */
export function basename(path: string | null | undefined): string {
  if (!path) return EMPTY;
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts.length ? (parts[parts.length - 1] as string) : path;
}
