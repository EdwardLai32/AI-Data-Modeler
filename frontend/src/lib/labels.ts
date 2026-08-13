/**
 * Human labels and status tones for the schema enums.
 *
 * Kept in one place so a status colour is decided once. Status hues are
 * reserved (good / warning / serious / critical) and are always paired with a
 * text label in the UI, never used as the only signal.
 */

import type {
  AgentName,
  ChartKind,
  EventKind,
  ModelFamily,
  RunStatus,
  Severity,
  StepStatus,
  TaskType,
  TuningMethod,
} from "@/types/api";
import { humanise } from "@/lib/format";

export type Tone = "neutral" | "accent" | "good" | "warning" | "serious" | "critical";

const AGENT_LABELS: Record<AgentName, string> = {
  dataset: "Dataset agent",
  problem: "Problem agent",
  planner: "Planning agent",
  cleaning: "Cleaning agent",
  features: "Feature agent",
  model_selection: "Model selection agent",
  experiment: "Experiment agent",
  tuning: "Tuning agent",
  explain: "Explainability agent",
  evaluation: "Evaluation agent",
  insight: "Insight agent",
  visualization: "Visualization agent",
  report: "Report agent",
};

/** Short form for dense rows (steppers, event feed gutters). */
const AGENT_SHORT: Record<AgentName, string> = {
  dataset: "Dataset",
  problem: "Problem",
  planner: "Planner",
  cleaning: "Cleaning",
  features: "Features",
  model_selection: "Selection",
  experiment: "Experiment",
  tuning: "Tuning",
  explain: "Explain",
  evaluation: "Evaluation",
  insight: "Insight",
  visualization: "Charts",
  report: "Report",
};

export function agentLabel(agent: AgentName | null | undefined): string {
  if (!agent) return "Orchestrator";
  return AGENT_LABELS[agent] ?? humanise(agent);
}

export function agentShort(agent: AgentName | null | undefined): string {
  if (!agent) return "System";
  return AGENT_SHORT[agent] ?? humanise(agent);
}

const TASK_LABELS: Record<TaskType, string> = {
  binary_classification: "Binary classification",
  multiclass_classification: "Multiclass classification",
  multilabel_classification: "Multilabel classification",
  regression: "Regression",
  time_series_forecasting: "Time-series forecasting",
  recommendation: "Recommendation",
  ranking: "Ranking",
  clustering: "Clustering",
  anomaly_detection: "Anomaly detection",
  nlp: "NLP",
  computer_vision: "Computer vision",
  graph_learning: "Graph learning",
  survival_analysis: "Survival analysis",
  causal_inference: "Causal inference",
};

export function taskLabel(task: TaskType | null | undefined): string {
  if (!task) return "Task not yet identified";
  return TASK_LABELS[task] ?? humanise(task);
}

const FAMILY_LABELS: Record<ModelFamily, string> = {
  linear: "Linear regression",
  logistic: "Logistic regression",
  ridge: "Ridge",
  lasso: "Lasso",
  elastic_net: "Elastic net",
  decision_tree: "Decision tree",
  random_forest: "Random forest",
  extra_trees: "Extra trees",
  gradient_boosting: "Gradient boosting",
  hist_gradient_boosting: "Hist gradient boosting",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  svm: "SVM",
  knn: "k-nearest neighbours",
  naive_bayes: "Naive Bayes",
  neural_network: "Neural network (MLP)",
  kmeans: "k-means",
  dbscan: "DBSCAN",
  gaussian_mixture: "Gaussian mixture",
  isolation_forest: "Isolation forest",
  local_outlier_factor: "Local outlier factor",
  one_class_svm: "One-class SVM",
  seasonal_naive: "Seasonal naive",
  theta: "Theta",
  exponential_smoothing: "Exponential smoothing",
  sarimax: "SARIMAX",
  baseline_dummy: "Dummy baseline",
};

export function familyLabel(family: ModelFamily | null | undefined): string {
  if (!family) return "—";
  return FAMILY_LABELS[family] ?? humanise(family);
}

const RUN_STATUS_LABELS: Record<RunStatus, string> = {
  pending: "Pending",
  running: "Running",
  awaiting_approval: "Awaiting approval",
  replanning: "Replanning",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

const RUN_STATUS_TONES: Record<RunStatus, Tone> = {
  pending: "neutral",
  running: "accent",
  awaiting_approval: "warning",
  replanning: "warning",
  completed: "good",
  failed: "critical",
  cancelled: "neutral",
};

export function runStatusLabel(status: RunStatus): string {
  return RUN_STATUS_LABELS[status] ?? humanise(status);
}

export function runStatusTone(status: RunStatus): Tone {
  return RUN_STATUS_TONES[status] ?? "neutral";
}

export function isRunActive(status: RunStatus): boolean {
  return status === "pending" || status === "running" || status === "replanning" ||
    status === "awaiting_approval";
}

const STEP_STATUS_LABELS: Record<StepStatus, string> = {
  pending: "Pending",
  running: "Running",
  completed: "Completed",
  skipped: "Skipped",
  failed: "Failed",
  awaiting_approval: "Awaiting approval",
};

const STEP_STATUS_TONES: Record<StepStatus, Tone> = {
  pending: "neutral",
  running: "accent",
  completed: "good",
  skipped: "neutral",
  failed: "critical",
  awaiting_approval: "warning",
};

export function stepStatusLabel(status: StepStatus): string {
  return STEP_STATUS_LABELS[status] ?? humanise(status);
}

export function stepStatusTone(status: StepStatus): Tone {
  return STEP_STATUS_TONES[status] ?? "neutral";
}

const SEVERITY_TONES: Record<Severity, Tone> = {
  info: "neutral",
  low: "neutral",
  medium: "warning",
  high: "serious",
  critical: "critical",
};

export function severityTone(severity: Severity | null | undefined): Tone {
  if (!severity) return "neutral";
  return SEVERITY_TONES[severity] ?? "neutral";
}

const EVENT_LABELS: Record<EventKind, string> = {
  run_started: "Run started",
  run_completed: "Run completed",
  run_failed: "Run failed",
  run_cancelled: "Run cancelled",
  step_started: "Step started",
  step_completed: "Step completed",
  step_failed: "Step failed",
  step_skipped: "Step skipped",
  step_retried: "Step retried",
  agent_thinking: "Thinking",
  agent_decision: "Decision",
  llm_call: "LLM call",
  artifact_written: "Artifact",
  approval_requested: "Approval requested",
  approval_resolved: "Approval resolved",
  replan_triggered: "Replan",
  metric_recorded: "Metric",
  warning: "Warning",
  log: "Log",
};

export function eventLabel(kind: EventKind): string {
  return EVENT_LABELS[kind] ?? humanise(kind);
}

const EVENT_TONES: Record<EventKind, Tone> = {
  run_started: "accent",
  run_completed: "good",
  run_failed: "critical",
  run_cancelled: "neutral",
  step_started: "accent",
  step_completed: "good",
  step_failed: "critical",
  step_skipped: "neutral",
  step_retried: "warning",
  agent_thinking: "accent",
  agent_decision: "accent",
  llm_call: "neutral",
  artifact_written: "neutral",
  approval_requested: "warning",
  approval_resolved: "good",
  replan_triggered: "warning",
  metric_recorded: "accent",
  warning: "serious",
  log: "neutral",
};

export function eventTone(kind: EventKind): Tone {
  return EVENT_TONES[kind] ?? "neutral";
}

/** Agent reasoning is rendered differently from machine logs. */
export function isReasoningEvent(kind: EventKind): boolean {
  return kind === "agent_thinking" || kind === "agent_decision";
}

/** Acronyms `humanise` would sentence-case incorrectly. */
const CHART_KIND_LABELS: Partial<Record<ChartKind, string>> = {
  roc_curve: "ROC curve",
  pr_curve: "PR curve",
  shap_summary: "SHAP summary",
};

export function chartKindLabel(kind: ChartKind | null | undefined): string {
  if (!kind) return "—";
  return CHART_KIND_LABELS[kind] ?? humanise(kind);
}

const TUNING_METHOD_LABELS: Record<TuningMethod, string> = {
  none: "No search",
  grid_search: "Grid search",
  random_search: "Random search",
  bayesian: "Bayesian optimisation",
  optuna_tpe: "Optuna TPE",
  halving_random: "Halving random search",
};

export function tuningMethodLabel(method: TuningMethod | null | undefined): string {
  if (!method) return "—";
  return TUNING_METHOD_LABELS[method] ?? humanise(method);
}

const GRADE_TONES: Record<string, Tone> = {
  A: "good",
  B: "good",
  C: "warning",
  D: "serious",
  F: "critical",
};

export function gradeTone(grade: string | null | undefined): Tone {
  if (!grade) return "neutral";
  return GRADE_TONES[grade.toUpperCase()] ?? "neutral";
}

/** Tailwind classes for the coloured dot that carries a tone. */
export const TONE_DOT: Record<Tone, string> = {
  neutral: "bg-baseline",
  accent: "bg-accent",
  good: "bg-good",
  warning: "bg-warning",
  serious: "bg-serious",
  critical: "bg-critical",
};

/** Tailwind classes for a tone's left rule / border accent. */
export const TONE_RULE: Record<Tone, string> = {
  neutral: "border-baseline",
  accent: "border-accent",
  good: "border-good",
  warning: "border-warning",
  serious: "border-serious",
  critical: "border-critical",
};
