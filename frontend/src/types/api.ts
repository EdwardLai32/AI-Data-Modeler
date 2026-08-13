/**
 * TypeScript mirror of `automl_architect/core/schemas.py`.
 *
 * Field names match the Pydantic models exactly (snake_case) so a JSON body can
 * be assigned to these types without any renaming layer. Python `X | None`
 * becomes `X | null`; `datetime` becomes `string` (ISO-8601, as emitted by
 * `model_dump(mode="json")`); `list[Param]` stays a `Param[]` because the
 * agent-facing schema subset cannot express an untyped object.
 *
 * Enums are string union types rather than TS `enum`s: the wire format is the
 * string value, and unions keep the JSON assignable without conversion.
 */

// ---------------------------------------------------------------------------
// Free-form parameters
// ---------------------------------------------------------------------------

export interface Param {
  key: string;
  value: string;
}

/** Mirror of `schemas.params_to_dict` for display purposes. */
export function paramsToRecord(params: Param[] | null | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  for (const p of params ?? []) out[p.key] = p.value;
  return out;
}

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

export type TaskType =
  | "binary_classification"
  | "multiclass_classification"
  | "multilabel_classification"
  | "regression"
  | "time_series_forecasting"
  | "recommendation"
  | "ranking"
  | "clustering"
  | "anomaly_detection"
  | "nlp"
  | "computer_vision"
  | "graph_learning"
  | "survival_analysis"
  | "causal_inference";

export type ColumnKind =
  | "numeric_continuous"
  | "numeric_discrete"
  | "categorical_nominal"
  | "categorical_ordinal"
  | "boolean"
  | "datetime"
  | "text"
  | "geo"
  | "identifier"
  | "constant"
  | "unknown";

export type ColumnRole =
  | "target"
  | "feature"
  | "identifier"
  | "temporal_index"
  | "group_key"
  | "weight"
  | "leakage_suspect"
  | "ignored";

export type Severity = "info" | "low" | "medium" | "high" | "critical";

export type RunStatus =
  | "pending"
  | "running"
  | "awaiting_approval"
  | "replanning"
  | "completed"
  | "failed"
  | "cancelled";

export type StepStatus =
  | "pending"
  | "running"
  | "completed"
  | "skipped"
  | "failed"
  | "awaiting_approval";

export type AgentName =
  | "dataset"
  | "problem"
  | "planner"
  | "cleaning"
  | "features"
  | "model_selection"
  | "experiment"
  | "tuning"
  | "explain"
  | "evaluation"
  | "insight"
  | "visualization"
  | "report";

export type SourceKind =
  | "csv"
  | "excel"
  | "json"
  | "parquet"
  | "sql"
  | "postgres"
  | "mysql"
  | "snowflake"
  | "databricks"
  | "duckdb"
  | "s3"
  | "azure_blob"
  | "gcs"
  | "rest_api"
  | "kaggle"
  | "dataframe";

export type Confidence = "high" | "medium" | "low";
export type Priority = "high" | "medium" | "low";

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

export interface DataSource {
  kind: SourceKind;
  uri: string;
  query: string | null;
  options: Param[];
  secret_env: string[];
}

export interface SchemaField {
  name: string;
  inferred_dtype: string;
  nullable: boolean;
  sample_values: string[];
}

export interface IngestionResult {
  source: DataSource;
  dataset_id: string;
  n_rows: number;
  n_columns: number;
  schema_fields: SchemaField[];
  validation_errors: string[];
  validation_warnings: string[];
  bytes_in_memory: number;
  load_seconds: number;
  truncated: boolean;
}

// ---------------------------------------------------------------------------
// Profiling
// ---------------------------------------------------------------------------

export interface Quantiles {
  p01: number | null;
  p05: number | null;
  p25: number | null;
  p50: number | null;
  p75: number | null;
  p95: number | null;
  p99: number | null;
}

export interface OutlierSummary {
  method: string;
  n_outliers: number;
  fraction: number;
  lower_bound: number | null;
  upper_bound: number | null;
}

export interface CategoryCount {
  value: string;
  count: number;
  fraction: number;
}

export interface ColumnProfile {
  name: string;
  kind: ColumnKind;
  dtype: string;
  n_missing: number;
  missing_fraction: number;
  n_unique: number;
  cardinality_ratio: number;
  is_constant: boolean;
  memory_bytes: number;

  mean: number | null;
  std: number | null;
  minimum: number | null;
  maximum: number | null;
  skewness: number | null;
  kurtosis: number | null;
  zero_fraction: number | null;
  negative_fraction: number | null;
  quantiles: Quantiles | null;
  outliers: OutlierSummary | null;
  variance: number | null;
  is_near_zero_variance: boolean;

  top_values: CategoryCount[];
  mean_string_length: number | null;
  max_string_length: number | null;
  mean_token_count: number | null;

  min_timestamp: string | null;
  max_timestamp: string | null;
  inferred_frequency: string | null;
  n_gaps: number | null;
  is_monotonic: boolean | null;

  looks_like_id: boolean;
  looks_like_geo: boolean;
  looks_like_text: boolean;
  looks_like_datetime: boolean;
  detected_semantic_type: string | null;
}

export interface CorrelationPair {
  left: string;
  right: string;
  coefficient: number;
  method: string;
}

export interface TargetSummary {
  name: string;
  kind: ColumnKind;
  n_classes: number | null;
  class_counts: CategoryCount[];
  imbalance_ratio: number | null;
  is_imbalanced: boolean;
  mean: number | null;
  std: number | null;
  skewness: number | null;
  n_missing: number;
}

export interface LeakageFinding {
  column: string;
  score: number;
  method: string;
  severity: Severity;
  reason: string;
}

export interface DataQualityIssue {
  code: string;
  severity: Severity;
  columns: string[];
  detail: string;
}

export interface DatasetProfile {
  dataset_id: string;
  n_rows: number;
  n_columns: number;
  memory_bytes: number;
  n_duplicate_rows: number;
  duplicate_fraction: number;
  total_missing_cells: number;
  missing_cell_fraction: number;
  columns: ColumnProfile[];
  target: TargetSummary | null;
  top_correlations: CorrelationPair[];
  target_correlations: CorrelationPair[];
  highly_correlated_pairs: CorrelationPair[];
  leakage_findings: LeakageFinding[];
  quality_issues: DataQualityIssue[];
  temporal_columns: string[];
  geo_columns: string[];
  text_columns: string[];
  identifier_columns: string[];
  constant_columns: string[];
  profiled_at: string;
  profile_seconds: number;
}

// ---------------------------------------------------------------------------
// Agent outputs
// ---------------------------------------------------------------------------

export interface ColumnAssessment {
  name: string;
  role: ColumnRole;
  predictive_potential: "high" | "medium" | "low" | "none";
  concerns: string[];
  notes: string;
}

export interface DatasetUnderstanding {
  headline: string;
  narrative: string;
  likely_domain: string;
  grain: string;
  column_assessments: ColumnAssessment[];
  key_findings: string[];
  risks: string[];
  suggested_target_columns: string[];
  data_readiness: "ready" | "needs_cleaning" | "needs_major_work" | "unusable";
  readiness_rationale: string;
}

export interface ProblemDefinition {
  task_type: TaskType;
  target_column: string | null;
  positive_class: string | null;
  temporal_column: string | null;
  group_column: string | null;
  horizon: number | null;
  rationale: string;
  alternatives_considered: string[];
  confidence: Confidence;
  primary_metric: string;
  secondary_metrics: string[];
  metric_rationale: string;
  business_objective: string;
  constraints: string[];
}

export interface PlanStep {
  step_id: string;
  order: number;
  title: string;
  agent: AgentName;
  objective: string;
  rationale: string;
  depends_on: string[];
  optional: boolean;
  destructive: boolean;
  estimated_seconds: number;
  success_criteria: string;
}

export interface ExecutionPlan {
  summary: string;
  steps: PlanStep[];
  dataset_specific_adaptations: string[];
  risks: string[];
  fallback_strategy: string;
  revision: number;
  revision_reason: string | null;
}

export type MissingStrategy =
  | "drop_column"
  | "drop_rows"
  | "mean"
  | "median"
  | "mode"
  | "constant"
  | "forward_fill"
  | "backward_fill"
  | "interpolate"
  | "knn"
  | "iterative"
  | "missing_category"
  | "leave_as_is";

export type CleaningAction =
  | "impute_missing"
  | "drop_column"
  | "drop_duplicate_rows"
  | "drop_rows_missing_target"
  | "clip_outliers"
  | "remove_outlier_rows"
  | "cast_dtype"
  | "parse_datetime"
  | "normalise_categories"
  | "strip_whitespace"
  | "drop_constant_column"
  | "drop_leakage_column";

export interface CleaningDecision {
  action: CleaningAction;
  columns: string[];
  strategy: MissingStrategy | null;
  parameters: Param[];
  rationale: string;
  expected_impact: string;
  destructive: boolean;
  severity_if_skipped: Severity;
}

export interface CleaningPlan {
  decisions: CleaningDecision[];
  summary: string;
  columns_to_drop: string[];
  drop_rationale: string[];
  skipped_considerations: string[];
}

export type FeatureOp =
  | "date_decompose"
  | "cyclical_encode"
  | "lag"
  | "rolling"
  | "diff"
  | "expanding"
  | "interaction"
  | "ratio"
  | "polynomial"
  | "log_transform"
  | "sqrt_transform"
  | "boxcox"
  | "binning"
  | "one_hot_encode"
  | "ordinal_encode"
  | "target_encode"
  | "frequency_encode"
  | "hash_encode"
  | "text_tfidf"
  | "text_length"
  | "geo_distance"
  | "aggregate_by_group"
  | "standard_scale"
  | "minmax_scale"
  | "robust_scale"
  | "quantile_transform"
  | "pca"
  | "svd"
  | "select_k_best"
  | "drop_correlated"
  | "variance_threshold";

export interface FeatureDecision {
  op: FeatureOp;
  input_columns: string[];
  output_name_hint: string;
  parameters: Param[];
  rationale: string;
  hypothesis: string;
  risk: string;
  priority: Priority;
}

export interface FeaturePlan {
  decisions: FeatureDecision[];
  summary: string;
  expected_feature_count_delta: number;
  dimensionality_strategy: string;
  selection_strategy: string;
}

export type ModelFamily =
  | "linear"
  | "logistic"
  | "ridge"
  | "lasso"
  | "elastic_net"
  | "decision_tree"
  | "random_forest"
  | "extra_trees"
  | "gradient_boosting"
  | "hist_gradient_boosting"
  | "xgboost"
  | "lightgbm"
  | "catboost"
  | "svm"
  | "knn"
  | "naive_bayes"
  | "neural_network"
  | "kmeans"
  | "dbscan"
  | "gaussian_mixture"
  | "isolation_forest"
  | "local_outlier_factor"
  | "one_class_svm"
  | "seasonal_naive"
  | "theta"
  | "exponential_smoothing"
  | "sarimax"
  | "baseline_dummy";

export interface ModelCandidate {
  family: ModelFamily;
  rank: number;
  suitability: "excellent" | "good" | "fair" | "poor";
  rationale: string;
  expected_strengths: string[];
  expected_weaknesses: string[];
  initial_params: Param[];
  is_baseline: boolean;
  tune_priority: "high" | "medium" | "low" | "none";
}

export interface ModelSelection {
  candidates: ModelCandidate[];
  summary: string;
  reasoning: string;
  excluded_families: string[];
  exclusion_rationale: string[];
  validation_strategy: string;
  validation_rationale: string;
}

export interface MetricValue {
  name: string;
  value: number;
  std: number | null;
}

export interface ExperimentResult {
  experiment_id: string;
  family: ModelFamily;
  label: string;
  params: Param[];
  metrics: MetricValue[];
  cv_scores: number[];
  primary_metric: string;
  primary_score: number | null;
  train_seconds: number;
  predict_seconds: number;
  peak_memory_mb: number;
  model_size_bytes: number;
  n_features_in: number;
  estimated_cost_usd: number;
  failed: boolean;
  error: string | null;
  is_baseline: boolean;
  tuned: boolean;
  artifact_path: string | null;
  created_at: string;
}

export interface ExperimentLog {
  results: ExperimentResult[];
  best_experiment_id: string | null;
  primary_metric: string;
  higher_is_better: boolean;
  leaderboard_notes: string;
}

export type TuningMethod =
  | "none"
  | "grid_search"
  | "random_search"
  | "bayesian"
  | "optuna_tpe"
  | "halving_random";

export interface SearchSpaceEntry {
  name: string;
  kind: "int" | "float" | "categorical" | "log_float";
  low: number | null;
  high: number | null;
  choices: string[];
  rationale: string;
}

export interface TuningDecision {
  worthwhile: boolean;
  rationale: string;
  method: TuningMethod;
  method_rationale: string;
  target_family: ModelFamily | null;
  n_trials: number;
  timeout_seconds: number;
  early_stopping: boolean;
  search_space: SearchSpaceEntry[];
  expected_gain: string;
}

export interface TuningResult {
  ran: boolean;
  method: TuningMethod;
  family: ModelFamily | null;
  n_trials_completed: number;
  best_params: Param[];
  best_score: number | null;
  baseline_score: number | null;
  improvement: number | null;
  seconds: number;
  trial_scores: number[];
  skipped_reason: string | null;
  error: string | null;
}

export interface FeatureAttribution {
  feature: string;
  importance: number;
  direction: "increases" | "decreases" | "mixed" | "unknown";
  method: string;
}

export interface Counterfactual {
  description: string;
  changed_features: Param[];
  original_prediction: string;
  new_prediction: string;
}

export interface ExplainabilityReport {
  global_attributions: FeatureAttribution[];
  permutation_importance: FeatureAttribution[];
  shap_available: boolean;
  shap_summary_path: string | null;
  partial_dependence_paths: string[];
  counterfactuals: Counterfactual[];
  plain_language_explanations: string[];
  narrative: string;
  method_notes: string;
}

export interface BiasVarianceDiagnosis {
  train_score: number | null;
  validation_score: number | null;
  test_score: number | null;
  gap: number | null;
  verdict: "underfitting" | "good_fit" | "overfitting" | "inconclusive";
  detail: string;
}

export interface CalibrationDiagnosis {
  applicable: boolean;
  brier_score: number | null;
  expected_calibration_error: number | null;
  verdict: string;
}

export interface FairnessSlice {
  attribute: string;
  slice_value: string;
  n_rows: number;
  metric_name: string;
  metric_value: number;
  delta_vs_overall: number;
}

export interface ConfidenceInterval {
  metric: string;
  point_estimate: number;
  lower: number;
  upper: number;
  level: number;
  method: string;
}

export type RecommendedAction =
  | "accept"
  | "retry_feature_engineering"
  | "retry_model_selection"
  | "retry_cleaning"
  | "collect_more_data"
  | "reject";

export interface EvaluationVerdict {
  acceptable: boolean;
  verdict_rationale: string;
  overall_grade: "A" | "B" | "C" | "D" | "F";
  bias_variance: BiasVarianceDiagnosis;
  calibration: CalibrationDiagnosis;
  generalisation_notes: string;
  drift_risk: "low" | "medium" | "high" | "unknown";
  drift_rationale: string;
  fairness_slices: FairnessSlice[];
  fairness_notes: string;
  confidence_intervals: ConfidenceInterval[];
  residual_notes: string;
  error_analysis: string[];
  learning_curve_notes: string;
  weaknesses: string[];
  recommended_action: RecommendedAction;
  action_rationale: string;
  specific_improvements: string[];
}

export interface BusinessInsight {
  headline: string;
  detail: string;
  supporting_evidence: string;
  recommended_action: string;
  expected_value: string;
  confidence: Confidence;
  audience: "executive" | "operations" | "marketing" | "product" | "data_team";
}

export interface InsightReport {
  executive_summary: string;
  insights: BusinessInsight[];
  key_drivers_plain_language: string[];
  caveats: string[];
  suggested_next_experiments: string[];
}

export type ChartKind =
  | "correlation_heatmap"
  | "histogram"
  | "box"
  | "scatter"
  | "bar"
  | "line"
  | "roc_curve"
  | "pr_curve"
  | "confusion_matrix"
  | "feature_importance"
  | "shap_summary"
  | "residuals"
  | "residual_histogram"
  | "learning_curve"
  | "prediction_distribution"
  | "calibration_curve"
  | "missingness"
  | "class_balance"
  | "leaderboard"
  | "time_series_forecast"
  | "partial_dependence";

export interface ChartSpec {
  kind: ChartKind;
  title: string;
  columns: string[];
  rationale: string;
  parameters: Param[];
  priority: Priority;
}

export interface ChartArtifact {
  spec: ChartSpec;
  html_path: string | null;
  png_path: string | null;
  json_path: string | null;
  caption: string;
  rendered: boolean;
  error: string | null;
}

export interface VisualizationPlan {
  charts: ChartSpec[];
  dashboard_narrative: string;
}

export interface VisualizationBundle {
  artifacts: ChartArtifact[];
  dashboard_path: string | null;
  narrative: string;
}

export type DeploymentPattern =
  | "batch_inference"
  | "rest_api"
  | "serverless"
  | "streaming"
  | "edge"
  | "embedded_sql";

export interface DeploymentRecommendation {
  pattern: DeploymentPattern;
  rationale: string;
  estimated_latency_ms: number | null;
  estimated_throughput_rps: number | null;
  model_size_mb: number | null;
  infrastructure_notes: string;
  monitoring_plan: string[];
  retraining_cadence: string;
  rollout_strategy: string;
  risks: string[];
}

export interface ReportSection {
  heading: string;
  order: number;
  body_markdown: string;
  chart_refs: string[];
}

export interface FinalReport {
  title: string;
  subtitle: string;
  executive_summary: string;
  sections: ReportSection[];
  deployment: DeploymentRecommendation;
  appendix_notes: string[];
}

export interface ReportBundle {
  markdown_path: string | null;
  html_path: string | null;
  pdf_path: string | null;
  pptx_path: string | null;
  json_path: string | null;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Run configuration & approvals
// ---------------------------------------------------------------------------

export type ApprovalDecision = "pending" | "approved" | "rejected";

export interface ApprovalRequest {
  request_id: string;
  step_id: string;
  agent: AgentName;
  action_summary: string;
  details: string[];
  affected_columns: string[];
  affected_row_estimate: number;
  severity: Severity;
  created_at: string;
  decision: ApprovalDecision;
  decided_at: string | null;
  decided_by: string | null;
  note: string | null;
}

export interface RunConfig {
  run_id: string;
  project: string;
  source: DataSource;
  target_column: string | null;
  task_type_override: TaskType | null;
  primary_metric_override: string | null;
  time_budget_seconds: number;
  max_experiments: number;
  max_rows: number | null;
  test_size: number;
  validation_size: number;
  cv_folds: number;
  random_state: number;
  require_approval: boolean;
  enable_tuning: boolean;
  enable_explainability: boolean;
  enable_self_improvement: boolean;
  max_replans: number;
  min_acceptable_score: number | null;
  report_formats: string[];
  fairness_attributes: string[];
  notes: string;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

export type EventKind =
  | "run_started"
  | "run_completed"
  | "run_failed"
  | "run_cancelled"
  | "step_started"
  | "step_completed"
  | "step_failed"
  | "step_skipped"
  | "step_retried"
  | "agent_thinking"
  | "agent_decision"
  | "llm_call"
  | "artifact_written"
  | "approval_requested"
  | "approval_resolved"
  | "replan_triggered"
  | "metric_recorded"
  | "warning"
  | "log";

export const EVENT_KINDS: EventKind[] = [
  "run_started",
  "run_completed",
  "run_failed",
  "run_cancelled",
  "step_started",
  "step_completed",
  "step_failed",
  "step_skipped",
  "step_retried",
  "agent_thinking",
  "agent_decision",
  "llm_call",
  "artifact_written",
  "approval_requested",
  "approval_resolved",
  "replan_triggered",
  "metric_recorded",
  "warning",
  "log",
];

/** Events that mean the orchestrator will emit nothing further. */
export const TERMINAL_EVENT_KINDS: ReadonlySet<EventKind> = new Set<EventKind>([
  "run_completed",
  "run_failed",
  "run_cancelled",
]);

export const TERMINAL_RUN_STATUSES: ReadonlySet<RunStatus> = new Set<RunStatus>([
  "completed",
  "failed",
  "cancelled",
]);

export interface RunEvent {
  event_id: string;
  run_id: string;
  sequence: number;
  kind: EventKind;
  agent: AgentName | null;
  step_id: string | null;
  message: string;
  payload: Param[];
  at: string;
  duration_seconds: number | null;
  tokens_in: number | null;
  tokens_out: number | null;
  cache_read_tokens: number | null;
  cost_usd: number | null;
}

export interface StepRecord {
  step_id: string;
  title: string;
  agent: AgentName | null;
  status: StepStatus;
  attempts: number;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number;
  error: string | null;
  summary: string;
}

export interface UsageTotals {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  llm_calls: number;
  cost_usd: number;
}

export interface RunSummary {
  run_id: string;
  project: string;
  status: RunStatus;
  config: RunConfig;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number;
  error: string | null;

  ingestion: IngestionResult | null;
  profile: DatasetProfile | null;
  understanding: DatasetUnderstanding | null;
  problem: ProblemDefinition | null;
  plan: ExecutionPlan | null;
  plan_history: ExecutionPlan[];
  cleaning: CleaningPlan | null;
  features: FeaturePlan | null;
  model_selection: ModelSelection | null;
  experiments: ExperimentLog | null;
  tuning_decision: TuningDecision | null;
  tuning: TuningResult | null;
  explainability: ExplainabilityReport | null;
  evaluation: EvaluationVerdict | null;
  insights: InsightReport | null;
  visualization_plan: VisualizationPlan | null;
  visualizations: VisualizationBundle | null;
  report: FinalReport | null;
  report_bundle: ReportBundle | null;

  steps: StepRecord[];
  approvals: ApprovalRequest[];
  usage: UsageTotals;
  replans: number;
  warnings: string[];
  artifact_dir: string | null;
}

// ---------------------------------------------------------------------------
// Dataset memory & natural-language Q&A
// ---------------------------------------------------------------------------

export interface DatasetFingerprint {
  fingerprint_id: string;
  run_id: string;
  project: string;
  n_rows: number;
  n_columns: number;
  n_numeric: number;
  n_categorical: number;
  n_datetime: number;
  n_text: number;
  missing_fraction: number;
  duplicate_fraction: number;
  task_type: TaskType | null;
  target_kind: ColumnKind | null;
  imbalance_ratio: number | null;
  column_names: string[];
  primary_metric: string;
  best_score: number | null;
  best_family: ModelFamily | null;
  winning_feature_ops: string[];
  created_at: string;
}

export interface SimilarRun {
  run_id: string;
  similarity: number;
  task_type: TaskType | null;
  best_family: ModelFamily | null;
  best_score: number | null;
  primary_metric: string;
  why_similar: string;
}

export interface MemorySuggestion {
  has_precedent: boolean;
  similar_runs: SimilarRun[];
  recommended_families: ModelFamily[];
  recommended_feature_ops: FeatureOp[];
  cautions: string[];
  narrative: string;
}

export interface QuestionAnswer {
  question: string;
  answer: string;
  evidence: string[];
  confidence: Confidence;
  caveats: string[];
  suggested_followups: string[];
}

// ---------------------------------------------------------------------------
// HTTP-only shapes — these live in the FastAPI layer, not in schemas.py
// ---------------------------------------------------------------------------

/** One entry of `HealthResponse.features` — mirrors `api/schemas.py:FeatureStatus`. */
export interface FeatureStatus {
  name: string;
  available: boolean;
  packages?: string[];
  install_hint?: string;
  detail?: string;
}

/**
 * `GET /api/health` — mirrors `api/schemas.py:HealthResponse`.
 *
 * Every field is optional here even though the backend always sends them: this
 * is also the reachability probe, and a partial answer from an older build
 * should still count as "the API is up".
 */
export interface HealthResponse {
  status?: "ok" | "degraded";
  version?: string;
  model?: string;
  python_version?: string;
  credentials_configured?: boolean;
  workspace?: string;
  workspace_writable?: boolean;
  database_url?: string;
  database_reachable?: boolean;
  active_runs?: number;
  stored_runs?: number;
  features?: FeatureStatus[];
  warnings?: string[];
}

/**
 * One row of `GET /api/runs` — mirrors `api/schemas.py:RunListItem`.
 *
 * The flat names (`target`, `grade`, …) are what that endpoint actually sends.
 * The nested ones are accepted too, because `summaryToCard` feeds a whole
 * `RunSummary` through the same derivation; `deriveRunCard` in `lib/runCard.ts`
 * reads whichever is present and reports "unknown" rather than inventing one.
 */
export interface RunListItem {
  run_id: string;
  project?: string;
  status: RunStatus;
  started_at?: string | null;
  finished_at?: string | null;
  duration_seconds?: number;
  error?: string | null;

  // Flat projection fields, as sent by GET /api/runs.
  task_type?: TaskType | null;
  target?: string | null;
  primary_metric?: string | null;
  best_score?: number | null;
  /** The stored `ModelFamily` value of the winning experiment. */
  best_family?: ModelFamily | null;
  n_rows?: number | null;
  n_columns?: number | null;
  n_experiments?: number | null;
  grade?: string | null;
  cost_usd?: number;
  is_active?: boolean;

  // Nested fields, when a full RunSummary row is derived from instead.
  config?: RunConfig;
  problem?: ProblemDefinition | null;
  experiments?: ExperimentLog | null;
  ingestion?: IngestionResult | null;
  evaluation?: EvaluationVerdict | null;
}

/** `POST /api/upload` — mirrors `api/schemas.py:UploadResponse`. */
export interface UploadResponse {
  filename: string;
  path: string;
  size_bytes: number;
  source: DataSource;
}

/** `POST /api/runs` — mirrors `api/schemas.py:StartRunResponse`. */
export interface CreateRunResponse {
  run_id: string;
  project: string;
  status: RunStatus;
  stream_url: string;
  events_url: string;
  run_url: string;
  accepted_at?: string;
}

/** `GET /api/runs/{id}/events` — mirrors `api/schemas.py:EventPage`. */
export interface EventPage {
  run_id: string;
  after: number;
  last_sequence: number;
  count: number;
  status: RunStatus | null;
  finished: boolean;
  events: RunEvent[];
}

/** `POST /api/runs/{id}/cancel` — mirrors `api/schemas.py:CancelResponse`. */
export interface CancelResponse {
  run_id: string;
  status: RunStatus | null;
  cancellation_requested: boolean;
  detail: string;
}

/**
 * One entry of `GET /api/runs/{id}/artifacts` — mirrors
 * `api/schemas.py:ArtifactEntry`. `relative_path` is keyed to the run
 * directory and `url` is already server-relative, so neither needs
 * `artifactUrl`'s path surgery.
 */
export interface ArtifactRef {
  relative_path: string;
  kind: string;
  size_bytes: number;
  modified_at: string;
  url: string;
}

/**
 * Body of `POST /api/runs/{id}/approvals/{request_id}` — mirrors
 * `api/schemas.py:ApprovalDecisionRequest`.
 */
export interface ApprovalDecisionBody {
  decision: "approved" | "rejected";
  note?: string;
  decided_by?: string;
}

/** Its response — mirrors `api/schemas.py:ApprovalDecisionResponse`. */
export interface ApprovalDecisionResponse {
  run_id: string;
  request_id: string;
  approval: ApprovalRequest | null;
  /** True when the paused run was released by this decision. */
  resumed: boolean;
  detail: string;
}

/**
 * JSON body of `POST /api/runs` — mirrors `api/schemas.py:StartRunRequest`.
 *
 * That model is `extra="forbid"`, so an unknown key is a 422 and the names here
 * must match it exactly: `task_type`/`primary_metric`, not the
 * `*_override` names `RunConfig` uses internally.
 */
export interface StartRunBody {
  source?: DataSource;
  uri?: string;
  upload_path?: string;
  query?: string;

  project?: string;
  target_column?: string;
  task_type?: TaskType;
  primary_metric?: string;
  time_budget_seconds?: number;
  max_experiments?: number;
  max_rows?: number | null;
  test_size?: number;
  validation_size?: number;
  cv_folds?: number;
  random_state?: number;
  require_approval?: boolean;
  enable_tuning?: boolean;
  enable_explainability?: boolean;
  enable_self_improvement?: boolean;
  max_replans?: number;
  min_acceptable_score?: number | null;
  report_formats?: string[];
  fairness_attributes?: string[];
  notes?: string;
}

/** Options accepted by `POST /api/runs`, mirroring the RunConfig knobs. */
export interface NewRunOptions {
  project: string;
  target_column: string;
  time_budget_seconds: number;
  max_experiments: number;
  max_rows: number | null;
  test_size: number;
  cv_folds: number;
  random_state: number;
  require_approval: boolean;
  enable_tuning: boolean;
  enable_explainability: boolean;
  enable_self_improvement: boolean;
  primary_metric_override: string;
  task_type_override: TaskType | "";
  fairness_attributes: string;
  report_formats: string[];
  notes: string;
}

/** A plotly figure as written by `plotly.io.write_json`. */
export interface PlotlyFigure {
  data: unknown[];
  layout?: Record<string, unknown>;
  frames?: unknown[];
  config?: Record<string, unknown>;
}
