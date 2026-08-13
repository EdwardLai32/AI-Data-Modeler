"""Canonical data contracts for AutoML Architect.

Every agent, executor, and API surface speaks these types. Two rules govern the
models here:

1.  **LLM-facing models** (anything an agent returns via structured output) must
    stay compatible with the Claude structured-output JSON-schema subset:
    no recursion, no free-form ``dict`` fields, and no numeric/string
    constraints that the schema compiler would reject. Free-form key/value data
    is expressed as ``list[Param]`` instead of ``dict[str, Any]`` — an object
    with no declared properties is not representable there.
2.  **Every decision carries its reasoning.** ``rationale`` is required, not
    optional, on each decision model. That is the product requirement expressed
    as a type: an agent physically cannot emit an unexplained transformation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Base(BaseModel):
    """Shared config. ``extra="forbid"`` mirrors ``additionalProperties: false``."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# ---------------------------------------------------------------------------
# Free-form parameters
# ---------------------------------------------------------------------------


class Param(Base):
    """One key/value pair.

    Used wherever an agent needs to emit open-ended configuration. Values are
    always strings on the wire; :func:`params_to_dict` coerces them back to
    Python scalars so executors receive real ints/floats/bools/lists.
    """

    key: str = Field(description="Parameter name, e.g. 'n_estimators'.")
    value: str = Field(
        description=(
            "Parameter value as a string. Numbers, booleans ('true'/'false'), "
            "null, and JSON arrays/objects are all accepted and will be parsed."
        )
    )


def _coerce(raw: str) -> Any:
    text = raw.strip()
    if text == "" or text.lower() in {"null", "none"}:
        return None
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def params_to_dict(params: list[Param] | None) -> dict[str, Any]:
    """Convert an agent-supplied ``list[Param]`` into a real kwargs dict."""
    if not params:
        return {}
    return {p.key: _coerce(p.value) for p in params}


def dict_to_params(data: dict[str, Any] | None) -> list[Param]:
    """Inverse of :func:`params_to_dict`, for round-tripping into reports."""
    if not data:
        return []
    out: list[Param] = []
    for key, value in data.items():
        if isinstance(value, str):
            rendered = value
        else:
            try:
                rendered = json.dumps(value)
            except (TypeError, ValueError):
                rendered = str(value)
        out.append(Param(key=key, value=rendered))
    return out


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    RECOMMENDATION = "recommendation"
    RANKING = "ranking"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    GRAPH_LEARNING = "graph_learning"
    SURVIVAL_ANALYSIS = "survival_analysis"
    CAUSAL_INFERENCE = "causal_inference"

    @property
    def is_classification(self) -> bool:
        return self in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        }

    @property
    def is_supervised(self) -> bool:
        return self not in {
            TaskType.CLUSTERING,
            TaskType.ANOMALY_DETECTION,
            TaskType.GRAPH_LEARNING,
        }

    @property
    def is_supported(self) -> bool:
        """Whether the execution layer can actually train this end-to-end."""
        return self in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.REGRESSION,
            TaskType.TIME_SERIES_FORECASTING,
            TaskType.CLUSTERING,
            TaskType.ANOMALY_DETECTION,
        }


class ColumnKind(str, Enum):
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    GEO = "geo"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


class ColumnRole(str, Enum):
    TARGET = "target"
    FEATURE = "feature"
    IDENTIFIER = "identifier"
    TEMPORAL_INDEX = "temporal_index"
    GROUP_KEY = "group_key"
    WEIGHT = "weight"
    LEAKAGE_SUSPECT = "leakage_suspect"
    IGNORED = "ignored"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


class AgentName(str, Enum):
    DATASET = "dataset"
    PROBLEM = "problem"
    PLANNER = "planner"
    CLEANING = "cleaning"
    FEATURES = "features"
    MODEL_SELECTION = "model_selection"
    EXPERIMENT = "experiment"
    TUNING = "tuning"
    EXPLAIN = "explain"
    EVALUATION = "evaluation"
    INSIGHT = "insight"
    VISUALIZATION = "visualization"
    REPORT = "report"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class SourceKind(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    SQL = "sql"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SNOWFLAKE = "snowflake"
    DATABRICKS = "databricks"
    DUCKDB = "duckdb"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"
    REST_API = "rest_api"
    KAGGLE = "kaggle"
    DATAFRAME = "dataframe"


class DataSource(Base):
    """Where a dataset comes from. ``options`` carries connector-specific knobs."""

    kind: SourceKind
    uri: str = Field(
        default="",
        description="Path, URL, connection string, or dataset slug for the source.",
    )
    query: str | None = Field(
        default=None, description="SQL query or table name, for database sources."
    )
    options: list[Param] = Field(default_factory=list)
    secret_env: list[str] = Field(
        default_factory=list,
        description="Names of environment variables holding credentials. "
        "Values are never stored on the source itself.",
    )


class SchemaField(Base):
    name: str
    inferred_dtype: str
    nullable: bool = True
    sample_values: list[str] = Field(default_factory=list)


class IngestionResult(Base):
    source: DataSource
    dataset_id: str = Field(default_factory=lambda: _new_id("ds"))
    n_rows: int
    n_columns: int
    schema_fields: list[SchemaField] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    bytes_in_memory: int = 0
    load_seconds: float = 0.0
    truncated: bool = Field(
        default=False, description="True when a row limit capped the load."
    )


# ---------------------------------------------------------------------------
# Profiling — computed deterministically, never hallucinated
# ---------------------------------------------------------------------------


class Quantiles(Base):
    p01: float | None = None
    p05: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p95: float | None = None
    p99: float | None = None


class OutlierSummary(Base):
    method: str = "iqr"
    n_outliers: int = 0
    fraction: float = 0.0
    lower_bound: float | None = None
    upper_bound: float | None = None


class CategoryCount(Base):
    value: str
    count: int
    fraction: float


class ColumnProfile(Base):
    """Deterministic statistics for one column."""

    name: str
    kind: ColumnKind
    dtype: str
    n_missing: int = 0
    missing_fraction: float = 0.0
    n_unique: int = 0
    cardinality_ratio: float = 0.0
    is_constant: bool = False
    memory_bytes: int = 0

    # numeric
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zero_fraction: float | None = None
    negative_fraction: float | None = None
    quantiles: Quantiles | None = None
    outliers: OutlierSummary | None = None
    variance: float | None = None
    is_near_zero_variance: bool = False

    # categorical / text
    top_values: list[CategoryCount] = Field(default_factory=list)
    mean_string_length: float | None = None
    max_string_length: int | None = None
    mean_token_count: float | None = None

    # datetime
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    inferred_frequency: str | None = None
    n_gaps: int | None = None
    is_monotonic: bool | None = None

    # semantic flags
    looks_like_id: bool = False
    looks_like_geo: bool = False
    looks_like_text: bool = False
    looks_like_datetime: bool = False
    detected_semantic_type: str | None = Field(
        default=None,
        description="e.g. email, url, ipv4, latitude, longitude, postal_code, currency.",
    )


class CorrelationPair(Base):
    left: str
    right: str
    coefficient: float
    method: str = "pearson"


class TargetSummary(Base):
    name: str
    kind: ColumnKind
    n_classes: int | None = None
    class_counts: list[CategoryCount] = Field(default_factory=list)
    imbalance_ratio: float | None = Field(
        default=None,
        description="majority_count / minority_count; >3 is meaningfully imbalanced.",
    )
    is_imbalanced: bool = False
    mean: float | None = None
    std: float | None = None
    skewness: float | None = None
    n_missing: int = 0


class LeakageFinding(Base):
    column: str
    score: float = Field(
        description="Association strength with the target, 0-1. "
        "Near 1.0 on a non-trivial feature is the classic leakage signature."
    )
    method: str
    severity: Severity
    reason: str


class DataQualityIssue(Base):
    code: str
    severity: Severity
    columns: list[str] = Field(default_factory=list)
    detail: str


class DatasetProfile(Base):
    """The full deterministic picture of a dataset."""

    dataset_id: str
    n_rows: int
    n_columns: int
    memory_bytes: int = 0
    n_duplicate_rows: int = 0
    duplicate_fraction: float = 0.0
    total_missing_cells: int = 0
    missing_cell_fraction: float = 0.0
    columns: list[ColumnProfile] = Field(default_factory=list)
    target: TargetSummary | None = None
    top_correlations: list[CorrelationPair] = Field(default_factory=list)
    target_correlations: list[CorrelationPair] = Field(default_factory=list)
    highly_correlated_pairs: list[CorrelationPair] = Field(default_factory=list)
    leakage_findings: list[LeakageFinding] = Field(default_factory=list)
    quality_issues: list[DataQualityIssue] = Field(default_factory=list)
    temporal_columns: list[str] = Field(default_factory=list)
    geo_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    identifier_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)
    profiled_at: datetime = Field(default_factory=_utcnow)
    profile_seconds: float = 0.0

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------


class ColumnAssessment(Base):
    """The Dataset Agent's read on one column, layered over the raw stats."""

    name: str
    role: ColumnRole
    predictive_potential: Literal["high", "medium", "low", "none"]
    concerns: list[str] = Field(default_factory=list)
    notes: str = Field(description="One or two sentences a data scientist would write.")


class DatasetUnderstanding(Base):
    """Output of the Dataset Understanding Agent."""

    headline: str = Field(
        description="One sentence capturing what this dataset appears to be."
    )
    narrative: str = Field(
        description=(
            "Several paragraphs of exploratory analysis in the voice of a senior "
            "data scientist: what the data is, its shape and quality, the "
            "distributions and relationships that matter, and what to watch out for."
        )
    )
    likely_domain: str = Field(
        description="Business domain, e.g. 'telecom customer churn', 'retail demand'."
    )
    grain: str = Field(
        description="What one row represents, e.g. 'one customer', 'one order-day'."
    )
    column_assessments: list[ColumnAssessment] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    suggested_target_columns: list[str] = Field(
        default_factory=list,
        description="Ranked candidates for the prediction target, best first.",
    )
    data_readiness: Literal["ready", "needs_cleaning", "needs_major_work", "unusable"]
    readiness_rationale: str


class ProblemDefinition(Base):
    """Output of the Problem Identification Agent."""

    task_type: TaskType
    target_column: str | None = Field(
        default=None, description="Null for unsupervised tasks."
    )
    positive_class: str | None = Field(
        default=None, description="For binary classification, the class of interest."
    )
    temporal_column: str | None = None
    group_column: str | None = Field(
        default=None,
        description="Entity key for grouped splits or per-series forecasting.",
    )
    horizon: int | None = Field(
        default=None, description="Forecast horizon in periods, for time series."
    )
    rationale: str = Field(
        description=(
            "Why this task type follows from the data. Reference the concrete "
            "evidence, e.g. 'the target is binary and categorical, indicating "
            "a binary classification problem'."
        )
    )
    alternatives_considered: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    primary_metric: str = Field(
        description="Metric to optimise, e.g. roc_auc, f1, rmse, mae, r2, silhouette."
    )
    secondary_metrics: list[str] = Field(default_factory=list)
    metric_rationale: str
    business_objective: str = Field(
        description="What a decision-maker would actually do with this model."
    )
    constraints: list[str] = Field(default_factory=list)


class PlanStep(Base):
    """One node of the execution plan."""

    step_id: str = Field(description="Stable slug, e.g. 'handle_missing_values'.")
    order: int = Field(description="1-based execution order.")
    title: str
    agent: AgentName = Field(description="Which agent owns this step.")
    objective: str = Field(description="What this step must achieve.")
    rationale: str = Field(description="Why this step is needed for THIS dataset.")
    depends_on: list[str] = Field(default_factory=list)
    optional: bool = False
    destructive: bool = Field(
        default=False,
        description="True if the step drops columns or rows; gates human approval.",
    )
    estimated_seconds: int = 30
    success_criteria: str = ""


class ExecutionPlan(Base):
    """Output of the Planning Agent — the project's brain."""

    summary: str = Field(description="A paragraph describing the overall strategy.")
    steps: list[PlanStep]
    dataset_specific_adaptations: list[str] = Field(
        default_factory=list,
        description="How this plan differs from a generic pipeline, and why.",
    )
    risks: list[str] = Field(default_factory=list)
    fallback_strategy: str = Field(
        description="What to try if evaluation comes back unacceptable."
    )
    revision: int = 0
    revision_reason: str | None = None

    def ordered(self) -> list[PlanStep]:
        return sorted(self.steps, key=lambda s: s.order)


class MissingStrategy(str, Enum):
    DROP_COLUMN = "drop_column"
    DROP_ROWS = "drop_rows"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    KNN = "knn"
    ITERATIVE = "iterative"
    MISSING_CATEGORY = "missing_category"
    LEAVE_AS_IS = "leave_as_is"


class CleaningAction(str, Enum):
    IMPUTE_MISSING = "impute_missing"
    DROP_COLUMN = "drop_column"
    DROP_DUPLICATE_ROWS = "drop_duplicate_rows"
    DROP_ROWS_MISSING_TARGET = "drop_rows_missing_target"
    CLIP_OUTLIERS = "clip_outliers"
    REMOVE_OUTLIER_ROWS = "remove_outlier_rows"
    CAST_DTYPE = "cast_dtype"
    PARSE_DATETIME = "parse_datetime"
    NORMALISE_CATEGORIES = "normalise_categories"
    STRIP_WHITESPACE = "strip_whitespace"
    DROP_CONSTANT_COLUMN = "drop_constant_column"
    DROP_LEAKAGE_COLUMN = "drop_leakage_column"


class CleaningDecision(Base):
    """One explained cleaning transformation."""

    action: CleaningAction
    columns: list[str] = Field(
        default_factory=list,
        description="Empty means the action applies to the whole table.",
    )
    strategy: MissingStrategy | None = None
    parameters: list[Param] = Field(default_factory=list)
    rationale: str = Field(
        description=(
            "The evidence-based reason, e.g. 'highly skewed distribution "
            "(skewness 3.4), so the median is robust to the tail'."
        )
    )
    expected_impact: str = ""
    destructive: bool = False
    severity_if_skipped: Severity = Severity.LOW


class CleaningPlan(Base):
    """Output of the Data Cleaning Agent."""

    decisions: list[CleaningDecision]
    summary: str
    columns_to_drop: list[str] = Field(default_factory=list)
    drop_rationale: list[str] = Field(default_factory=list)
    skipped_considerations: list[str] = Field(
        default_factory=list,
        description="Transformations deliberately NOT applied, and why.",
    )


class FeatureOp(str, Enum):
    DATE_DECOMPOSE = "date_decompose"
    CYCLICAL_ENCODE = "cyclical_encode"
    LAG = "lag"
    ROLLING = "rolling"
    DIFF = "diff"
    EXPANDING = "expanding"
    INTERACTION = "interaction"
    RATIO = "ratio"
    POLYNOMIAL = "polynomial"
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    BOXCOX = "boxcox"
    BINNING = "binning"
    ONE_HOT_ENCODE = "one_hot_encode"
    ORDINAL_ENCODE = "ordinal_encode"
    TARGET_ENCODE = "target_encode"
    FREQUENCY_ENCODE = "frequency_encode"
    HASH_ENCODE = "hash_encode"
    TEXT_TFIDF = "text_tfidf"
    TEXT_LENGTH = "text_length"
    GEO_DISTANCE = "geo_distance"
    AGGREGATE_BY_GROUP = "aggregate_by_group"
    STANDARD_SCALE = "standard_scale"
    MINMAX_SCALE = "minmax_scale"
    ROBUST_SCALE = "robust_scale"
    QUANTILE_TRANSFORM = "quantile_transform"
    PCA = "pca"
    SVD = "svd"
    SELECT_K_BEST = "select_k_best"
    DROP_CORRELATED = "drop_correlated"
    VARIANCE_THRESHOLD = "variance_threshold"


class FeatureDecision(Base):
    """One explained feature-engineering operation."""

    op: FeatureOp
    input_columns: list[str] = Field(default_factory=list)
    output_name_hint: str = ""
    parameters: list[Param] = Field(default_factory=list)
    rationale: str = Field(
        description="Why this feature should improve predictive performance."
    )
    hypothesis: str = Field(
        default="",
        description="The mechanism, e.g. 'weekend orders behave differently from weekdays'.",
    )
    risk: str = Field(
        default="",
        description="Leakage or overfitting risk, and how it is mitigated.",
    )
    priority: Literal["high", "medium", "low"] = "medium"


class FeaturePlan(Base):
    """Output of the Feature Engineering Agent."""

    decisions: list[FeatureDecision]
    summary: str
    expected_feature_count_delta: int = 0
    dimensionality_strategy: str = ""
    selection_strategy: str = ""


class ModelFamily(str, Enum):
    LINEAR = "linear"
    LOGISTIC = "logistic"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SVM = "svm"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    NEURAL_NETWORK = "neural_network"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    SEASONAL_NAIVE = "seasonal_naive"
    THETA = "theta"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    SARIMAX = "sarimax"
    BASELINE_DUMMY = "baseline_dummy"


class ModelCandidate(Base):
    family: ModelFamily
    rank: int = Field(description="1 is the most promising candidate.")
    suitability: Literal["excellent", "good", "fair", "poor"]
    rationale: str = Field(
        description="Why this model suits THIS dataset's size, shape, and signal."
    )
    expected_strengths: list[str] = Field(default_factory=list)
    expected_weaknesses: list[str] = Field(default_factory=list)
    initial_params: list[Param] = Field(default_factory=list)
    is_baseline: bool = False
    tune_priority: Literal["high", "medium", "low", "none"] = "medium"


class ModelSelection(Base):
    """Output of the Model Selection Agent."""

    candidates: list[ModelCandidate]
    summary: str
    reasoning: str = Field(
        description="The comparative argument across candidates, not a list of facts."
    )
    excluded_families: list[str] = Field(default_factory=list)
    exclusion_rationale: list[str] = Field(default_factory=list)
    validation_strategy: str = Field(
        description="e.g. 'stratified 5-fold', 'expanding-window time series split'."
    )
    validation_rationale: str = ""


class MetricValue(Base):
    name: str
    value: float
    std: float | None = None


class ExperimentResult(Base):
    """One trained-and-scored model. Populated by the execution layer."""

    experiment_id: str = Field(default_factory=lambda: _new_id("exp"))
    family: ModelFamily
    label: str = ""
    params: list[Param] = Field(default_factory=list)
    metrics: list[MetricValue] = Field(default_factory=list)
    cv_scores: list[float] = Field(default_factory=list)
    primary_metric: str = ""
    primary_score: float | None = None
    train_seconds: float = 0.0
    predict_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    model_size_bytes: int = 0
    n_features_in: int = 0
    estimated_cost_usd: float = 0.0
    failed: bool = False
    error: str | None = None
    is_baseline: bool = False
    tuned: bool = False
    artifact_path: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    def metric(self, name: str) -> float | None:
        return next((m.value for m in self.metrics if m.name == name), None)


class ExperimentLog(Base):
    results: list[ExperimentResult] = Field(default_factory=list)
    best_experiment_id: str | None = None
    primary_metric: str = ""
    higher_is_better: bool = True
    leaderboard_notes: str = ""

    def best(self) -> ExperimentResult | None:
        return next(
            (r for r in self.results if r.experiment_id == self.best_experiment_id),
            None,
        )


class TuningMethod(str, Enum):
    NONE = "none"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    OPTUNA_TPE = "optuna_tpe"
    HALVING_RANDOM = "halving_random"


class SearchSpaceEntry(Base):
    """One hyperparameter's search range."""

    name: str
    kind: Literal["int", "float", "categorical", "log_float"]
    low: float | None = None
    high: float | None = None
    choices: list[str] = Field(default_factory=list)
    rationale: str = ""


class TuningDecision(Base):
    """Output of the Hyperparameter Optimization Agent."""

    worthwhile: bool = Field(
        description="Whether tuning is worth the compute for this dataset and gap."
    )
    rationale: str = Field(
        description="The cost/benefit argument, referencing baseline scores and data size."
    )
    method: TuningMethod
    method_rationale: str = ""
    target_family: ModelFamily | None = None
    n_trials: int = 25
    timeout_seconds: int = 300
    early_stopping: bool = True
    search_space: list[SearchSpaceEntry] = Field(default_factory=list)
    expected_gain: str = ""


class TuningResult(Base):
    ran: bool = False
    method: TuningMethod = TuningMethod.NONE
    family: ModelFamily | None = None
    n_trials_completed: int = 0
    best_params: list[Param] = Field(default_factory=list)
    best_score: float | None = None
    baseline_score: float | None = None
    improvement: float | None = None
    seconds: float = 0.0
    trial_scores: list[float] = Field(default_factory=list)
    skipped_reason: str | None = None
    error: str | None = None


class FeatureAttribution(Base):
    feature: str
    importance: float = Field(description="Normalised so the values sum to 1.0.")
    direction: Literal["increases", "decreases", "mixed", "unknown"] = "unknown"
    method: str = "shap"


class Counterfactual(Base):
    description: str
    changed_features: list[Param] = Field(default_factory=list)
    original_prediction: str = ""
    new_prediction: str = ""


class ExplainabilityReport(Base):
    """Populated by the execution layer, narrated by the Explainability Agent."""

    global_attributions: list[FeatureAttribution] = Field(default_factory=list)
    permutation_importance: list[FeatureAttribution] = Field(default_factory=list)
    shap_available: bool = False
    shap_summary_path: str | None = None
    partial_dependence_paths: list[str] = Field(default_factory=list)
    counterfactuals: list[Counterfactual] = Field(default_factory=list)
    plain_language_explanations: list[str] = Field(
        default_factory=list,
        description=(
            "Business-readable sentences, e.g. 'Customer tenure contributes "
            "approximately 31% of churn prediction importance.'"
        ),
    )
    narrative: str = ""
    method_notes: str = ""


class BiasVarianceDiagnosis(Base):
    train_score: float | None = None
    validation_score: float | None = None
    test_score: float | None = None
    gap: float | None = None
    verdict: Literal["underfitting", "good_fit", "overfitting", "inconclusive"] = (
        "inconclusive"
    )
    detail: str = ""


class CalibrationDiagnosis(Base):
    applicable: bool = False
    brier_score: float | None = None
    expected_calibration_error: float | None = None
    verdict: str = ""


class FairnessSlice(Base):
    attribute: str
    slice_value: str
    n_rows: int
    metric_name: str
    metric_value: float
    delta_vs_overall: float


class ConfidenceInterval(Base):
    metric: str
    point_estimate: float
    lower: float
    upper: float
    level: float = 0.95
    method: str = "bootstrap"


class EvaluationVerdict(Base):
    """Output of the Evaluation Agent — the quality gate."""

    acceptable: bool = Field(
        description="Whether the model is fit to recommend for deployment."
    )
    verdict_rationale: str
    overall_grade: Literal["A", "B", "C", "D", "F"]
    bias_variance: BiasVarianceDiagnosis = Field(
        default_factory=BiasVarianceDiagnosis
    )
    calibration: CalibrationDiagnosis = Field(default_factory=CalibrationDiagnosis)
    generalisation_notes: str = ""
    drift_risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    drift_rationale: str = ""
    fairness_slices: list[FairnessSlice] = Field(default_factory=list)
    fairness_notes: str = ""
    confidence_intervals: list[ConfidenceInterval] = Field(default_factory=list)
    residual_notes: str = ""
    error_analysis: list[str] = Field(default_factory=list)
    learning_curve_notes: str = ""
    weaknesses: list[str] = Field(default_factory=list)
    recommended_action: Literal[
        "accept",
        "retry_feature_engineering",
        "retry_model_selection",
        "retry_cleaning",
        "collect_more_data",
        "reject",
    ]
    action_rationale: str = ""
    specific_improvements: list[str] = Field(default_factory=list)


class BusinessInsight(Base):
    headline: str = Field(description="A statement a non-technical executive can act on.")
    detail: str
    supporting_evidence: str = Field(
        description="Which metric or feature attribution backs this up."
    )
    recommended_action: str
    expected_value: str = Field(
        default="", description="Quantified where possible, e.g. 'retain ~120 accounts/quarter'."
    )
    confidence: Literal["high", "medium", "low"]
    audience: Literal["executive", "operations", "marketing", "product", "data_team"] = (
        "executive"
    )


class InsightReport(Base):
    """Output of the Business Insight Agent."""

    executive_summary: str
    insights: list[BusinessInsight]
    key_drivers_plain_language: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    suggested_next_experiments: list[str] = Field(default_factory=list)


class ChartKind(str, Enum):
    CORRELATION_HEATMAP = "correlation_heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    SCATTER = "scatter"
    BAR = "bar"
    LINE = "line"
    ROC_CURVE = "roc_curve"
    PR_CURVE = "pr_curve"
    CONFUSION_MATRIX = "confusion_matrix"
    FEATURE_IMPORTANCE = "feature_importance"
    SHAP_SUMMARY = "shap_summary"
    RESIDUALS = "residuals"
    RESIDUAL_HISTOGRAM = "residual_histogram"
    LEARNING_CURVE = "learning_curve"
    PREDICTION_DISTRIBUTION = "prediction_distribution"
    CALIBRATION_CURVE = "calibration_curve"
    MISSINGNESS = "missingness"
    CLASS_BALANCE = "class_balance"
    LEADERBOARD = "leaderboard"
    TIME_SERIES_FORECAST = "time_series_forecast"
    PARTIAL_DEPENDENCE = "partial_dependence"


class ChartSpec(Base):
    kind: ChartKind
    title: str
    columns: list[str] = Field(default_factory=list)
    rationale: str = Field(description="What question this chart answers.")
    parameters: list[Param] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"


class ChartArtifact(Base):
    spec: ChartSpec
    html_path: str | None = None
    png_path: str | None = None
    json_path: str | None = None
    caption: str = ""
    rendered: bool = False
    error: str | None = None


class VisualizationPlan(Base):
    """Output of the Visualization Agent."""

    charts: list[ChartSpec]
    dashboard_narrative: str = ""


class VisualizationBundle(Base):
    artifacts: list[ChartArtifact] = Field(default_factory=list)
    dashboard_path: str | None = None
    narrative: str = ""


class DeploymentPattern(str, Enum):
    BATCH_INFERENCE = "batch_inference"
    REST_API = "rest_api"
    SERVERLESS = "serverless"
    STREAMING = "streaming"
    EDGE = "edge"
    EMBEDDED_SQL = "embedded_sql"


class DeploymentRecommendation(Base):
    pattern: DeploymentPattern
    rationale: str = Field(
        description="Grounded in the measured latency, throughput need, and model size."
    )
    estimated_latency_ms: float | None = None
    estimated_throughput_rps: float | None = None
    model_size_mb: float | None = None
    infrastructure_notes: str = ""
    monitoring_plan: list[str] = Field(default_factory=list)
    retraining_cadence: str = ""
    rollout_strategy: str = ""
    risks: list[str] = Field(default_factory=list)


class ReportSection(Base):
    heading: str
    order: int
    body_markdown: str
    chart_refs: list[str] = Field(default_factory=list)


class FinalReport(Base):
    """Output of the Report Agent."""

    title: str
    subtitle: str = ""
    executive_summary: str
    sections: list[ReportSection]
    deployment: DeploymentRecommendation
    appendix_notes: list[str] = Field(default_factory=list)

    def ordered_sections(self) -> list[ReportSection]:
        return sorted(self.sections, key=lambda s: s.order)


class ReportBundle(Base):
    markdown_path: str | None = None
    html_path: str | None = None
    pdf_path: str | None = None
    pptx_path: str | None = None
    json_path: str | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Run configuration & approvals
# ---------------------------------------------------------------------------


class ApprovalRequest(Base):
    request_id: str = Field(default_factory=lambda: _new_id("apr"))
    step_id: str
    agent: AgentName
    action_summary: str
    details: list[str] = Field(default_factory=list)
    affected_columns: list[str] = Field(default_factory=list)
    affected_row_estimate: int = 0
    severity: Severity = Severity.MEDIUM
    created_at: datetime = Field(default_factory=_utcnow)
    decision: Literal["pending", "approved", "rejected"] = "pending"
    decided_at: datetime | None = None
    decided_by: str | None = None
    note: str | None = None


class RunConfig(Base):
    """Everything that shapes a run, in one auditable object."""

    run_id: str = Field(default_factory=lambda: _new_id("run"))
    project: str = "default"
    source: DataSource
    target_column: str | None = None
    task_type_override: TaskType | None = None
    primary_metric_override: str | None = None
    time_budget_seconds: int = 900
    max_experiments: int = 8
    max_rows: int | None = Field(
        default=None, description="Sample cap for very large tables."
    )
    test_size: float = 0.2
    validation_size: float = 0.15
    cv_folds: int = 5
    random_state: int = 42
    require_approval: bool = Field(
        default=False,
        description="Pause before destructive operations and wait for a human.",
    )
    enable_tuning: bool = True
    enable_explainability: bool = True
    enable_self_improvement: bool = True
    max_replans: int = 2
    min_acceptable_score: float | None = None
    report_formats: list[str] = Field(
        default_factory=lambda: ["markdown", "html", "json"]
    )
    fairness_attributes: list[str] = Field(default_factory=list)
    notes: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventKind(str, Enum):
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_SKIPPED = "step_skipped"
    STEP_RETRIED = "step_retried"
    AGENT_THINKING = "agent_thinking"
    AGENT_DECISION = "agent_decision"
    LLM_CALL = "llm_call"
    ARTIFACT_WRITTEN = "artifact_written"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    REPLAN_TRIGGERED = "replan_triggered"
    METRIC_RECORDED = "metric_recorded"
    WARNING = "warning"
    LOG = "log"


class RunEvent(Base):
    event_id: str = Field(default_factory=lambda: _new_id("ev"))
    run_id: str
    sequence: int = 0
    kind: EventKind
    agent: AgentName | None = None
    step_id: str | None = None
    message: str = ""
    payload: list[Param] = Field(default_factory=list)
    at: datetime = Field(default_factory=_utcnow)
    duration_seconds: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cache_read_tokens: int | None = None
    cost_usd: float | None = None


class StepRecord(Base):
    step_id: str
    title: str = ""
    agent: AgentName | None = None
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    summary: str = ""


class UsageTotals(Base):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0


class RunSummary(Base):
    """The complete, serialisable outcome of a run."""

    run_id: str
    project: str = "default"
    status: RunStatus
    config: RunConfig
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float = 0.0
    error: str | None = None

    ingestion: IngestionResult | None = None
    profile: DatasetProfile | None = None
    understanding: DatasetUnderstanding | None = None
    problem: ProblemDefinition | None = None
    plan: ExecutionPlan | None = None
    plan_history: list[ExecutionPlan] = Field(default_factory=list)
    cleaning: CleaningPlan | None = None
    features: FeaturePlan | None = None
    model_selection: ModelSelection | None = None
    experiments: ExperimentLog | None = None
    tuning_decision: TuningDecision | None = None
    tuning: TuningResult | None = None
    explainability: ExplainabilityReport | None = None
    evaluation: EvaluationVerdict | None = None
    insights: InsightReport | None = None
    visualization_plan: VisualizationPlan | None = None
    visualizations: VisualizationBundle | None = None
    report: FinalReport | None = None
    report_bundle: ReportBundle | None = None

    steps: list[StepRecord] = Field(default_factory=list)
    approvals: list[ApprovalRequest] = Field(default_factory=list)
    usage: UsageTotals = Field(default_factory=UsageTotals)
    replans: int = 0
    warnings: list[str] = Field(default_factory=list)
    artifact_dir: str | None = None


# ---------------------------------------------------------------------------
# Dataset memory & natural-language Q&A
# ---------------------------------------------------------------------------


class DatasetFingerprint(Base):
    """Structural signature used to find similar past datasets."""

    fingerprint_id: str = Field(default_factory=lambda: _new_id("fp"))
    run_id: str
    project: str = "default"
    n_rows: int
    n_columns: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_text: int
    missing_fraction: float
    duplicate_fraction: float
    task_type: TaskType | None = None
    target_kind: ColumnKind | None = None
    imbalance_ratio: float | None = None
    column_names: list[str] = Field(default_factory=list)
    primary_metric: str = ""
    best_score: float | None = None
    best_family: ModelFamily | None = None
    winning_feature_ops: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class SimilarRun(Base):
    run_id: str
    similarity: float
    task_type: TaskType | None = None
    best_family: ModelFamily | None = None
    best_score: float | None = None
    primary_metric: str = ""
    why_similar: str = ""


class MemorySuggestion(Base):
    """Output of the dataset-memory lookup, fed to the Planner."""

    has_precedent: bool = False
    similar_runs: list[SimilarRun] = Field(default_factory=list)
    recommended_families: list[ModelFamily] = Field(default_factory=list)
    recommended_feature_ops: list[FeatureOp] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    narrative: str = ""


class QuestionAnswer(Base):
    """Output of the natural-language interface."""

    question: str
    answer: str = Field(description="Grounded in the run history; no speculation.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Concrete references to metrics, decisions, or events.",
    )
    confidence: Literal["high", "medium", "low"]
    caveats: list[str] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)


__all__ = [name for name in dir() if not name.startswith("_")]
