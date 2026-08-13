export const meta = {
  name: 'automl-architect-build',
  description: 'Build the AutoML Architect platform modules in parallel against a fixed core contract, then verify each imports.',
  phases: [
    { title: 'Build', detail: 'one agent per module, all coding against automl_architect/core/*' },
    { title: 'Verify', detail: 'import-check and fix each module in place' },
  ],
}

const ROOT = 'C:/Users/Edward Lai/Documents/AI Agent Data Modeler'
const PY = '"C:/Users/Edward Lai/Documents/AI Agent Data Modeler/.venv/Scripts/python.exe"'

const SHARED = `
# PROJECT: AutoML Architect

You are building one module of an autonomous multi-agent AI data scientist.
Repo root: ${ROOT}
Python interpreter (ALWAYS use this, never bare 'python'): ${PY}
Installed: python 3.11.9, pandas 3.0.5, numpy, scipy, scikit-learn 1.9.0, pyarrow,
openpyxl, pydantic 2.x, pydantic-settings, SQLAlchemy 2.x, duckdb, jinja2, typer,
rich, joblib, statsmodels, plotly, fastapi, uvicorn, sse-starlette, httpx, pytest,
xgboost, lightgbm, optuna, shap, kaleido, reportlab, python-pptx, markdown.
NOT installed: catboost, boto3, azure-storage-blob, google-cloud-storage, kaggle,
psycopg, PyMySQL, snowflake, mlflow. Treat those as OPTIONAL: import them lazily
inside the function that needs them and raise MissingDependencyError if absent.

## THE ARCHITECTURE, AND WHY

Claude reasons; deterministic Python computes. Agents receive measured facts and
return typed decisions with a mandatory rationale. Executors apply those
decisions with real pandas/sklearn code. No agent ever computes a statistic or
trains a model, and no executor ever makes an unexplained choice. That split is
the product: every number is reproducible and every choice is auditable.

## FILES THAT ALREADY EXIST — READ THEM FIRST, NEVER EDIT THEM

  automl_architect/core/schemas.py   <- ALL data contracts. Read this first.
  automl_architect/core/llm.py       <- LLMClient, PromptBlock, build_system
  automl_architect/core/agent.py     <- BaseAgent, HybridAgent
  automl_architect/core/state.py     <- RunState, DataSplits
  automl_architect/core/events.py    <- EventBus, EventKind helpers
  automl_architect/core/context.py   <- build_run_context (profile -> prompt text)
  automl_architect/core/errors.py    <- exception hierarchy
  automl_architect/config.py         <- Settings, get_settings, estimate_cost_usd
  automl_architect/__init__.py       <- lazy entry points (expects automl_architect/runner.py)

Read every symbol you use from schemas.py. Do NOT invent field names, do NOT
add fields to existing models, and do NOT edit any file listed above. If a
contract genuinely blocks you, work around it in your own module and note it in
your final report.

## HARD RULES

1. Agent-facing free-form config is 'list[Param]', never 'dict[str, Any]' —
   Claude's structured-output schema subset cannot express an untyped object.
   Convert with schemas.params_to_dict() / dict_to_params().
2. Never trust an agent-supplied column name. Filter against the real dataframe
   columns (BaseAgent.keep_known_columns does this) before touching pandas.
3. Every failure path must degrade, not crash the run. An optional dependency
   missing, one model family failing to fit, SHAP unavailable — log a warning
   via state.add_warning() and continue with reduced capability.
4. pandas 3.x: '.applymap' is gone (use '.map'), copy-on-write is default, and
   chained-assignment writes silently do nothing. Assign explicitly.
5. Use only real, documented library APIs. If unsure of a signature, check it
   with the interpreter rather than guessing.
6. Type-annotate public functions. Google-style docstrings on public functions
   and classes. Comment only to explain WHY, never to restate the code.
7. Every file you create must compile and import cleanly under the venv python.

## EXECUTOR INTERFACE CONTRACT (build to this exactly; other agents depend on it)

    # execution/splitter.py
    def make_splits(state: RunState) -> DataSplits
    def resolve_split_strategy(state: RunState) -> tuple[str, str]

    # execution/cleaning_ops.py
    def apply_cleaning_plan(state: RunState, plan: CleaningPlan) -> Any  # sets state.working_df, returns it

    # execution/feature_ops.py
    def apply_feature_plan(state: RunState, plan: FeaturePlan) -> Any  # sets state.feature_frame, returns it

    # execution/model_zoo.py
    def available_families(task: TaskType) -> list[ModelFamily]
    def is_available(family: ModelFamily) -> bool
    def build_estimator(family: ModelFamily, task: TaskType, params: dict, random_state: int = 42, **ctx) -> Any
    def default_search_space(family: ModelFamily, task: TaskType) -> dict
    def supports_proba(family: ModelFamily, task: TaskType) -> bool

    # execution/metrics.py
    def primary_metric_for(task: TaskType) -> str
    def higher_is_better(metric: str) -> bool
    def score_predictions(task: TaskType, y_true, y_pred, y_proba=None, labels=None) -> dict[str, float]
    def sklearn_scorer_name(metric: str, task: TaskType) -> str

    # execution/trainer.py
    def run_experiments(state: RunState) -> ExperimentLog   # trains every selected candidate
    def fit_final_model(state: RunState, family: ModelFamily, params: dict) -> Any

    # execution/tuner.py
    def run_tuning(state: RunState, decision: TuningDecision) -> TuningResult

    # execution/explainer.py
    def compute_explanations(state: RunState) -> ExplainabilityReport

    # execution/diagnostics.py
    def compute_diagnostics(state: RunState) -> DiagnosticsBundle  # dataclass you define in this module
      # must expose: bias_variance: BiasVarianceDiagnosis, calibration: CalibrationDiagnosis,
      # confidence_intervals: list[ConfidenceInterval], fairness: list[FairnessSlice],
      # residual_stats: dict[str, float], learning_curve: dict[str, list[float]],
      # error_examples: list[str], plus a to_prompt() -> str method rendering it all as text.

    # profiling/profiler.py
    def profile_dataframe(df, *, target: str | None = None, settings=None, sample_rows: int | None = None) -> DatasetProfile

    # ingestion/router.py
    def load_source(source: DataSource, *, max_rows: int | None = None) -> tuple[Any, IngestionResult]

    # reporting/charts.py
    def render_charts(state: RunState, plan: VisualizationPlan) -> VisualizationBundle

    # reporting/writer.py
    def write_report(state: RunState, report: FinalReport, formats: list[str]) -> ReportBundle

    # storage/repository.py
    class RunRepository:
        def __init__(self, database_url: str | None = None)
        def save_run(self, summary: RunSummary) -> None
        def get_run(self, run_id: str) -> RunSummary | None
        def list_runs(self, project: str | None = None, limit: int = 50) -> list[RunSummary]
        def append_events(self, run_id: str, events: list[RunEvent]) -> None
        def get_events(self, run_id: str, after: int = 0) -> list[RunEvent]
        def save_fingerprint(self, fp: DatasetFingerprint) -> None
        def find_similar(self, fp: DatasetFingerprint, limit: int = 5) -> list[SimilarRun]

## OUTPUT

Write real files to disk with Write/Edit. Verify each compiles:
  ${PY} -c "import <your.module>"
Return a terse report: files created, public symbols, anything you stubbed or
deviated from, and any contract mismatch you noticed in another module's area.
`

const AGENT_PROMPT_GUIDANCE = `
## WRITING AGENT PROMPTS — THIS IS THE PRODUCT, NOT BOILERPLATE

Each agent subclasses BaseAgent (or HybridAgent) from core/agent.py and implements:
  instructions(state) -> str    # the system prompt: role, method, output discipline
  build_prompt(state) -> str    # the user turn: measured facts + the concrete question
  postprocess(value, state)     # ground the output against the real schema
  apply(state, value)           # write into RunState
  decision_summary(value) -> str
  plus class attrs: name (AgentName), output_model, effort, max_tokens

instructions() quality determines the whole product's quality. Write prompts that:
  - Give the agent a specific expert identity and a METHOD, not just a role name.
    Not "You are a data cleaning expert" but the actual decision procedure a
    senior practitioner follows, in order, with the tradeoffs at each branch.
  - State how to choose between options using the evidence available. E.g. for
    imputation: skew magnitude decides mean vs median; cardinality decides mode
    vs a MISSING sentinel; missingness above ~40% argues for dropping the column
    or encoding missingness as its own signal rather than imputing.
  - Demand evidence-citing rationales. Show one worked example of a good
    rationale and one of a bad one, so the contrast is concrete.
  - Warn about the specific failure modes of THAT agent's job. Target encoding
    leaks without out-of-fold fitting. Lag features leak without a time-ordered
    split. Accuracy misleads on imbalanced targets. SMOTE before splitting leaks.
    Name these explicitly.
  - Tell it to prefer fewer, well-argued decisions over an exhaustive list.
  - Do NOT restate the JSON schema in prose. The schema is enforced by the API;
    spending prompt on it wastes tokens and invites contradiction.

build_prompt() should pull the facts the agent needs off state (profile, problem,
experiments, diagnostics) and render them as compact readable text. The shared
dataset digest is already in the cached system prefix via state.run_context — do
NOT repeat it. Add only what is new for this agent, then ask the question.

Set 'effort' deliberately: 'max' or 'xhigh' for planning, model selection, and
evaluation; 'high' for cleaning, features, insight; 'medium' for narration-heavy
work like visualization specs. Set max_tokens with room for thinking too
(Opus 5 counts thinking against max_tokens): 16000 typical, 32000 for planner
and report.
`

phase('Build')

const MODULES = [
  {
    key: 'profiling',
    label: 'profiling',
    prompt: `Build the deterministic profiling engine: the module that produces every
number the agents reason over. If this module is wrong or thin, every agent
downstream reasons about fiction.

Create:
  automl_architect/profiling/__init__.py
  automl_architect/profiling/profiler.py   -> profile_dataframe(...) per the contract
  automl_architect/profiling/stats.py      -> per-column statistics
  automl_architect/profiling/semantic.py   -> type/semantic inference
  automl_architect/profiling/leakage.py    -> leakage + quality detection

Populate EVERY field of DatasetProfile and ColumnProfile that the data supports.
Requirements:

1. Column kind inference (semantic.py): distinguish numeric continuous vs
   discrete, categorical nominal vs ordinal, boolean, datetime (including
   strings that parse as dates — try parsing, do not just trust dtype), free
   text vs short categorical (use mean token count and cardinality, not just
   length), identifier-like (near-unique + name hints like id/uuid/key/code),
   geographic (lat/lon value ranges AND name hints; also postal/country/city),
   constant. Also detect semantic types by regex on a sample: email, url, ipv4,
   uuid, phone, postal_code, currency, percentage.
2. Statistics (stats.py): missing counts, unique counts, cardinality ratio,
   memory, mean/std/min/max, all 7 quantiles, skewness, kurtosis, variance,
   near-zero-variance flag, zero and negative fractions, IQR outlier bounds and
   counts, top-K value counts with fractions, string length and token stats,
   datetime min/max/inferred frequency/gap count/monotonicity.
3. Correlations: Pearson AND Spearman for numeric pairs; use Cramer's V for
   categorical-categorical and correlation ratio (eta) for categorical-numeric
   so mixed-type tables are covered. Report top_correlations,
   target_correlations (sorted by absolute strength), and
   highly_correlated_pairs (abs > 0.95, the multicollinearity set).
4. Leakage (leakage.py): score each feature's association with the target
   (mutual information, plus AUC for binary targets and correlation for
   regression). Flag near-perfect single-feature association as leakage with a
   severity and a human reason. Also flag: a feature that is a monotone
   transform of the target, duplicate-of-target columns, and columns whose name
   suggests post-outcome knowledge (contains 'churn','cancel','refund','exit',
   'outcome','label','target','result','_after','post_') while ALSO being
   highly predictive. Name-only matches are NOT leakage — require the
   statistical signal too, and say so in the reason.
5. Quality issues: high missingness, constant columns, near-zero variance,
   duplicate rows, skewed numerics, high-cardinality categoricals, mixed types
   in one column, single-row classes in the target, imbalanced target, tiny
   dataset relative to feature count (p > n), datetime gaps.
6. Performance: sample to settings.max_profile_rows for expensive passes
   (correlations, outliers, mutual info) but ALWAYS report true n_rows. Set
   profile_seconds. Never let a single column's failure abort the profile —
   wrap per-column work and degrade.

Write and run a self-test script (delete it after, or leave under tests/) that
profiles a synthetic frame exercising every branch: skewed numeric, missing
values, constant column, high-cardinality id, datetime, free text, lat/lon,
imbalanced boolean target, an obvious leakage column, and duplicate rows.
Confirm the flags fire correctly and print the profile summary.`,
  },
  {
    key: 'ingestion',
    label: 'ingestion',
    prompt: `Build the data ingestion layer: 13 source kinds behind one router, each
inferring schema and validating what it loaded.

Create:
  automl_architect/ingestion/__init__.py
  automl_architect/ingestion/base.py     -> Connector ABC + registry decorator
  automl_architect/ingestion/files.py    -> csv, excel, json, parquet
  automl_architect/ingestion/sql.py      -> sql/postgres/mysql/snowflake/databricks/duckdb via SQLAlchemy
  automl_architect/ingestion/cloud.py    -> s3, azure_blob, gcs
  automl_architect/ingestion/rest.py     -> rest_api (httpx; JSON -> frame, pagination, auth header)
  automl_architect/ingestion/kaggle.py   -> kaggle dataset download
  automl_architect/ingestion/router.py   -> load_source(...) per the contract

Requirements:
1. CSV must be robust on real-world files: sniff the delimiter, detect encoding
   (try utf-8, utf-8-sig, latin-1 in order), handle a thousands separator, and
   try to parse object columns that look like dates. Excel: read a named or
   indexed sheet, default to the first. JSON: handle records, a nested object
   with a data key, and JSON Lines. Parquet via pyarrow.
2. SQL: build the URL from DataSource.uri or from options + credentials read
   from the env var names in DataSource.secret_env. NEVER log or store a
   credential value, and never put one in IngestionResult. Support either a
   query or a bare table name. duckdb works without extra deps; the others need
   optional drivers, so import lazily and raise MissingDependencyError with the
   correct 'pip install' hint.
3. Cloud: prefer the fsspec/s3fs path where available so pandas can read the
   URI directly; otherwise use the vendor SDK. Support a public URL fallback
   for anonymous objects. All optional deps, lazily imported.
4. REST: GET or POST, JSON body, bearer/api-key header from env, follow simple
   page/offset or cursor pagination up to a cap, normalise nested JSON with
   pandas.json_normalize.
5. Every connector returns (DataFrame, IngestionResult) with n_rows, n_columns,
   schema_fields (name, inferred dtype, nullable, 3 sample values as strings),
   bytes_in_memory, load_seconds, and truncated set when max_rows capped it.
6. Validation, into IngestionResult.validation_errors / validation_warnings:
   error on an empty frame or zero columns; warn on duplicate column names
   (and de-duplicate them), fully-empty columns, unnamed/Unnamed index columns,
   whitespace in headers (strip them), a single column (suggests a delimiter
   mis-sniff), and a mixed-type column. Also normalise column names minimally:
   strip whitespace, and make duplicates unique — but do NOT lowercase or
   mangle, since users refer to columns by their real names.
7. Registry: a decorator maps SourceKind -> Connector so router.load_source
   dispatches without an if-chain, and UnsupportedSourceError names the kinds
   that ARE registered.

Also create automl_architect/ingestion/dataframe.py registering SourceKind.DATAFRAME,
which reads a frame passed through DataSource.options — needed for library use
and for tests.

Self-test: write CSV/JSON/Parquet/Excel fixtures to a temp dir, load each
through load_source, and print the IngestionResult. Confirm the duplicate-header
and encoding paths work.`,
  },
  {
    key: 'agents_understand',
    label: 'agents:understand',
    prompt: `${AGENT_PROMPT_GUIDANCE}

Build the three agents that establish what the problem IS. These have the
highest leverage on final quality — a wrong task type or a generic plan poisons
everything after it.

Create:
  automl_architect/agents/__init__.py    (registry: AGENTS dict AgentName -> class, get_agent(name), plus __all__.
                                          Import lazily inside get_agent so a broken sibling module cannot
                                          break the whole package import.)
  automl_architect/agents/dataset.py     -> DatasetAgent, output DatasetUnderstanding
  automl_architect/agents/problem.py     -> ProblemAgent, output ProblemDefinition
  automl_architect/agents/planner.py     -> PlannerAgent, output ExecutionPlan

DatasetAgent (effort 'high', max_tokens 16000):
Its job is a senior data scientist's EDA writeup, not a stats dump — the stats
are already in the cached context. The narrative must interpret: what this data
appears to be, what one row represents, which relationships look real vs
spurious, which columns are dangerous and why. Require it to assess every column
(role + predictive potential + concerns) and to rank candidate target columns
when the operator did not name one. data_readiness must follow from the measured
missingness, duplication, and leakage findings, not vibes.
postprocess: filter column_assessments and suggested_target_columns to real
columns; if the operator named a target, force it to the front of the ranking.

ProblemAgent (effort 'xhigh', max_tokens 16000):
Infers the task type from target cardinality, dtype, and distribution, plus the
presence of a usable temporal column. Its rationale must name the evidence, in
the style of "the target is binary and categorical, indicating a binary
classification problem". It must also choose the primary metric and justify it
against the class balance — accuracy on a 95/5 target is a trap, and the prompt
should say so; prefer roc_auc or average_precision or f1 there, and explain the
choice. For regression, reason about whether the target's skew and outliers
favour RMSE, MAE, or a log-target. Set positive_class for binary tasks, and
temporal_column/horizon for forecasting. Consider and record alternatives.
postprocess: honour config.task_type_override and config.primary_metric_override
absolutely; validate target/temporal/group columns exist; if the chosen task
type is not TaskType.is_supported, keep the honest diagnosis but add a warning
via state.add_warning that execution will fall back, and map to the nearest
supported type in a new field-free way (record it via state.extras).

PlannerAgent (effort 'max', max_tokens 32000):
The brain. It writes the whole execution strategy BEFORE any computation. Steps
must be drawn from the real agent set (AgentName) and must adapt to THIS
dataset: skip an imputation step when nothing is missing, add a temporal
feature step only when a temporal column exists, add a leakage-removal step
only when leakage was found, and choose the split strategy implied by the data.
Require dataset_specific_adaptations to be genuinely specific — the prompt
should say that a plan which would suit any dataset is a plan it has not thought
about. Mark steps destructive=True when they drop columns or rows, since that
gates human approval. Require a fallback_strategy that names what to change if
evaluation rejects the model.
The planner also handles REPLANNING: when state.evaluation exists and rejected
the model, build_prompt must include the previous plan, the measured scores, the
evaluation's weaknesses and recommended_action, and what was already tried, and
instruct it to produce a MATERIALLY different plan rather than a reshuffle.
Set revision and revision_reason. Also feed state.memory (MemorySuggestion) into
the prompt when present, framed as prior evidence to weigh, not orders.
postprocess: renumber 'order' to a dense 1..N sequence, drop steps whose agent
is not a real AgentName, drop dependencies pointing at non-existent step_ids,
and guarantee the plan contains the mandatory terminal steps (evaluation,
insight, report) — appending them with an explanatory rationale if the model
omitted them.

Verify each module imports.`,
  },
  {
    key: 'agents_prepare',
    label: 'agents:prepare',
    prompt: `${AGENT_PROMPT_GUIDANCE}

Build the three agents that shape the data and choose the models.

Create:
  automl_architect/agents/cleaning.py         -> CleaningAgent, output CleaningPlan
  automl_architect/agents/features.py         -> FeatureAgent, output FeaturePlan
  automl_architect/agents/model_selection.py  -> ModelSelectionAgent, output ModelSelection
(agents/__init__.py is owned by another agent — do not create it.)

CleaningAgent (effort 'high', max_tokens 16000):
The product requirement is that NO transformation is applied without a reason
tied to evidence. The prompt must teach the actual decision procedure:
 - Imputation choice follows the distribution: near-symmetric -> mean; skewed
   (|skew| > ~1) -> median, because the mean sits away from the bulk; categorical
   -> mode when one value dominates, else an explicit MISSING category, because
   inventing the mode fabricates signal; temporal -> forward fill only when rows
   are genuinely ordered in time.
 - Missingness above ~40-50% usually argues for dropping the column or keeping
   missingness itself as a binary feature, not for imputing 45% of the values.
 - Missing values in the TARGET are dropped, never imputed — imputing the label
   is fabricating ground truth.
 - Outliers: clip rather than delete by default, and only act when the outliers
   are implausible values rather than a genuine heavy tail. Deleting rows on a
   real heavy tail throws away the signal you are trying to model.
 - Duplicates: drop exact duplicate rows, but say what a duplicate MEANS here
   (identical customer records vs a legitimately repeated measurement).
 - Leakage columns from the profile: drop them, and the rationale must explain
   the mechanism, not just cite the score.
 - Include skipped_considerations: transformations deliberately NOT applied and
   why. That field is a quality signal — populate it.
Show a worked good rationale ("skewness of 3.42 with 4% IQR outliers puts the
mean well above the median, so the median imputes a more representative value")
against a bad one ("median is standard practice for numeric columns").
postprocess: filter to real columns; drop any decision targeting the target
column for imputation; force destructive=True on drop/remove actions; ensure
columns_to_drop is consistent with the decisions.

FeatureAgent (effort 'high', max_tokens 16000):
Proposes engineered features with a stated HYPOTHESIS about the mechanism, not a
catalogue of transforms. Teach when each op earns its place: date decomposition
and cyclical encoding when a temporal column exists and periodicity is plausible;
lags/rolling/diff ONLY for forecasting or genuinely time-ordered data, with an
explicit warning that they leak unless the split is time-ordered; one-hot for
low cardinality; target encoding for high cardinality WITH the mandatory caveat
that it must be fitted out-of-fold or it leaks the target; frequency encoding as
the safer high-cardinality option; log/sqrt/boxcox for right-skewed positive
numerics; interactions and ratios only where a domain mechanism makes them
meaningful; TF-IDF for genuine free text; PCA/SVD only when dimensionality is
actually a problem, noting the explainability cost.
Require each decision to carry the leakage/overfitting risk in the 'risk' field.
Require priority ordering, and tell it that ten well-argued features beat sixty
speculative ones — the feature count is not the score.
postprocess: filter to real columns; drop lag/rolling/diff/expanding ops when
the task is not temporal and no temporal column exists (warn when doing so);
drop target-encoding decisions that reference the target as an input column.

ModelSelectionAgent (effort 'xhigh', max_tokens 16000):
Must NOT default to gradient boosting. The prompt must force a comparative
argument grounded in this dataset's n, p, n/p ratio, feature types, class
balance, missingness, and the explainability requirement. Teach the real
tradeoffs: linear/logistic models as an interpretable strong baseline that wins
on small-n or near-linear signal; tree ensembles for mixed types and
interactions; boosted trees when n is large enough to avoid overfitting;
regularised linear when p approaches or exceeds n; SVM only at modest n because
it scales poorly; KNN only with a meaningful distance metric and scaled
features; neural nets rarely justified on tabular data of this size and it
should say so rather than including one for completeness.
It MUST always include a trivial baseline (BASELINE_DUMMY) marked
is_baseline=True — a model that cannot beat the majority class or the mean has
learned nothing, and the report needs that comparison to be honest.
It must choose validation_strategy from the data: stratified k-fold for
imbalanced classification, grouped splits when a group column exists (to stop
the same entity appearing in train and test), expanding-window time-series
splits for temporal data, and say why.
Only propose families that exist in ModelFamily. Rank them, with excluded
families and exclusion_rationale populated.
postprocess: drop candidates whose family is unavailable in the environment
(import execution.model_zoo.is_available lazily inside postprocess and warn on
each drop); guarantee at least one baseline candidate, injecting one if absent;
truncate to config.max_experiments candidates preserving rank and the baseline;
renumber ranks densely.

Verify each module imports.`,
  },
  {
    key: 'agents_evaluate',
    label: 'agents:evaluate',
    prompt: `${AGENT_PROMPT_GUIDANCE}

Build the four agents that run experiments, tune, explain, and judge. Three of
these are HybridAgent subclasses: deterministic code computes the facts, then
the agent interprets them.

Create:
  automl_architect/agents/experiment.py  -> ExperimentAgent(HybridAgent), output ExperimentLog
  automl_architect/agents/tuning.py      -> TuningAgent, output TuningDecision
  automl_architect/agents/explain.py     -> ExplainAgent(HybridAgent), output ExplainabilityReport
  automl_architect/agents/evaluation.py  -> EvaluationAgent, output EvaluationVerdict
(agents/__init__.py is owned by another agent — do not create it.)

ExperimentAgent (HybridAgent, effort 'medium', max_tokens 16000):
compute() calls execution.trainer.run_experiments(state) — real training, real
cross-validation, real timings — and stores the ExperimentLog. The agent then
returns an ExperimentLog whose numeric fields are copied VERBATIM from the
computed one; only leaderboard_notes is authored. Enforce that in postprocess:
overwrite every numeric field and the results list from the computed log so the
model physically cannot alter a measured score. leaderboard_notes should read
like an analyst's comparison — which model won, by how much over the baseline,
whether the margin is meaningful given the CV standard deviation, and what the
train-time cost was.

TuningAgent (effort 'high', max_tokens 16000):
Decides whether tuning is WORTH IT before doing any. The prompt must make it a
cost/benefit judgement: with a small dataset, tuning mostly fits the validation
noise; when the gap between the best model and the baseline is already large,
tuning yields little; when the remaining time budget is short, spend it
elsewhere; when two candidates are within a CV standard deviation of each other,
tuning may just reorder noise. Teach method choice: grid only for a tiny
discrete space, random for a broad cheap sweep, TPE/Optuna when trials are
expensive and the space is continuous, halving when early trials are cheap to
kill. It defines the search space per hyperparameter with a rationale for each
range — not a textbook default grid. Respect state.time_remaining and set
n_trials and timeout_seconds inside it.
postprocess: force worthwhile=False with a reason when config.enable_tuning is
False, when state.time_remaining is under ~90s, or when no successful experiment
exists; clamp n_trials and timeout_seconds to the remaining budget; ensure
target_family is one that actually trained successfully.

ExplainAgent (HybridAgent, effort 'high', max_tokens 16000):
compute() calls execution.explainer.compute_explanations(state) — real SHAP
where available, permutation importance always, partial dependence for top
features. The agent authors narrative and plain_language_explanations, which are
the business-readable sentences the spec asks for: percentage-of-importance
phrasing like "customer tenure contributes approximately 31% of churn prediction
importance", plus the DIRECTION of each effect where SHAP gives it, and an
honest caveat that importance is not causation. postprocess must copy all
numeric attributions and file paths verbatim from the computed report so the
model cannot invent an importance value, and must renormalise importances to sum
to 1.0. If SHAP was unavailable, the narrative must say which method was used
instead rather than implying SHAP ran.

EvaluationAgent (effort 'max', max_tokens 16000):
The quality gate, and the trigger for the self-improvement loop. build_prompt
calls execution.diagnostics.compute_diagnostics(state) and renders its
to_prompt() text: train/validation/test scores, the generalisation gap,
calibration, bootstrap confidence intervals, residual statistics, learning-curve
points, fairness slices, and error examples. Teach it to read them: a large
train-test gap is overfitting; uniformly poor scores are underfitting or a
signal-free feature set; wide confidence intervals mean the dataset is too small
to distinguish models; a score that barely beats the baseline means the model
learned little regardless of its absolute value; strong per-slice disparity is a
fairness problem even when the aggregate looks fine.
recommended_action drives control flow, so the prompt must be explicit that
choosing retry_feature_engineering / retry_model_selection / retry_cleaning
causes the orchestrator to REPLAN and re-run, and that it should only do so when
it can name a specific change likely to help — specific_improvements must be
concrete and actionable, not "try more features".
postprocess: force acceptable=False if the best score fails
config.min_acceptable_score; force recommended_action='accept' when
config.enable_self_improvement is False or state.replans >= config.max_replans
(a retry that cannot run is a lie about what will happen next) and note that
clamp in action_rationale; copy the computed diagnostic numbers verbatim over
whatever the model returned.

Verify each module imports.`,
  },
  {
    key: 'agents_deliver',
    label: 'agents:deliver',
    prompt: `${AGENT_PROMPT_GUIDANCE}

Build the three agents that turn technical results into things a human uses.

Create:
  automl_architect/agents/insight.py        -> InsightAgent, output InsightReport
  automl_architect/agents/visualization.py  -> VisualizationAgent, output VisualizationPlan
  automl_architect/agents/report.py         -> ReportAgent, output FinalReport
  automl_architect/agents/qa.py             -> QAAgent, output QuestionAnswer (natural-language interface)
(agents/__init__.py is owned by another agent — do not create it.)

InsightAgent (effort 'high', max_tokens 16000):
Translates metrics and feature attributions into statements a decision-maker can
act on. The prompt must forbid restating technical output: not "income has
importance 0.41" but "customers with higher annual income churn substantially
less, so retention spend is better aimed at the middle-income band where
engagement is already declining". Every insight needs supporting_evidence naming
the metric or attribution it rests on, a recommended_action, and a quantified
expected_value where the data supports one. Require caveats — correlation is not
causation, the model reflects the historical period it was trained on, slices
with few rows are unreliable — and forbid overclaiming beyond what the measured
scores support. Audience-tag each insight.
build_prompt must include the problem's business_objective, the winning model and
its scores versus the baseline, the top feature attributions with direction, the
evaluation verdict and weaknesses, and the class balance or target distribution.
postprocess: filter feature references to real columns; drop insights whose
confidence is 'high' while the evaluation verdict was unacceptable, downgrading
them to 'low' rather than deleting, and note it.

VisualizationAgent (effort 'medium', max_tokens 16000):
Chooses which charts actually answer a question about THIS run, from ChartKind.
The prompt must gate by task: ROC/PR/confusion/calibration only for
classification; residual plots and residual histograms only for regression;
forecast plots only for time series. Every spec needs a rationale stating the
question the chart answers — a chart nobody has a question for is noise.
Require the columns named in each spec to exist. Ask for 8-14 charts prioritised,
not everything possible, and a dashboard_narrative that reads as a guided tour.
postprocess: filter to real columns, drop kinds that do not fit the task type
(warn on each), deduplicate by (kind, columns), and guarantee the essential ones
for the task are present (feature importance and leaderboard always; ROC plus
confusion matrix for classification; residuals for regression).

ReportAgent (effort 'xhigh', max_tokens 32000):
Assembles the deliverable with EXACTLY the sections the spec names, in order:
Executive Summary, Dataset Overview, Methodology, Cleaning Decisions, Feature
Engineering, Models Tested, Evaluation, Business Recommendations, Deployment
Suggestions. Section bodies are markdown, and the agent should use real markdown
tables for the leaderboard, the cleaning decision log, and the feature list —
those read far better as tables than prose. Each cleaning and feature section
must carry the rationale that was recorded, because the audit trail IS the
deliverable.
It also produces the DeploymentRecommendation, and the prompt must ground that
in the MEASURED numbers: predict_seconds and model_size_bytes from the winning
experiment decide batch vs REST vs serverless vs edge; a large model plus tight
latency argues against serverless cold starts; a tiny model with high throughput
suits edge or embedded SQL. Require a monitoring plan naming the specific drift
signals to watch for this dataset, and a retraining cadence justified by the
data's temporal span and drift risk.
Tell it the executive summary is written for someone who will read nothing else:
lead with the outcome and the decision it enables, not the methodology.
postprocess: renumber section order densely; verify all nine required headings
exist and synthesise any missing one from state rather than shipping a gap;
never let it claim deployment readiness when state.evaluation.acceptable is
False.

QAAgent (effort 'high', max_tokens 16000):
Answers free-form questions about a completed run ("why did accuracy decrease?",
"what happens if I remove income?") strictly from the run's recorded history.
It takes the question from state.extras['question']. build_prompt must render
the evidence: the plan and any revisions, cleaning and feature decisions with
rationales, the full experiment leaderboard, the tuning outcome, feature
attributions, the evaluation verdict, and the recent event log
(state.bus.events, most recent ~80, rendered compactly).
The prompt's central rule: answer ONLY from that history. For a counterfactual
it cannot test, it must say what the evidence suggests and what would have to be
run to actually answer, rather than inventing a result. Populate evidence[] with
concrete references and caveats[] with the limits of the answer.
apply(): store the QuestionAnswer in state.extras['answer'].

Verify each module imports.`,
  },
  {
    key: 'exec_prepare',
    label: 'execution:prepare',
    prompt: `Build the executors that apply cleaning and feature decisions, and that split
the data. This is where an agent's stated intent becomes a real dataframe, so
correctness here is what separates an explainable pipeline from a plausible-looking one.

Create:
  automl_architect/execution/__init__.py  (re-export the public functions from every
      execution module named in the contract, importing lazily/defensively so a
      missing optional dependency in one module does not break the package import)
  automl_architect/execution/cleaning_ops.py  -> apply_cleaning_plan(state, plan)
  automl_architect/execution/feature_ops.py   -> apply_feature_plan(state, plan)
  automl_architect/execution/splitter.py      -> make_splits(state), resolve_split_strategy(state)

cleaning_ops.py:
Implement every CleaningAction and MissingStrategy in schemas.py. Apply
decisions in a safe order regardless of the order the agent listed them:
target-row drops first, then column drops, then duplicates, then dtype/datetime
parsing, then imputation, then outlier handling. Record each applied action as a
human-readable line in state.applied_cleaning and each dropped column in
state.dropped_columns. Emit an event per applied action.
Critical correctness points:
 - NEVER impute the target column. Drop rows where the target is null.
 - Fit imputation statistics on the data being cleaned, and store the fitted
   values on state.extras['cleaning_fill_values'] so scoring new data later can
   reuse them rather than recomputing (that recomputation is a subtle train/serve skew).
 - KNN and iterative imputation are expensive: guard by row count and column
   count, fall back to median/mode with a warning above the guard.
 - Clip outliers to the profile's IQR bounds by default; only remove rows when
   the decision explicitly says REMOVE_OUTLIER_ROWS, and never remove more than
   ~5% of rows without a warning.
 - A decision that would empty the frame or drop the target must be refused with
   a warning, not executed.
 - Return the cleaned frame AND set state.working_df.

feature_ops.py:
Implement every FeatureOp. Generate deterministic, collision-free output column
names and record them in state.applied_features. Leakage discipline is the whole
point of this file:
 - TARGET_ENCODE must be out-of-fold (KFold, or the split strategy's folds) with
   smoothing toward the global mean; fitting it on the full column leaks the
   target and inflates every downstream score. Store the fitted mapping for reuse.
 - LAG / ROLLING / DIFF / EXPANDING must sort by the temporal column first,
   group by the group column when one exists, and must refuse to run when no
   temporal column is available (warn and skip).
 - Scaling, PCA/SVD, SELECT_K_BEST, and VARIANCE_THRESHOLD are fit-on-train
   concerns. Either build them into the sklearn pipeline handed to the trainer
   (preferred — put the transformer on state.preprocessor) or fit them on train
   only. Do not fit any of them on the full dataset before splitting.
 - DROP_CORRELATED uses the profile's highly_correlated_pairs, keeping the
   member of each pair more correlated with the target.
 - TEXT_TFIDF must cap max_features and be a pipeline transformer, not a
   pre-expanded dense frame.
 - Wrap each op individually: one failing op warns and is skipped, it does not
   abort the run.
Set state.feature_frame and state.feature_names.

splitter.py:
resolve_split_strategy inspects the problem and profile and returns
(strategy, rationale) choosing among: 'stratified' (imbalanced or multiclass
classification), 'grouped' (a group column exists — the same entity must never
appear in both train and test), 'temporal' (forecasting or a temporal column
present: split by time order, never randomly, or the model trains on the future),
'random'. make_splits produces train/validation/test honouring
config.test_size, config.validation_size, and config.random_state, sets
DataSplits.strategy and .rationale, and handles the edge cases: a class with
fewer members than the fold count (fall back from stratified with a warning), a
dataset too small to justify three partitions (skip validation, note it), and a
target needing label encoding for classification (store the encoder on
state.label_encoder).

Self-test each file against a synthetic frame and print what was applied.`,
  },
  {
    key: 'exec_train',
    label: 'execution:train',
    prompt: `Build the model zoo, the training loop, the metric layer, and the tuner. Every
score the whole system reports comes from here, so measurement honesty is the
requirement above all others.

Create:
  automl_architect/execution/model_zoo.py  -> available_families, is_available, build_estimator, default_search_space, supports_proba
  automl_architect/execution/metrics.py    -> primary_metric_for, higher_is_better, score_predictions, sklearn_scorer_name
  automl_architect/execution/trainer.py    -> run_experiments(state), fit_final_model(state, family, params)
  automl_architect/execution/tuner.py      -> run_tuning(state, decision)
(execution/__init__.py is owned by another agent — do not create it.)

model_zoo.py:
Map every ModelFamily to a real estimator for each applicable TaskType.
sklearn covers linear/logistic/ridge/lasso/elastic_net/trees/forests/extra_trees/
gradient_boosting/hist_gradient_boosting/svm/knn/naive_bayes/neural_network(MLP)/
kmeans/dbscan/gaussian_mixture/isolation_forest/local_outlier_factor/
one_class_svm/baseline_dummy. xgboost, lightgbm, catboost are optional: probe
importability once at module scope and have is_available() reflect it, so the
Model Selection Agent's postprocess can drop what is not installed. Time-series
families (seasonal_naive, theta, exponential_smoothing, sarimax) use statsmodels,
with seasonal_naive implemented directly as a sklearn-compatible estimator.
Set sensible defaults: n_jobs from settings, random_state, and for boosted trees
verbosity off. default_search_space returns ranges appropriate to the family and
task, used as a fallback when the Tuning Agent's space is unusable.

metrics.py:
Classification: accuracy, balanced_accuracy, precision, recall, f1 (binary and
weighted/macro variants), roc_auc (with the multiclass ovr form), average_precision,
log_loss, matthews_corrcoef, cohen_kappa. Regression: rmse, mae, mape (guarding
divide-by-zero on near-zero actuals), r2, explained_variance, median_absolute_error.
Clustering: silhouette, calinski_harabasz, davies_bouldin. Anomaly: whatever is
computable without labels. higher_is_better must be correct for every metric —
this drives model selection, so getting rmse or log_loss backwards would silently
pick the worst model. score_predictions must never raise: a metric that cannot be
computed for the given inputs is omitted, not faked.

trainer.py — run_experiments(state):
For each candidate in state.model_selection.candidates, in rank order:
 - build the estimator, wrap it with state.preprocessor in a sklearn Pipeline
   when one exists so all fitting happens inside cross-validation folds;
 - cross-validate using the fold strategy implied by state.splits.strategy
   (StratifiedKFold / GroupKFold / TimeSeriesSplit / KFold);
 - fit on train, predict on validation (falling back to test when there is no
   validation partition), and score with score_predictions;
 - measure train_seconds, predict_seconds, model_size_bytes (via joblib to a
   BytesIO), n_features_in, and peak memory with tracemalloc;
 - persist the fitted model to state.artifact_path('models', ...) and record
   artifact_path;
 - on failure, record ExperimentResult(failed=True, error=...) and CONTINUE.
   One broken family must not end the run.
Respect config.max_experiments and state.time_remaining, stopping early with a
warning rather than blowing the budget. Choose the best by primary_score using
higher_is_better. Set ExperimentLog.best_experiment_id, primary_metric,
higher_is_better. Set state.best_model and state.best_pipeline to the winner.
Raise NoViableModelError only when EVERY candidate failed.
fit_final_model refits one family with given params on train+validation and
returns the fitted pipeline.

tuner.py — run_tuning(state, decision):
Honour decision.worthwhile (return TuningResult(ran=False, skipped_reason=...)
when False). Implement grid, random, halving_random via sklearn, and
bayesian/optuna_tpe via optuna (lazily imported; fall back to random search with
a warning when optuna is missing). Translate decision.search_space
(SearchSpaceEntry list, with kinds int/float/log_float/categorical) into the
chosen backend's space, ignoring entries whose parameter the estimator does not
accept — warn on each rather than crashing. Enforce decision.timeout_seconds and
min(decision.n_trials, budget), report n_trials_completed, best_params,
best_score, baseline_score (the untuned score for that family), improvement, and
trial_scores. If tuning IMPROVED on the baseline, refit and update
state.best_model / state.best_pipeline and append a tuned ExperimentResult to
state.experiments; if it did not improve, say so and leave the untuned winner in
place — reporting a worse tuned model as the winner would be a measurement lie.

Self-test: build a synthetic classification frame and a regression frame, run a
2-candidate experiment loop end to end, and print the leaderboard. Confirm a
deliberately broken candidate is recorded as failed without aborting.`,
  },
  {
    key: 'exec_explain',
    label: 'execution:explain',
    prompt: `Build the explainability and diagnostics executors — the modules that make the
Explainability and Evaluation agents' claims real rather than plausible.

Create:
  automl_architect/execution/explainer.py    -> compute_explanations(state) -> ExplainabilityReport
  automl_architect/execution/diagnostics.py  -> compute_diagnostics(state) -> DiagnosticsBundle
(execution/__init__.py is owned by another agent — do not create it.)

explainer.py:
 - Permutation importance ALWAYS (sklearn.inspection.permutation_importance on
   the validation/test partition, not train — importance measured on training
   data reflects memorisation).
 - SHAP when the shap package imports: TreeExplainer for tree/boosted models,
   LinearExplainer for linear ones, KernelExplainer only as a last resort on a
   small background sample because it is very slow. Cap the sample size, guard
   with a timeout-ish row budget, and set shap_available accordingly. Save a
   summary plot PNG under state.artifact_path('charts', ...) via matplotlib with
   the Agg backend.
 - Native feature_importances_ or coef_ as global_attributions when available,
   otherwise fall back to the permutation values.
 - Partial dependence for the top ~4 features via sklearn PartialDependenceDisplay,
   saved as PNGs, paths recorded.
 - Counterfactuals: for the top 2-3 features, perturb a representative row
   (e.g. to its p10 and p90) and record the prediction change as a
   Counterfactual with a readable description. This must use the real fitted
   pipeline, and be skipped with a warning if prediction fails.
 - Normalise importances to sum to 1.0 and map feature names back to original
   column names where one-hot expansion renamed them, so the report speaks the
   user's vocabulary rather than 'onehot__city_Paris'.
 - Every step individually guarded: SHAP failing must leave permutation
   importance intact.

diagnostics.py:
Define a DiagnosticsBundle dataclass exposing exactly: bias_variance
(BiasVarianceDiagnosis), calibration (CalibrationDiagnosis),
confidence_intervals (list[ConfidenceInterval]), fairness (list[FairnessSlice]),
residual_stats (dict[str, float]), learning_curve (dict[str, list[float]]),
error_examples (list[str]), plus to_prompt() -> str.
Compute:
 - bias/variance: score on train, validation, and test with the primary metric;
   the gap; and a verdict of underfitting / good_fit / overfitting /
   inconclusive using sensible thresholds that you document in a comment.
 - calibration: for classifiers with predict_proba, Brier score and expected
   calibration error over ~10 bins, with a readable verdict. Mark
   applicable=False otherwise.
 - confidence intervals: bootstrap the test predictions (a few hundred
   resamples, capped for runtime) for the primary and one or two secondary
   metrics.
 - fairness: for each attribute in config.fairness_attributes (skipping ones not
   present, with a warning), compute the primary metric per slice and the delta
   versus overall, skipping slices below a minimum row count since a metric on
   9 rows is noise.
 - residuals (regression): mean, std, skew of residuals, a normality statistic,
   and a heteroscedasticity signal (correlation of |residual| with the prediction).
 - learning curve: sklearn learning_curve over ~5 train sizes, capped for
   runtime, returning train and validation score lists plus the sizes.
 - error_examples: a handful of the worst-predicted rows rendered as short
   readable strings (top misclassifications, or largest absolute residuals),
   which is what lets the Evaluation Agent say something concrete about WHERE
   the model fails.
to_prompt() renders all of the above as compact labelled text for the agent
prompt. Everything guarded — a failed diagnostic yields a note in the bundle,
never an exception that kills the run.

Self-test both modules against a small fitted pipeline for classification and
for regression, and print the rendered to_prompt() output.`,
  },
  {
    key: 'orchestrator',
    label: 'orchestrator',
    prompt: `Build the orchestration engine and the top-level entry point. This is the
component that makes the platform autonomous: it owns workflow state, dispatch,
retries, failure handling, human approval, and the self-improvement loop.

Create:
  automl_architect/orchestrator/__init__.py
  automl_architect/orchestrator/graph.py     -> the canonical step graph
  automl_architect/orchestrator/policies.py  -> retry / approval / replan / budget policies
  automl_architect/orchestrator/engine.py    -> Orchestrator
  automl_architect/runner.py                 -> AutoMLArchitect facade + analyse(...) convenience function

graph.py:
Define the canonical pipeline as ordered step definitions, each with a step_id,
title, owning AgentName or executor callable, whether it is skippable, whether
it is destructive, and its dependencies. Canonical order: ingest, profile,
understand, identify_problem, plan, clean, engineer_features, split,
select_models, run_experiments, tune, explain, evaluate, insights, visualise,
report. Provide a way to reconcile the Planner's ExecutionPlan against this
graph: the plan decides WHICH steps run and in what order, but the graph owns
HOW each step executes, so an agent cannot invent an unexecutable step. Steps in
the plan that map to no known executor are recorded as skipped with a warning.

policies.py:
 - RetryPolicy: attempts and backoff, retrying only RetryableError subclasses
   (see core/errors.py) and never FatalError. LLM transient failures and
   agent output errors are retryable; a missing dependency is not.
 - BudgetPolicy: check state.time_remaining before each step; when the budget is
   nearly exhausted, skip optional steps (tuning, explainability, extra charts)
   rather than truncating the report, and record why.
 - ApprovalPolicy: when config.require_approval and a step is destructive,
   create an ApprovalRequest on state, emit APPROVAL_REQUESTED, and raise
   ApprovalRequired to suspend the run. Provide resume support so a caller can
   approve/reject and continue.
 - ReplanPolicy: decide from an EvaluationVerdict whether to replan, honouring
   config.max_replans and config.enable_self_improvement, and mapping
   recommended_action to the step the plan should restart from.

engine.py — class Orchestrator:
    def __init__(self, config: RunConfig, *, settings=None, repository=None, bus=None)
    def run(self) -> RunSummary
    def resume(self, request_id: str, approved: bool, note: str | None = None) -> RunSummary
    @property state -> RunState
Responsibilities:
 - Build RunState and EventBus; emit RUN_STARTED / STEP_* / RUN_COMPLETED etc.
 - Ingest, profile, then call state.freeze_context(build_run_context(profile, config))
   BEFORE any agent runs — every agent shares that cached prompt prefix, so it
   must be frozen exactly once and never rebuilt mid-run.
 - Consult dataset memory (storage.memory) before planning and attach the
   MemorySuggestion to state.memory when a repository is available.
 - Dispatch each step through the retry/budget/approval policies, updating
   StepRecord status and emitting events.
 - THE SELF-IMPROVEMENT LOOP: after evaluation, if ReplanPolicy says to retry,
   emit REPLAN_TRIGGERED, increment state.replans, push the current plan onto
   state.plan_history, re-run the PlannerAgent (which sees the failed evaluation
   and prior plan), and re-execute from the indicated step. Bound by
   config.max_replans, and the FINAL evaluation is the one that ships. If a
   replan produces a worse model, keep the better one and say so in a warning —
   silently shipping a regression would defeat the purpose of the loop.
 - Persist: save the run summary and events through the repository when given,
   always write state.save_summary(), and register a DatasetFingerprint for
   future memory lookups.
 - Never let a single non-critical step failure kill the run: ingest, profile,
   problem identification, planning, and training are critical; tuning,
   explainability, visualisation, and insights are not — degrade with a warning
   and continue to produce a report.
 - Support cancellation via a threading.Event so the API can stop a run.

runner.py:
    class AutoMLArchitect:
        def __init__(self, settings=None, repository=None)
        def analyse(self, source, *, target=None, **config_kwargs) -> RunSummary
        def ask(self, run_id: str, question: str) -> QuestionAnswer   # uses agents.qa.QAAgent
    def analyse(source, *, target=None, **kwargs) -> RunSummary
'source' must accept a filesystem path (str or Path — infer SourceKind from the
extension), a DataSource, or a pandas DataFrame (via SourceKind.DATAFRAME).
Keep 'analyze' as an alias. Ensure automl_architect/__init__.py's lazy imports
resolve against what you build.

Verify: ${PY} -c "from automl_architect.orchestrator.engine import Orchestrator; from automl_architect.runner import analyse; print('ok')"`,
  },
  {
    key: 'reporting',
    label: 'reporting',
    prompt: `Build chart rendering and multi-format report export.

Create:
  automl_architect/reporting/__init__.py
  automl_architect/reporting/charts.py     -> render_charts(state, plan) -> VisualizationBundle
  automl_architect/reporting/dashboard.py  -> a self-contained interactive HTML dashboard
  automl_architect/reporting/markdown.py   -> markdown renderer
  automl_architect/reporting/html.py       -> styled HTML renderer (jinja2)
  automl_architect/reporting/pdf.py        -> PDF via reportlab
  automl_architect/reporting/pptx.py       -> PowerPoint via python-pptx
  automl_architect/reporting/writer.py     -> write_report(state, report, formats) -> ReportBundle
  automl_architect/reporting/templates/report.html.j2

charts.py:
Implement every ChartKind in schemas.py with plotly, reading real data off
state (profile, splits, experiments, explainability, best_pipeline). Save each
as standalone HTML (include_plotlyjs='cdn') plus a PNG when kaleido is
importable, and record ChartArtifact with rendered/error set. Guard every chart
individually — a chart that cannot be built records its error and the rest still
render. Charts must be readable: axis titles, a real title, a colourblind-safe
qualitative palette applied consistently across all charts, and a light template.
ROC/PR/calibration need probabilities (skip with a note when the model has no
predict_proba); confusion matrices need labels in a stable order; feature
importance should show the top ~20 sorted; the leaderboard chart compares models
on the primary metric with the baseline visually distinguished.

dashboard.py:
Compose the rendered charts into ONE self-contained HTML file with a clean
layout: a header with the run's headline metrics, a section per chart group with
the agent's caption, and the dashboard_narrative. It must open correctly from
the filesystem with no server and no build step.

markdown.py / html.py / pdf.py / pptx.py:
Render FinalReport.ordered_sections() faithfully — the section markdown is
already authored, so do not rewrite it, only present it. Markdown: add a title
block, a run-metadata table, a table of contents, and an appendix listing every
recorded decision rationale plus the usage/cost totals. HTML: a jinja2 template
with embedded CSS, readable typography, a sticky table of contents, and the
charts linked or inlined; must render correctly offline. PDF (reportlab): title
page, headings, paragraphs, real tables for the leaderboard and decision log,
and embedded chart PNGs where they exist; convert the markdown to flowables
rather than dumping raw markup. PPTX: a title slide, an executive-summary slide,
one slide per major section with bullet points extracted from the markdown, chart
image slides, and a recommendations slide.

writer.py — write_report(state, report, formats):
Dispatch on the requested formats, write into state.artifact_path('report', ...),
always emit a JSON dump of the RunSummary, populate ReportBundle paths, and
record a warning for any format that failed or whose optional dependency is
missing (never raise — a missing PDF library must not lose the markdown report).

Self-test: build a minimal RunState with a synthetic profile, a couple of
ExperimentResults, and a small FinalReport, then render every format and print
the resulting paths. Confirm the HTML and dashboard open standalone.`,
  },
  {
    key: 'platform',
    label: 'storage+api+cli',
    prompt: `Build persistence, the HTTP API, and the CLI — the surfaces a user actually touches.

Create:
  automl_architect/storage/__init__.py
  automl_architect/storage/models.py      -> SQLAlchemy 2.0 ORM tables
  automl_architect/storage/repository.py  -> RunRepository per the contract
  automl_architect/storage/artifacts.py   -> artifact path helpers + save/load of fitted models
  automl_architect/storage/memory.py      -> DatasetMemory: fingerprint, similarity search, MemorySuggestion
  automl_architect/api/__init__.py
  automl_architect/api/app.py             -> FastAPI app factory
  automl_architect/api/routes.py          -> endpoints
  automl_architect/api/schemas.py         -> request/response models
  automl_architect/cli.py                 -> typer app (entry points 'automl-architect' and 'amla')

storage:
Tables for runs (id, project, status, task type, target, primary metric, best
score, timestamps, the full RunSummary JSON), events (run_id, sequence, kind,
agent, step, message, payload JSON, timestamp — indexed on (run_id, sequence)),
experiments (queryable leaderboard rows), approvals, and fingerprints. Default
to SQLite from settings.resolved_database_url; the code must work unchanged on
Postgres, so use JSON column types portably and do not rely on SQLite specifics.
memory.py implements the dataset-memory stretch goal: build a DatasetFingerprint
from a profile and problem, then score similarity against stored fingerprints
using a weighted blend of shape (log row/column counts), column-type mix,
missingness, task type, target kind, imbalance, and column-name overlap
(Jaccard). Return SimilarRun entries with a why_similar explanation, and
assemble a MemorySuggestion recommending the families and feature ops that
actually won on similar past runs, with cautions from what failed. It must
behave sanely with an empty history (has_precedent=False).

api:
Endpoints:
  POST   /api/runs                      start a run (accepts an uploaded file OR a DataSource); returns run_id immediately, executing in a background thread
  GET    /api/runs                      list runs, filterable by project
  GET    /api/runs/{run_id}             the full RunSummary
  GET    /api/runs/{run_id}/events      poll events, with an 'after' sequence cursor
  GET    /api/runs/{run_id}/stream      Server-Sent Events live stream (sse-starlette)
  POST   /api/runs/{run_id}/cancel      cancel
  GET    /api/runs/{run_id}/approvals   pending approval requests
  POST   /api/runs/{run_id}/approvals/{request_id}   approve or reject, resuming the run
  POST   /api/runs/{run_id}/ask         natural-language question -> QuestionAnswer
  GET    /api/runs/{run_id}/report      serve the rendered report (format query param)
  GET    /api/runs/{run_id}/artifacts   list artifacts
  GET    /api/runs/{run_id}/artifacts/{path}  serve one artifact
  GET    /api/health                    health + whether credentials are configured
  POST   /api/upload                    upload a dataset, returns a DataSource
The SSE stream must be lossless on reconnect: accept a 'last_sequence' cursor,
replay everything after it from the repository, then tail live events from the
bus — a client that reconnects mid-run must not silently miss the events that
happened during the gap. Bridge the synchronous EventBus to async with a queue
per subscriber; never block the event loop. CORS from settings.cors_origins.
Serve artifacts only from within the run directory — resolve the requested path
and reject anything that escapes it (path traversal).

cli.py (typer + rich):
  amla run <source> [--target] [--project] [--time-budget] [--max-experiments]
      [--metric] [--task] [--approve] [--no-tuning] [--format md,html,pdf,pptx] [--open]
      -> live rich progress driven by EventBus subscription: current step, a spinner,
         agent decisions as they land, and a running token/cost counter
  amla profile <source> [--target]   -> profile only, printed as rich tables
  amla list / amla show <run_id> / amla events <run_id> [--follow]
  amla ask <run_id> "<question>"
  amla serve [--host] [--port]       -> uvicorn
  amla report <run_id> [--format]
  amla doctor                        -> environment check: python version, credentials
                                        resolvable, which optional deps are installed,
                                        workspace writable, database reachable
'amla doctor' should be genuinely useful for diagnosing a broken install: report
each optional feature (boosted trees, tuning, SHAP, chart PNGs, PDF, PPTX, cloud
connectors) as available or missing WITH the pip command to fix it.

Verify: ${PY} -c "from automl_architect.api.app import create_app; create_app(); import automl_architect.cli; print('ok')"`,
  },
  {
    key: 'frontend',
    label: 'frontend',
    prompt: `Build the Next.js + TypeScript + Tailwind frontend dashboard at ${ROOT}/frontend.

It talks to the FastAPI backend at http://127.0.0.1:8000 (configurable via
NEXT_PUBLIC_API_URL). Use: Next.js (App Router), TypeScript strict, TailwindCSS,
and react-plotly.js OR plotly.js-dist-min for charts. Scaffold with npm, install
dependencies, and make sure 'npm run build' actually succeeds before you finish.
Run npm commands from ${ROOT}/frontend.

Pages:
  /                     Dashboard: run list with status badges, best score, task
                        type, duration; a "New analysis" panel with drag-and-drop
                        file upload, target column, and run options.
  /runs/[runId]         Live run view — the centrepiece.
  /runs/[runId]/report  The rendered report, with a format switcher.

The live run view must make the agent reasoning visible, because that
transparency is the product's actual differentiator over a black-box AutoML tool:
  - A pipeline stepper showing every plan step with its status (pending/running/
    completed/failed/skipped), the owning agent, and elapsed time.
  - A live event feed streaming from GET /api/runs/{id}/stream via EventSource,
    with agent thinking summaries and decisions rendered distinctly from logs.
    Auto-scroll with a pause-on-manual-scroll, and reconnect using the
    last_sequence cursor so nothing is lost on a dropped connection.
  - Panels that fill in as the run progresses: dataset profile summary, the
    execution plan with per-step rationale, cleaning decisions with their
    rationale, engineered features, the model leaderboard sorted by the primary
    metric with the baseline marked, feature importance, evaluation verdict, and
    business insights.
  - A token/cost meter.
  - An approval modal when the run suspends on a destructive step, posting the
    decision back and resuming.
  - A "Ask about this run" input posting to /api/runs/{id}/ask, rendering the
    answer with its evidence list.
  - Embedded charts: fetch the artifact list and render the plotly JSON, or
    iframe the standalone chart HTML.

Engineering requirements:
  - Generate TypeScript types mirroring the Pydantic models in
    automl_architect/core/schemas.py (read that file). Put them in
    frontend/src/types/api.ts and keep names aligned with the Python fields.
  - A typed API client in frontend/src/lib/api.ts with real error handling.
  - A useRunStream hook encapsulating the EventSource lifecycle, cursor
    tracking, and reconnect.
  - Handle every state honestly: loading, empty, error, and backend-unreachable
    (show a clear "start the API with 'amla serve'" hint rather than a blank page).
  - Dark mode, responsive down to tablet width, accessible (real labels, focus
    states, aria-live on the event feed, keyboard-navigable).
  - No placeholder or lorem text anywhere, and no fabricated numbers — every
    value shown must come from the API.

Also write frontend/README.md covering install, dev, build, and the env var.`,
  },
  {
    key: 'quality',
    label: 'tests+docs+examples',
    prompt: `Build the test suite, example data, and documentation. Other agents are writing
the modules concurrently, so write tests against the CONTRACT in this brief and
in automl_architect/core/schemas.py, not against implementation details you
cannot see. Prefer testing behaviour and invariants over internal call shapes.

Create:
  tests/conftest.py            shared fixtures
  tests/test_schemas.py        contract invariants
  tests/test_profiling.py      profiler correctness on frames with known properties
  tests/test_ingestion.py      round-trip every file connector
  tests/test_execution.py      cleaning ops, feature ops, splitter, metrics, model zoo, trainer
  tests/test_agents.py         agent prompt/postprocess logic with a FAKE LLM (no network)
  tests/test_orchestrator.py   end-to-end run with a fake LLM
  tests/test_api.py            FastAPI TestClient over the endpoints
  tests/test_reporting.py      chart + report rendering
  examples/generate_datasets.py  writes the example CSVs
  examples/churn.csv, examples/house_prices.csv, examples/sales_timeseries.csv
  examples/quickstart.py       library usage
  docs/ARCHITECTURE.md, docs/AGENTS.md, docs/API.md, docs/DEPLOYMENT.md
  README.md   (repo root — this is the project's front door)
  .env.example, .gitignore, Dockerfile, docker-compose.yml, Makefile

The FAKE LLM is the most important test asset. Build a FakeLLMClient in
conftest.py that satisfies core/llm.py's LLMClient surface (a .structured(...)
returning a StructuredResult whose .value is a caller-supplied, schema-valid
instance, a .text(...), a .usage accumulator) so the ENTIRE pipeline including
the orchestrator can be tested offline and deterministically. Register canned
responses per output model type. Tests must never hit the network — mark any
that would with @pytest.mark.live and skip without credentials.

Example datasets must be synthetic but REALISTIC, generated with a fixed seed,
and each must exercise specific paths:
 - churn.csv: binary imbalanced target (~26% positive), mixed types, missing
   values in a skewed income column, a high-cardinality id, a categorical with a
   rare level, and ONE deliberate leakage column so the leakage detector has
   something true to find.
 - house_prices.csv: regression with a right-skewed target, correlated features,
   and outliers.
 - sales_timeseries.csv: a date column, trend plus weekly seasonality, a group
   key with several series, and a couple of gaps.
Write generate_datasets.py so the CSVs are reproducible, and actually run it to
create them.

README.md must be the real front door: what the system does and the reasoning-
vs-computation split that makes it trustworthy, an architecture diagram (ASCII
is fine), install (including which optional extras unlock what), quickstart for
CLI + library + API + UI, a walked-through example of the output including a
sample of the explained decisions, the full agent roster with each agent's job,
configuration reference, and honest limitations. No marketing copy and no
claimed benchmark numbers.
docs/AGENTS.md documents each agent: inputs, output schema, effort level, and
the failure modes its prompt guards against.
Dockerfile: multi-stage, non-root user, the API as the entrypoint. compose:
api + postgres + the frontend.

Run the suite with ${PY} -m pytest and iterate until the tests you can run pass
or fail only because a sibling module is still being written. Report clearly
which tests pass, which are blocked on other modules, and any real defect you
found in another module's contract.`,
  },
]

const built = await pipeline(
  MODULES,
  (mod) => agent(`${SHARED}\n\n# YOUR MODULE: ${mod.label}\n\n${mod.prompt}`, {
    label: `build:${mod.key}`,
    phase: 'Build',
  }),
  (report, mod) => agent(
    `Repo: ${ROOT}. Python: ${PY}.\n\n` +
    `Another agent just built the '${mod.label}' module of AutoML Architect and reported:\n\n` +
    `${String(report).slice(0, 4000)}\n\n` +
    `Your job is a narrow correctness pass on THAT module's files only.\n\n` +
    `1. Import every module it created: ${PY} -c "import <module>". Fix every ` +
    `ImportError, SyntaxError, NameError, and bad-signature call you find.\n` +
    `2. Check its calls against the real contracts in automl_architect/core/schemas.py ` +
    `and core/*.py — wrong field names, wrong enum members, and constructing a ` +
    `Pydantic model without its required fields are the common failures. Verify by ` +
    `importing and instantiating, not by eye.\n` +
    `3. Check optional dependencies are imported lazily and raise MissingDependencyError, ` +
    `not at module scope (catboost, boto3, azure, gcs, kaggle, psycopg, PyMySQL, ` +
    `snowflake, mlflow are NOT installed and must not break an import).\n` +
    `4. Do NOT edit automl_architect/core/*, automl_architect/config.py, or another ` +
    `module's files. Do NOT redesign anything. Fix defects only.\n\n` +
    `Report: what you fixed, what still fails and why, and any contract mismatch that ` +
    `belongs to a different module (name the file and symbol so it can be routed).`,
    { label: `verify:${mod.key}`, phase: 'Verify', effort: 'high' }
  ),
)

return {
  modules: MODULES.map((m) => m.key),
  reports: built.map((r, i) => ({ module: MODULES[i]?.key, report: String(r ?? 'FAILED').slice(0, 2500) })),
}
