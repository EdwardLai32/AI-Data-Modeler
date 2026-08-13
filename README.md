# AutoML Architect

An autonomous multi-agent data scientist. Point it at a table and it profiles the
data, decides what problem the table poses, plans a pipeline for *that* dataset,
cleans and engineers features, trains and compares models, tunes the leader,
explains what the model learned, judges whether the result is fit to deploy, and
writes the report — recording a reason for every choice it made along the way.

It is built on one structural rule, and that rule is the product:

> **Claude reasons. Python computes.**
>
> No agent ever calculates a statistic or trains a model. No executor ever makes
> an unexplained choice.

Thirteen specialist agents receive *measured facts* — skewness computed by pandas,
a class ratio counted from the actual column, an AUC from scikit-learn — and
return typed decisions. Every decision schema has a mandatory `rationale` field,
so an agent physically cannot emit a transformation it did not argue for. Executor
modules then apply those decisions with real library code.

The consequence is worth being concrete about. When a report says *"imputed
`annual_income` with the median"*, you can read the skewness value that drove the
choice, re-derive it from the CSV yourself, and find the sentence the agent wrote
explaining why the mean would have been worse. Every number is reproducible and
every choice is auditable. That is the whole point.

---

## Contents

- [How it works](#how-it-works)
- [Install](#install)
- [Quickstart](#quickstart)
- [Offline mode](#offline-mode)
- [A worked example](#a-worked-example)
- [The agent roster](#the-agent-roster)
- [Configuration](#configuration)
- [Example datasets](#example-datasets)
- [Development](#development)
- [Limitations](#limitations)

---

## How it works

```
                              ┌─────────────────────────────┐
   data source ──────────────►│  ingestion/  →  profiling/  │
   csv · parquet · excel      │  load & validate, then      │
   json · sql · duckdb        │  MEASURE everything         │
   s3 · gcs · azure · kaggle  └──────────────┬──────────────┘
                                             │  DatasetProfile
                                             │  (deterministic, no LLM involved)
                                             ▼
                              ┌─────────────────────────────┐
                              │  core/context.py            │
                              │  render facts as prompt text│
                              │  ─ frozen once per run ─    │
                              └──────────────┬──────────────┘
                                             │  cached prompt prefix
              ┌──────────────────────────────┴──────────────────────────────┐
              │                    orchestrator/engine.py                   │
              │        walks the plan · retries · replans · approvals        │
              └───┬──────────────────────────────────────────────────┬──────┘
                  │                                                  │
      ┌───────────▼───────────┐                        ┌─────────────▼──────────────┐
      │   agents/  (REASON)   │                        │  execution/  (COMPUTE)     │
      │                       │  typed decision        │                            │
      │  dataset              │ ─────────────────────► │  cleaning_ops   pandas     │
      │  problem              │                        │  feature_ops    sklearn    │
      │  planner              │                        │  splitter       stratified │
      │  cleaning             │                        │  model_zoo      xgb / lgbm │
      │  features             │  ◄───────────────────  │  trainer        cross-val  │
      │  model_selection      │  measured results      │  tuner          optuna     │
      │  experiment           │                        │  explainer      shap       │
      │  tuning               │                        │  diagnostics    calibration│
      │  explain              │                        │  metrics                   │
      │  evaluation ──────────┼── not acceptable ──┐   └────────────────────────────┘
      │  insight              │                    │
      │  visualization        │                    │  replan (bounded)
      │  report               │◄───────────────────┘
      └───────────┬───────────┘
                  │
   ┌──────────────▼───────────────┐    ┌──────────────────────────────────┐
   │  reporting/                  │    │  storage/                        │
   │  charts · md · html · pdf    │    │  runs · events · fingerprints    │
   │  pptx · dashboard            │    │  dataset memory ("seen this      │
   └──────────────┬───────────────┘    │  shape before")                  │
                  │                    └──────────────────────────────────┘
   ┌──────────────▼───────────────────────────────────────────────────────┐
   │  surfaces:   cli.py (typer)  ·  api/ (FastAPI + SSE)  ·  frontend/   │
   └──────────────────────────────────────────────────────────────────────┘
```

Three design decisions carry most of the weight.

**Facts are measured once, then frozen.** `core/context.py` renders the profile
into prompt text exactly once per run, and every agent reuses that identical
string as its cached prompt prefix. It keeps costs down, and more importantly it
guarantees all thirteen agents reason over the same facts rather than thirteen
slightly different summaries.

**Free-form config is `list[Param]`, never `dict[str, Any]`.** Claude's
structured-output schema subset cannot express an object with undeclared
properties, so open-ended parameters travel as typed key/value pairs and
`params_to_dict()` converts them back to real Python scalars at the executor
boundary.

**Evaluation is a real gate.** The Evaluation Agent can return
`acceptable=False` with a `recommended_action`, which unwinds to the Planner for a
bounded number of revisions. A run that ends with a mediocre model ends with a
mediocre model *and an honest verdict saying so*, rather than a confident summary.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full walkthrough.

---

## Install

Requires **Python 3.11+**.

```bash
git clone <this-repo> && cd automl-architect
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

Then set a credential. The Anthropic SDK reads `ANTHROPIC_API_KEY`, or
`ANTHROPIC_AUTH_TOKEN`, or an OAuth profile written by `ant auth login`:

```bash
cp .env.example .env      # then edit it
```

Verify the install, including which optional features resolved:

```bash
amla doctor
```

### Optional extras, and what each unlocks

The base install runs the full pipeline with scikit-learn models, Plotly charts,
Markdown/HTML/JSON reports, and SQLite storage. Everything below is additive —
when a package is absent the feature degrades with a logged warning and the run
continues.

| Extra | Install | Unlocks | Without it |
|---|---|---|---|
| `boost` | `pip install -e ".[boost]"` | XGBoost, LightGBM, CatBoost model families | Those families are excluded from selection; sklearn's `HistGradientBoosting` remains |
| `tuning` | `.[tuning]` | Optuna TPE and pruning-based search | Grid and random search only |
| `explain` | `.[explain]` | SHAP attributions and summary plots | Permutation importance and model-native importances only |
| `charts` | `.[charts]` | Static PNG export via kaleido | Interactive HTML charts only; PDF/PPTX lose embedded images |
| `reports` | `.[reports]` | PDF (reportlab) and PPTX (python-pptx) output | Markdown, HTML, JSON |
| `api` | `.[api]` | FastAPI server, SSE event stream | CLI and library only |
| `sql` | `.[sql]` | PostgreSQL and MySQL sources | SQLite and DuckDB still work through the generic SQL connector |
| `cloud` | `.[cloud]` | S3, GCS, Azure Blob sources | Local files and databases |
| `kaggle` | `.[kaggle]` | `kaggle:owner/dataset` sources | — |
| `snowflake`, `databricks` | `.[snowflake]` | Those warehouses | — |
| `all` | `.[all]` | Everything above | — |

---

## Quickstart

### CLI

```bash
# Profile only. No model calls, no cost — read the facts before spending anything.
amla profile examples/churn.csv --target churned

# A full run with a live display.
amla run examples/churn.csv --target churned --format md,html

# Bound the compute, force the metric, audit fairness by region.
amla run examples/churn.csv -t churned \
    --time-budget 600 --max-experiments 6 --metric roc_auc --fairness region

# Pause before anything destructive and confirm interactively.
amla run examples/churn.csv -t churned --approve

# No API key, no model calls, no cost. See "Offline mode" below.
amla run examples/churn.csv -t churned --offline

# After the run.
amla list
amla show <run_id>
amla events <run_id> --follow
amla ask <run_id> "Why was gradient boosting chosen over logistic regression?"
amla report <run_id> --open
```

### Library

```python
from automl_architect import analyse

summary = analyse("examples/churn.csv", target="churned", max_experiments=6)

print(summary.problem.task_type.value)        # binary_classification
print(summary.problem.metric_rationale)       # why ROC AUC, not accuracy
print(summary.experiments.best().primary_score)
print(summary.evaluation.overall_grade)       # A-F, with a written verdict
print(summary.report_bundle.markdown_path)

for decision in summary.cleaning.decisions:
    print(f"{decision.action.value}: {decision.rationale}")
```

For repeated use, hold the facade so runs accumulate into dataset memory:

```python
from automl_architect import AutoMLArchitect

architect = AutoMLArchitect()
summary = architect.analyse("data/q3.csv", target="renewed")
print(architect.ask(summary.run_id, "Which feature mattered most, and by how much?").answer)
```

A DataFrame works as a source directly — no temporary file needed:

```python
import pandas as pd
summary = analyse(pd.read_csv("data/q3.csv"), target="renewed")
```

### API

```bash
amla serve --host 127.0.0.1 --port 8000     # or: uvicorn automl_architect.api.app:create_app --factory
```

```bash
# Start a run; returns immediately with a run id and follow URLs.
curl -X POST localhost:8000/api/runs \
  -H 'content-type: application/json' \
  -d '{"uri": "examples/churn.csv", "target_column": "churned", "max_experiments": 4}'

# Or upload the file in the same request.
curl -X POST localhost:8000/api/runs -F file=@data/churn.csv -F target_column=churned

# Follow it live over server-sent events.
curl -N localhost:8000/api/runs/<run_id>/stream

curl localhost:8000/api/runs/<run_id>            # full RunSummary
curl localhost:8000/api/runs/<run_id>/report     # rendered report
```

Interactive docs at `http://localhost:8000/docs`. Full endpoint reference in
[`docs/API.md`](docs/API.md).

### UI

```bash
cd frontend
cp .env.local.example .env.local     # point NEXT_PUBLIC_API_URL at the API
npm install && npm run dev           # http://localhost:3000
```

The browser UI streams the run as it happens: the plan as the Planner writes it,
each agent's decision and reasoning as it lands, the leaderboard filling in, and
the charts and report at the end. Make sure the UI's origin appears in the API's
`AUTOML_CORS_ORIGINS`.

Or run both with Docker:

```bash
cp .env.example .env         # set ANTHROPIC_API_KEY
docker compose up --build    # api :8000, frontend :3000, postgres :5432
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) before exposing this to anything
you do not control — the API ships without authentication on purpose, and `uri`
reads from the server's filesystem.

---

## Offline mode

```bash
amla run examples/churn.csv --target churned --offline    # or AUTOML_OFFLINE=1
```

```python
from automl_architect import AutoMLArchitect
from automl_architect.config import Settings

summary = AutoMLArchitect(settings=Settings(offline=True)).analyse(
    "examples/churn.csv", target="churned"
)
assert summary.usage.cost_usd == 0.0
```

The API server honours the same switch through `AUTOML_OFFLINE=1` in its
environment.

Runs the whole pipeline with **zero model calls and no credentials**. Not a mock
and not a dry run: it ingests, profiles, cleans, engineers features, trains,
tunes, computes SHAP, renders charts, and writes all four report formats. A real
run on `examples/churn.csv` finishes in about 40 seconds for `$0.00`.

**What changes.** Only the reasoning. Each agent, instead of asking Claude for a
typed decision, asks a deterministic rule engine
([`core/offline.py`](automl_architect/core/offline.py)) for the same type. The
rules read the measured profile and branch on it:

| Measurement | Rule | Rationale it writes |
|---|---|---|
| `annual_income` skewness 2.43 | median impute | *"Highly skewed distribution: skewness 2.425 (threshold 1.0), with mean 62,854 against median 50,560. The mean is pulled toward the tail…"* |
| `age` skewness 0.12 | mean impute | *"Distribution is approximately normal: skewness 0.12 is within ±1.0…"* |
| `city`, 14 levels | mode impute | *"Categorical feature with 14 levels; the mean and median are undefined for unordered categories."* |
| `cancellation_tickets` associates at 0.998 | drop as leakage | *"A genuine predictor rarely reaches this strength; keeping it would produce a score that collapses in production."* |
| 3,000 distinct values over 3,000 rows | drop as identifier | *"A tree can memorise the target through it and learn nothing that generalises."* |

Everything downstream is untouched, because it was always deterministic Python.
The models are real, the scores are honest, and the leakage column really does
get caught.

**What you lose.** Rules read statistics, not meaning. They cannot infer that
`last_login_days` is a churn precursor, notice that two columns encode the same
quantity in different units, or write the business narrative the Insight Agent
exists to write — offline insights restate measured drivers rather than
explaining them. Domain inference falls back to keyword-matching column names,
and `amla ask` returns a run summary instead of answering the question.

Offline output is **correct and defensible; online output is insightful.** Every
rule-derived string is tagged `[rule]` so the two are never confused in a report,
and `amla doctor` reports which mode you are in.

Useful for CI (the suite runs with no key), demos, cost control, air-gapped
environments, and as a fallback when the API is unreachable.

---

## A worked example

`examples/churn.csv` — 3,000 subscribers, 26% churn. Here is the shape of what a
run produces. The numbers below come from a real run on this file; yours will
differ slightly with model versions and seeds.

**1. Measured facts** (deterministic, before any model call)

```
rows: 3,000   columns: 16   duplicate rows: 0   missing cells: 259 (0.54%)

target `churned`: 2 classes — 0: 2,221 (74.03%), 1: 779 (25.97%)
                  imbalance ratio 2.851

`annual_income`  [numeric_continuous]  missing=259 (8.63%)  skew=2.43
                 min=5,498 p25=33,740 median=50,560 p75=78,470 max=489,303
                 mean=62,854  -> mean sits at the 63rd percentile

target-leakage candidates
  `cancellation_tickets`: score=0.9975 severity=critical method=auc_single_feature
     — separates the target almost perfectly; recorded after the churn decision
```

**2. Decisions, each with the evidence that drove it**

> **Problem Agent** → `binary_classification`, metric `roc_auc`
> *"`churned` takes exactly two integer values with 26% positive, so this is
> binary classification on an imbalanced target. At 26% positive, accuracy is
> beaten by predicting the majority class; ROC AUC is threshold-free and average
> precision tracks the minority class."*

> **Cleaning Agent** → `drop_leakage_column: [cancellation_tickets]`
> *"AUC of 0.9975 against the target versus 0.627 for the next-best feature, and
> the column records an event that happens after churn."*
> `severity_if_skipped: critical` · `destructive: true`

> **Cleaning Agent** → `impute_missing: [annual_income], strategy=median`
> *"8.6% missing on a log-normal column: the mean sits well above the median, so
> mean imputation would invent atypically wealthy subscribers."*

> **Cleaning Agent** → *deliberately skipped:* outlier clipping on `total_charges`
> *"The tail is real billing history, not error."*

> **Feature Agent** → `frequency_encode: [payment_method]`
> *"`crypto_wallet` holds 0.5% of rows; one-hot would produce a near-constant
> column, so frequency encoding preserves the level."*

> **Model Selection Agent** → 4 candidates, `hist_gradient_boosting` ranked 1
> *"The comparison that matters is boosting against logistic regression: if
> boosting wins by less than a couple of AUC points, the interpretable model is
> the better product for a retention team that must justify each offer."*

**3. Leaderboard**

```
family                    roc_auc   avg_prec   accuracy   train_s
hist_gradient_boosting     0.712      0.451      0.766       1.9   <- best
random_forest              0.704      0.441      0.761       2.3
logistic                   0.689      0.418      0.758       0.2
baseline_dummy             0.500      0.260      0.740       0.0   <- floor
```

Note the baseline: 74% accuracy with zero predictive power. That row is why the
metric argument above matters, and it is trained on every run for exactly this
reason.

**4. Verdict** — the gate, not a formality

```
acceptable: true   grade: B   recommended_action: accept
bias/variance: train 0.742 · validation 0.718 · test 0.712 · gap 0.030 -> good_fit
calibration:   brier 0.164 · ECE 0.041 -> "usable as a ranking; apply isotonic
               calibration before quoting probabilities"
drift risk:    medium — "contract mix shifts with pricing changes, so the
               dominant feature is not stationary"
weaknesses:    probabilities are uncalibrated; no temporal holdout, so drift
               is untested
```

**5. Business translation**

> **Month-to-month subscribers are the entire churn problem.** They are 57% of the
> base and the largest positive driver of predicted risk, second only to short
> tenure. *Evidence: `contract_type_month_to_month` holds 24% of SHAP
> attribution.* → Offer a one-year term at a discount to the top risk decile.
> *~190 accounts; a 15% save rate retains ~28 per quarter.* Confidence: medium.
>
> **Caveat:** AUC 0.71 ranks well but does not predict individuals reliably, and
> cross-sectional data cannot establish that contracts *cause* retention.

Everything above lands in `workspace/runs/<run_id>/`: the report in each requested
format, the charts, the fitted model, and `run_summary.json` — the complete typed
record of the run, including every decision, its rationale, the event log, and
token/cost accounting.

---

## The agent roster

Thirteen agents, each a narrow contract: a system prompt, measured facts in, a
typed decision out. Effort is the reasoning-depth dial; planning and evaluation
get more of it than narration does.

| Agent | Job | Returns | Effort |
|---|---|---|---|
| **Dataset Understanding** | Read the profile like a senior analyst would: what this data is, its grain, what to watch out for | `DatasetUnderstanding` | high |
| **Problem Identification** | Decide the task type, target, and primary metric — and argue the metric against the class balance | `ProblemDefinition` | xhigh |
| **Planning** | Write an execution plan fitted to *this* dataset, naming its actual columns and risks | `ExecutionPlan` | max |
| **Data Cleaning** | Choose imputation, drops, and outlier handling from the measured distributions | `CleaningPlan` | high |
| **Feature Engineering** | Propose features with a stated mechanism and a leakage risk assessment for each | `FeaturePlan` | high |
| **Model Selection** | Rank candidate families for this data's size and shape, and say what it excluded | `ModelSelection` | xhigh |
| **Experiment** | Interpret real training results: what the leaderboard actually shows | `ExperimentLog` | medium |
| **Hyperparameter Optimization** | Decide whether tuning is worth the compute, then design the search space | `TuningDecision` | high |
| **Explainability** | Narrate computed SHAP and permutation attributions in business language | `ExplainabilityReport` | high |
| **Evaluation** | The quality gate: bias/variance, calibration, fairness, drift — accept or send back | `EvaluationVerdict` | max |
| **Business Insight** | Turn attributions into actions with expected value and honest caveats | `InsightReport` | high |
| **Visualization** | Choose the charts that carry the argument, and say what question each answers | `VisualizationPlan` | medium |
| **Report** | Assemble the deliverable and recommend a deployment pattern from measured latency | `FinalReport` | xhigh |

A fourteenth, the **Q&A Agent**, answers natural-language questions about a
finished run, grounded in its recorded events and metrics rather than in
recollection.

Per-agent inputs, output schemas, and the failure modes each prompt guards
against: [`docs/AGENTS.md`](docs/AGENTS.md).

---

## Configuration

Two layers. **`Settings`** is deployment wiring, read from the environment with an
`AUTOML_` prefix and from `.env`. **`RunConfig`** is per-run behaviour, passed to
`analyse()` or the CLI.

### Settings (environment)

| Variable | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Credential. `ANTHROPIC_AUTH_TOKEN` and OAuth profiles also work |
| `AUTOML_OFFLINE` | `false` | Zero model calls, no credential needed. See [Offline mode](#offline-mode) |
| `AUTOML_MODEL` | `claude-opus-5` | Model id. Prompts here are tuned against Opus 5's effort ladder |
| `AUTOML_DEFAULT_EFFORT` | `high` | Reasoning depth when an agent does not override: `low`…`max` |
| `AUTOML_MAX_OUTPUT_TOKENS` | `16000` | Caps thinking *plus* visible output per call |
| `AUTOML_ENABLE_THINKING` | `true` | Adaptive thinking. Disabling it clamps effort to `high` |
| `AUTOML_ENABLE_PROMPT_CACHING` | `true` | Cache breakpoints on the stable prompt prefix |
| `AUTOML_ENABLE_REFUSAL_FALLBACK` | `true` | Re-serve a policy decline via the fallback model instead of failing the run |
| `AUTOML_WORKSPACE` | `./workspace` | Artifacts, models, charts, reports, SQLite database |
| `AUTOML_DATABASE_URL` | SQLite in workspace | Any SQLAlchemy URL; Postgres for shared deployments |
| `AUTOML_MAX_PROFILE_ROWS` | `250000` | Row cap for expensive profiling passes. The true row count is always reported |
| `AUTOML_MAX_TRAIN_ROWS` | `500000` | Row cap before training subsamples |
| `AUTOML_N_JOBS` | `-1` | Parallelism for sklearn |
| `AUTOML_API_HOST` / `AUTOML_API_PORT` | `127.0.0.1` / `8000` | API bind address |
| `AUTOML_CORS_ORIGINS` | `["http://localhost:3000"]` | Origins allowed to call the API |
| `AUTOML_LOG_LEVEL` | `INFO` | Standard logging levels |

### RunConfig (per run)

| Field | Default | What it does |
|---|---|---|
| `target_column` | `None` | Omit to let the agents propose one |
| `task_type_override` | `None` | Force the task type; agents must respect it |
| `primary_metric_override` | `None` | Force the metric, overriding the agent's argument |
| `time_budget_seconds` | `900` | Wall-clock budget for the whole run |
| `max_experiments` | `8` | Cap on models trained |
| `max_rows` | `None` | Sample cap for very large tables |
| `test_size` / `validation_size` | `0.2` / `0.15` | Holdout fractions |
| `cv_folds` | `5` | Cross-validation folds |
| `random_state` | `42` | Seed threaded through every stochastic step |
| `require_approval` | `False` | Pause before destructive steps and wait for a human |
| `enable_tuning` | `True` | Allow hyperparameter search |
| `enable_explainability` | `True` | Allow SHAP and permutation importance |
| `enable_self_improvement` | `True` | Allow the evaluation → replan loop |
| `max_replans` | `2` | Bound on that loop |
| `min_acceptable_score` | `None` | Floor the Evaluation Agent must respect |
| `report_formats` | `["markdown","html","json"]` | `markdown`, `html`, `pdf`, `pptx`, `json` |
| `fairness_attributes` | `[]` | Columns to slice metrics by |
| `notes` | `""` | Free text passed to the agents as operator context |

---

## Example datasets

Generated with a fixed seed, each exercising a specific code path:

```bash
python examples/generate_datasets.py
```

| File | Shape | Exercises |
|---|---|---|
| `churn.csv` | 3,000 × 16 | Imbalanced binary target (25.97% positive), mixed dtypes, log-normal income with 8.6% missing, a high-cardinality id, a 0.5% rare category level, and **one deliberate leakage column** (`cancellation_tickets`, AUC 0.9975 vs 0.627 for the best legitimate feature) |
| `house_prices.csv` | 1,600 × 14 | Regression on a right-skewed target (skew 6.9), a collinear pair at r=0.945 (`sqft_living`/`sqft_above`), 14 luxury outliers, 5.5% missing lot sizes |
| `sales_timeseries.csv` | 2,914 × 9 | Four daily series under one group key, linear trend, weekly and annual seasonality, promo effects, and two calendar gaps (7 days and 3 days) |

The leakage column in `churn.csv` exists so the leakage detector has something
*true* to find. A detector that only ever avoids false positives has not been
tested.

`examples/quickstart.py` walks the library API end to end.

---

## Development

```bash
pip install -e ".[dev,all]"
pytest                       # the whole suite, fully offline
pytest -m "not slow"         # skip the end-to-end runs
ruff check . && mypy automl_architect
make help                    # every task
```

The suite never touches the network. `tests/conftest.py` provides a
`FakeLLMClient` that implements the real client's surface and resolves each
`structured()` call by output-model type, so the entire pipeline — orchestrator,
replanning, approvals, API — is tested offline and deterministically. Tests that
would hit the real API are marked `@pytest.mark.live` and skip without
credentials. The end-to-end files are marked `slow`; `make test-fast` skips them.

### Documentation

| Document | Covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The reasoning/computation split, prompt caching, leakage discipline, the replan loop, and how to extend each layer |
| [`docs/AGENTS.md`](docs/AGENTS.md) | Each agent's inputs, output schema, effort level, and the failure modes its prompt and postprocessor guard against |
| [`docs/API.md`](docs/API.md) | Every HTTP endpoint, the SSE event stream, and the error envelope |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Docker, Compose, Postgres, reverse proxies, scaling, sizing, and cost control |

---

## Limitations

Stated plainly, because a tool that reasons about data should not be vague about
its own boundaries.

**Scope.** Tabular supervised learning is what this does well: binary and
multiclass classification, regression, and basic time-series forecasting.
Clustering and anomaly detection work but get less attention from the agents. NLP,
computer vision, graph learning, survival analysis, causal inference, ranking, and
recommendation are present in the task taxonomy and **not implemented** end to
end — `TaskType.is_supported` tells you which is which, and the orchestrator will
say so rather than pretend.

**Scale.** Everything runs in-process on one machine, in pandas. Comfortable to
roughly a few million rows depending on width and RAM; beyond that, sample first
or push the aggregation into your warehouse and point this at the result. There is
no distributed execution and no GPU path.

**Cost and latency.** A run makes 13+ calls to a frontier model at high reasoning
effort. Prompt caching makes the shared prefix cheap after the first call, and
`amla profile` costs nothing at all, but a full run is neither free nor instant.
Every run records its exact token and dollar cost in `summary.usage` — check it
before running a hundred.

**Determinism.** The computational half is reproducible: fixed seeds, recorded
`random_state`, identical profiles across runs. The reasoning half is not. The
same dataset can yield a different plan on a different day. The *decisions* are
always recorded with their evidence, so a plan you disagree with is legible — but
do not expect byte-identical runs.

**Judgement.** The agents reason well about distributions, and cannot know your
business. Whether a column is legitimately available at prediction time, whether a
proxy is ethically acceptable, whether 0.71 AUC is worth deploying — these need a
human who knows the domain. The leakage detector finds statistical anomalies; you
decide what they mean. The fairness audit slices metrics by the attributes you
name; it does not know which attributes matter.

**Not a production ML platform.** There is no feature store, no model registry
with promotion gates, no drift monitoring in the loop, no scheduled retraining.
The Report Agent *recommends* a deployment pattern with a monitoring plan; it does
not deploy or monitor anything. Treat the output as a rigorous first pass by a
capable analyst who documents everything — not as a shipped system.

**No benchmark claims.** This repository makes no accuracy claims against AutoML
baselines, and the numbers in the worked example above are from one run on one
synthetic file, included to show the *shape* of the output. Run it on your data
and read the verdict.

---

## License

MIT. See `pyproject.toml`.
