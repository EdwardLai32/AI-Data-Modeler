# The agents

Fourteen reasoning specialists. Each is a narrow contract: a system prompt
defining its role and method, measured facts as its input, and a Pydantic model as
its output. None of them computes a statistic or trains a model.

Every entry below documents four things:

- **Inputs** — what state the agent reads, and what facts its prompt renders.
- **Output** — the schema it must return.
- **Effort / budget** — reasoning depth and the token cap covering thinking *plus*
  visible output.
- **Guards** — the failure modes its prompt argues against and its `postprocess`
  enforces in code. This is the section worth reading. A prompt instruction is a
  request; a postprocess check is a guarantee, and the split between them is
  deliberate everywhere it appears.

Jump to an agent: [Dataset Understanding](#1-dataset-understanding-agent) ·
[Problem Identification](#2-problem-identification-agent) ·
[Planning](#3-planning-agent) · [Data Cleaning](#4-data-cleaning-agent) ·
[Feature Engineering](#5-feature-engineering-agent) ·
[Model Selection](#6-model-selection-agent) · [Experiment](#7-experiment-agent) ·
[Hyperparameter Optimization](#8-hyperparameter-optimization-agent) ·
[Explainability](#9-explainability-agent) · [Evaluation](#10-evaluation-agent) ·
[Business Insight](#11-business-insight-agent) ·
[Visualization](#12-visualization-agent) · [Report](#13-report-agent) ·
[Q&A](#14-qa-agent) · [Effort ladder](#effort-ladder-at-a-glance)

Common to all of them, inherited from `BaseAgent`:

- The system prompt is three blocks — platform preamble, frozen run-context
  digest, agent instructions — with cache breakpoints on the first two.
- Usage and cost accumulate onto `state.usage`; `LLM_CALL`, `AGENT_THINKING`, and
  `AGENT_DECISION` events are emitted around every call.
- `keep_known_columns()` filters column references against the real schema and
  records a warning for anything it drops.
- An unexpected exception becomes an `AgentError` carrying the agent's name.

---

## 1. Dataset Understanding Agent

`agents/dataset.py` · `AgentName.DATASET`

**Inputs.** `state.profile` via the cached context digest. The prompt adds nothing
numeric — the statistics are already in the shared prefix, and repeating them
costs tokens for no information.

**Output.** `DatasetUnderstanding` — a headline, a multi-paragraph narrative, the
likely domain, the grain, per-column assessments with roles and predictive
potential, key findings, risks, ranked target candidates, and a readiness verdict
(`ready` / `needs_cleaning` / `needs_major_work` / `unusable`) with its rationale.

**Effort.** `high`, 16k tokens.

**Method the prompt imposes.** Establish the grain *first* — one customer, one
customer-month, one transaction? — read off the evidence: which column combination
is unique, whether an entity id repeats, whether a timestamp advances per row. The
grain determines whether aggregation, lag features, and grouped splitting are even
meaningful. An ambiguous grain is a finding, not a failure.

**Guards.**

- *Restating the profile instead of interpreting it.* The prompt is explicit that
  the numbers are already given and the deliverable is what they mean.
- *Hallucinated targets.* `suggested_target_columns` is filtered against the real
  schema. Left unfiltered it becomes a `KeyError` three steps later inside pandas,
  at a point with no visible connection to its cause.
- *Token blowout on wide tables.* Above 80 columns, one assessment per column costs
  more than the marginal insight is worth, so coverage becomes prioritised rather
  than exhaustive.

---

## 2. Problem Identification Agent

`agents/problem.py` · `AgentName.PROBLEM`

**Inputs.** The profile, the target summary (class counts, imbalance ratio, or
mean/std/skew), temporal-column candidates, and any operator overrides on
`RunConfig`.

**Output.** `ProblemDefinition` — task type, target, positive class, temporal and
group columns, forecast horizon, primary and secondary metrics, two separate
rationales (one for the task, one for the metric), alternatives considered,
confidence, business objective, and constraints.

**Effort.** `xhigh`, 16k tokens. Everything downstream is conditioned on the two
decisions made here, which is what buys the extra depth.

**Guards.**

- *A metric that reports success while being useless.* Accuracy on a 95/5 target
  is the canonical case. The prompt requires the metric choice to be argued
  against the measured class balance, and `metric_rationale` is a separate
  required field so the argument cannot be folded into the task rationale and
  skipped.
- *A metric from the wrong family.* `rmse` on a classifier is a crash, not a
  debate — postprocess replaces it rather than passing it to the scorer.
- *Overriding the operator.* A human who forced the task type or the metric is not
  asking for a second opinion. Overrides win absolutely.
- *Silently downgrading an honest diagnosis.* When the diagnosed task is real but
  unsupported end to end (survival analysis, say), the diagnosis is kept as
  diagnosed and the execution fallback is recorded on `state.extras`, so the
  report can say what happened instead of quietly training the wrong thing.

---

## 3. Planning Agent

`agents/planner.py` · `AgentName.PLANNER`

**Inputs.** The profile, the understanding, the problem definition, run
constraints, `MemorySuggestion` from dataset memory, and on a replan: the previous
plan, the measured scores, the evaluation's stated weaknesses, and everything
already tried.

**Output.** `ExecutionPlan` — a strategy summary, ordered steps with dependencies
and per-step rationales, `dataset_specific_adaptations`, risks, a fallback
strategy, and a revision counter.

**Effort.** `max`, 32k tokens. The most expensive call in the run, and the one that
shapes every other.

**Why the plan comes first.** The reason each step exists is recorded *before* its
outcome is known, so nobody can retrofit a justification after seeing the score.
That ordering is what makes the run auditable rather than merely documented.

**Guards.**

- *The generic pipeline.* A plan that would suit any dataset is a plan nobody
  thought about. The prompt is fed deterministic **triggers** computed from the
  profile — nothing missing means no imputation step, leakage found means a removal
  step, a monotonic timestamp means a time-ordered split — so "adapt to this
  dataset" is checkable rather than aspirational.
- *A replan that reshuffles the same plan.* The replan prompt carries an explicit
  definition of what "materially different" means, plus the full record of what was
  already tried.
- *Structural damage.* `postprocess` repairs rather than trusts: slugs are
  normalised, dangling and forward dependencies dropped, the pre-plan agents
  removed so the plan cannot recurse into itself, and the mandatory terminal steps
  (evaluation, insight, report) appended if absent.

---

## 4. Data Cleaning Agent

`agents/cleaning.py` · `AgentName.CLEANING`

**Inputs.** Per-column missingness, skewness, outlier summaries, cardinality,
duplicate counts, leakage findings, and the target definition.

**Output.** `CleaningPlan` — decisions with an action, columns, a strategy, a
rationale, an expected impact, a `destructive` flag, and `severity_if_skipped`;
plus `columns_to_drop`, `drop_rationale`, and `skipped_considerations`.

**Effort.** `high`, 16k tokens.

**What the prompt actually teaches.** The schema already forces a `rationale`
string; the prompt is what makes it worth reading. It lays out the decision
procedure a senior practitioner follows — which statistic decides mean vs median,
when missingness is itself the signal, when an outlier is an error rather than the
tail you are being paid to model — and shows a good rationale against a bad one so
the contrast is concrete rather than exhorted:

> Bad: *"Imputed with the median as this is best practice for numeric columns."*
> Good: *"8.6% missing on a log-normal column (skew 2.09): the mean sits at the
> 62nd percentile, so mean imputation would invent atypically wealthy subscribers."*

`skipped_considerations` is required by design. A plan that records only what it
did hides half its reasoning, and "we considered clipping the billing tail and
decided it was real history" is exactly the sentence a reviewer needs.

**Guards.** Three failures must never reach pandas:

- *A column that does not exist* — filtered.
- *An imputation aimed at the target* — that fabricates ground truth. Removed.
- *A "drop" not marked `destructive`* — the orchestrator gates human approval on
  that flag, so an unmarked drop is a silent bypass of the gate. Postprocess sets
  it for every action in `_DESTRUCTIVE_ACTIONS`.

---

## 5. Feature Engineering Agent

`agents/features.py` · `AgentName.FEATURES`

**Inputs.** Column kinds and cardinalities, target correlations, collinear pairs,
temporal and group columns, the task type, and what cleaning already did.

**Output.** `FeaturePlan` — decisions each carrying an op, input columns, an output
name hint, parameters, a rationale, a **hypothesis**, a **risk**, and a priority;
plus a dimensionality strategy and a selection strategy.

**Effort.** `high`, 16k tokens.

**Why three text fields per feature.** A feature without a stated mechanism is a
lottery ticket. `hypothesis` is why this quantity should relate to the target
("weekend orders behave differently from weekdays"); `risk` is how it could leak or
overfit and how that is mitigated. The prompt is written to make both carry real
content rather than restating the operation's name.

**Guards.**

- *Stale column references.* Cleaning may have dropped a column the agent is still
  reasoning about. Filtered.
- *Time-travel features on unordered data.* `LAG`, `ROLLING`, `DIFF`, and
  `EXPANDING` read values from *other rows*. With no temporal order at all they
  silently import future information into the training set — the single most common
  way a tabular pipeline reports a score it cannot reproduce in production. These
  ops are removed when the data has no time column.
- *Ops with a missing operand.* A `RATIO` can lose its denominator to column
  filtering, which would reach the executor as an undefined operation.
  Minimum-input-count checks drop those.

---

## 6. Model Selection Agent

`agents/model_selection.py` · `AgentName.MODEL_SELECTION`

**Inputs.** n, p, the n/p ratio, feature mix, class balance, missingness, the
explainability requirement, available families from the model zoo, the experiment
budget, and memory suggestions.

**Output.** `ModelSelection` — ranked candidates with suitability, rationale,
expected strengths and weaknesses, initial params, a baseline flag, and a tuning
priority; plus a `summary`, a `reasoning` field that must be the *comparative*
argument, `excluded_families` with an `exclusion_rationale`, and a validation
strategy with its own rationale.

**Effort.** `xhigh`, 16k tokens.

**The reflex it exists to prevent.** Reach for gradient boosting, add a random
forest for contrast, ship. That produces a model that is sometimes right and an
argument that is never checkable. The prompt forbids including a family "for
completeness" without saying why it belongs, and requires the case be grounded in
this dataset's measured shape.

**Guards.**

- *No baseline.* Mandated. A model that cannot beat the majority class or the mean
  has learned nothing, and without that row in the leaderboard an ROC AUC of 0.71
  reads as competence rather than as whatever it actually is. Postprocess injects
  a baseline if the agent forgot one.
- *Families whose library is absent.* Dropped with a warning rather than exploding
  inside the trainer.
- *Ignoring the compute budget.* The candidate list is truncated to
  `max_experiments`, with ranks renumbered densely.

---

## 7. Experiment Agent

`agents/experiment.py` · `AgentName.EXPERIMENT` · **hybrid**

**Inputs.** `compute()` calls `execution.trainer.run_experiments` first — real
fits, real cross-validation, real wall-clock timings. The prompt then renders the
leaderboard, a fold-to-fold margin analysis, and training costs.

**Output.** `ExperimentLog` — the leaderboard plus `leaderboard_notes`.

**Effort.** `medium`, 16k tokens. Interpretation of an existing table needs less
depth than choosing what to put in it.

**Guards.**

- *Altering a measured score.* `postprocess` rebuilds the returned `ExperimentLog`
  from the computed one field by field, so the model **physically cannot** change a
  number even if it tries. This is not paranoia about a particular model; it is
  what makes the leaderboard citable in an audit.
- *Reading noise as a win.* The margin analysis puts the fold-to-fold standard
  deviation next to the between-model gap, so "boosting won by 0.004 with a fold
  sd of 0.013" is visible as the non-result it is.

This is the purest expression of the platform's split: every number measured, the
only contribution the sentence underneath the table.

---

## 8. Hyperparameter Optimization Agent

`agents/tuning.py` · `AgentName.TUNING`

**Inputs.** The leaderboard, the best-vs-baseline gap, the cross-validation noise
floor, the row count, seconds remaining in the budget, available search methods,
and the zoo's default search space for the target family.

**Output.** `TuningDecision` — `worthwhile` (a bool, argued first), method,
target family, trial count, timeout, early stopping, a typed `search_space` of
`SearchSpaceEntry` each with its own rationale, and an expected gain.

**Effort.** `high`, 16k tokens.

**The judgement, not the step.** Most AutoML systems tune because tuning is a step
in their pipeline. Here it is a decision. On a 3,000-row table with a 0.013
cross-validation standard deviation, a 40-trial search mostly discovers which
hyperparameters happen to suit the validation folds — an overfit dressed as an
improvement, paid for out of the time budget explainability and evaluation need.
So the first output is `worthwhile`, argued from measured evidence.

**Guards.** Postprocess enforces the constraints the model cannot be trusted to
respect on its own:

- Tuning disabled by the operator.
- An exhausted clock — a trial count the remaining budget cannot pay for is
  reduced, not attempted.
- A target family that never successfully trained.

---

## 9. Explainability Agent

`agents/explain.py` · `AgentName.EXPLAIN` · **hybrid**

**Inputs.** `compute()` calls `execution.explainer.compute_explanations`: SHAP
where the library and model support it, permutation importance always, partial
dependence for the top features.

**Output.** `ExplainabilityReport` — attributions, permutation importance, a SHAP
availability flag and paths, counterfactuals,
`plain_language_explanations`, a narrative, and method notes.

**Effort.** `high`, 16k tokens.

**Guards.** Two are enforced in code rather than trusted to the prompt:

- *Numbers drifting between the prose and the charts.* Every importance value,
  direction, and file path is copied verbatim from the computed report, and
  importances are renormalised to sum to 1.0, so the percentages in the narrative
  and the numbers in the chart cannot disagree.
- *Misstating the method.* If SHAP did not run, the narrative is not allowed to
  imply that it did. An explanation that misstates its own method is worse than no
  explanation, because it is trusted more than it deserves.

With no fitted model available, the agent degrades: it records a warning and
returns an empty report rather than narrating attributions that do not exist.

---

## 10. Evaluation Agent

`agents/evaluation.py` · `AgentName.EVALUATION`

**Inputs.** `execution.diagnostics.compute_diagnostics` renders train/validation/
test scores, the generalisation gap, calibration (Brier, ECE), bootstrap
confidence intervals, residual statistics, learning-curve points, fairness slices
for the configured attributes, and concrete error examples.

**Output.** `EvaluationVerdict` — `acceptable`, a verdict rationale, an A–F grade,
bias/variance and calibration diagnoses, generalisation notes, drift risk with a
rationale, fairness slices and notes, confidence intervals, residual notes, error
analysis, learning-curve notes, weaknesses, a `recommended_action`, an action
rationale, and specific improvements.

**Effort.** `max`, 16k tokens.

**Why it is different from every other agent.** Every other agent pushes forward.
This one is allowed to say no, and its `recommended_action` **is control flow**:
`retry_feature_engineering`, `retry_model_selection`, and `retry_cleaning` send the
orchestrator back to the planner for a real second pass.

**Guards.**

- *Judging with the model's own arithmetic.* Postprocess copies the measured values
  back over whatever the model returned, keeping its *judgements* — the verdict
  labels and prose — but never its numbers.
- *A retry that cannot happen.* The recommendation is clamped when the requested
  retry could not actually run. A retry that cannot happen is a false statement
  about what comes next.
- *Vague retries.* The prompt spends its length on how to read each diagnostic and
  on the discipline of only asking for a retry when a specific change can be
  named.

---

## 11. Business Insight Agent

`agents/insight.py` · `AgentName.INSIGHT`

**Inputs.** Attributions, the leaderboard, the evaluation verdict, the business
objective from the problem definition, and the dataset's domain and grain.

**Output.** `InsightReport` — an executive summary, insights each with a headline,
detail, supporting evidence, a recommended action, an expected value, a confidence
level and an audience; plus plain-language key drivers, caveats, and suggested next
experiments.

**Effort.** `high`, 16k tokens.

**The translation, and the refusal.** From *"`tenure` carries 0.31 of the
attribution mass"* to *"accounts churn hardest in their first four months, so
onboarding is where retention spend earns the most"* — and a refusal to translate
where the measured evidence does not support it.

**Guards.** This is exactly where an LLM is most tempted to overclaim, so two
checks run:

- References to features that do not exist are demoted to plain prose.
- Any insight asserted at `high` confidence while the evaluation gate judged the
  model unacceptable is **downgraded rather than deleted**, so the reviewer sees
  both the claim and the reason to distrust it.

---

## 12. Visualization Agent

`agents/visualization.py` · `AgentName.VISUALIZATION`

**Inputs.** The task type, available columns, the leaderboard, attributions, and
the evaluation diagnostics.

**Output.** `VisualizationPlan` — prioritised `ChartSpec`s, each with a kind, a
title, columns, parameters, and a `rationale` stating the question it answers; plus
a dashboard narrative.

**Effort.** `medium`, 16k tokens.

**Why every chart states its question.** A dashboard of every plot the library can
draw is worse than six plots that each answer something a reader was going to ask,
because the reader has to work out which ones matter.

**Guards.**

- *Impossible charts.* A ROC curve on a regression run is dropped with a warning
  rather than handed to the renderer to fail on.
- *Missing staples.* The handful of charts any reader of this task type expects are
  synthesised if the agent left them out.

---

## 13. Report Agent

`agents/report.py` · `AgentName.REPORT`

**Inputs.** Everything on the state: profile, understanding, problem, plan,
cleaning and feature decisions *with their rationales*, leaderboard, tuning,
explanations, verdict, insights, chart artifacts, and measured prediction latency
and model size.

**Output.** `FinalReport` — title, subtitle, executive summary, ordered sections
with chart references, a `DeploymentRecommendation`, and appendix notes.

**Effort.** `xhigh`, 32k tokens.

**Two commitments.**

- *The audit trail is the product.* Each cleaning and feature section must carry
  the rationale recorded at the time. A report that lists transformations without
  reasons is a changelog, and a reviewer cannot audit a changelog.
- *The section set is fixed.* Nine headings, in order, always. Downstream renderers
  and human readers both rely on the shape being identical every time, so
  `postprocess` synthesises any skipped section out of recorded state rather than
  shipping a document with a hole in it.

**Guards.**

- *Reading as deployment approval when it is not.* When the evaluation gate judged
  the model unacceptable, no wording may read as approval. The agent is told, and
  the postprocessor enforces it.
- *Deployment advice from nothing.* `DeploymentRecommendation.rationale` must be
  grounded in the measured latency, throughput need, and model size — all of which
  are on the state as real numbers.

---

## 14. Q&A Agent

`agents/qa.py` · reuses `AgentName.REPORT` · invoked outside the plan graph

**Inputs.** A completed `RunSummary` and a question. `state.extras['question']` in,
`state.extras['answer']` out.

**Output.** `QuestionAnswer` — the question, a grounded answer, an `evidence` list
of concrete references, a confidence level, caveats, and suggested follow-ups.

**Effort.** `high`, 16k tokens.

**Two question shapes needing opposite treatments.** *"Why did accuracy decrease?"*
is answerable: the recorded history contains the plan revision, the leaderboard
delta, and the surrounding events, so it is retrieval and explanation. *"What
happens if I remove income?"* is a counterfactual the run never tested, and the
only honest answer names what the evidence suggests and what would have to be
re-run to actually know.

**Guards.**

- *The plausible fabrication* — inventing a score for a model that was never
  trained, or a reason that appears nowhere in the log. The prompt's central rule
  and the required `evidence` field mean every claim must trace to something the
  orchestrator recorded.

> **Note.** `AgentName` has no dedicated member for this agent, so it declares
> `AgentName.REPORT`. It is resolved directly by `runner.py` rather than through
> the `AGENTS` registry, so the two do not collide in practice — but the enum could
> use a `QA` member.

---

## Effort ladder at a glance

| Effort | Agents | Rationale |
|---|---|---|
| `max` | Planner, Evaluation | These two shape the whole run. The planner decides what happens; the evaluator decides whether it happens again. |
| `xhigh` | Problem, Model Selection, Report | Decisions everything downstream is conditioned on, and the deliverable. |
| `high` | Dataset, Cleaning, Features, Tuning, Explain, Insight, Q&A | Substantive judgement over measured facts. |
| `medium` | Experiment, Visualization | Interpretation and selection over results that already exist. |

Token budgets are 16k except the Planner and Report Agent at 32k. On Opus 5 the
budget covers thinking **plus** visible output, so these are sized for both.

Set `AUTOML_DEFAULT_EFFORT` to change the floor for agents that do not override,
and `AUTOML_ENABLE_THINKING=false` to switch adaptive thinking off — which clamps
effort to `high`, because the API rejects disabled thinking above that.

---

## What each agent does offline

With `AUTOML_OFFLINE=1` (or `amla run --offline`) no agent calls Claude. Each one
asks the rule engine in `core/offline.py` for the same output type, and the result
flows through the same `postprocess` grounding and `apply` as a model's would.
The rules are keyed by output model, so every agent above is covered.

How much each agent loses is not uniform, and it is worth knowing which is which:

| Agent | Offline quality | Why |
|---|---|---|
| Cleaning | **Near-parity** | The decision is a function of measured distribution shape. Skewness picks median over mean; cardinality picks mode over a missing-category level; a leakage score picks a drop. |
| Problem | **Near-parity** | Task type follows from the target's kind and class count; the metric follows from the task and the measured imbalance. |
| Model Selection | **Good** | Row count, column count, and installed libraries genuinely determine the sensible shortlist. |
| Tuning | **Good** | Worthwhileness is a budget-and-headroom calculation against measured scores. |
| Experiment, Explain, Evaluation | **Near-parity on numbers, weaker on prose** | These are `HybridAgent`s: the measurements were always computed deterministically, so only the narration changes. |
| Features | **Fair** | Encoding, scaling, and skew correction are mechanical. Interaction and domain-derived features are not, and offline mode proposes none. |
| Dataset, Visualization, Report | **Fair** | Structure is described accurately; domain framing degrades to keyword-matching column names. |
| Insight | **Weak** | Restates measured drivers in plain language. It cannot explain *why* a driver drives, which is the agent's actual purpose. |
| Q&A | **Not supported** | Returns a run summary rather than answering the question, and says so. |

Rule-derived text is prefixed `[rule]` everywhere it appears.
