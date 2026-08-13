# Architecture

This document explains *why* the system is shaped the way it is. The README covers
what it does; this covers the decisions that constrain how it does it, and what
breaks if you ignore them.

## Contents

- [The central split](#the-central-split)
- [Module map](#module-map)
- [Data contracts](#data-contracts)
- [The run lifecycle](#the-run-lifecycle)
- [Prompt caching, and why context is frozen](#prompt-caching-and-why-context-is-frozen)
- [State: the blackboard](#state-the-blackboard)
- [Grounding: never trust an agent's column name](#grounding-never-trust-an-agents-column-name)
- [Leakage discipline](#leakage-discipline)
- [The self-improvement loop](#the-self-improvement-loop)
- [Human-in-the-loop approvals](#human-in-the-loop-approvals)
- [Failure policy](#failure-policy)
- [Events, and the three features built on them](#events-and-the-three-features-built-on-them)
- [Dataset memory](#dataset-memory)
- [Testing strategy](#testing-strategy)
- [Extending the system](#extending-the-system)

---

## The central split

Every component is on exactly one side of a line:

| | **Agents** (`agents/`) | **Executors** (`execution/`, `profiling/`, `ingestion/`, `reporting/`) |
|---|---|---|
| Does | Reason, choose, explain | Measure, transform, fit, render |
| Input | Measured facts as prompt text | A typed decision object |
| Output | A validated Pydantic model with a `rationale` | Mutated dataframes, fitted models, files |
| Determinism | No | Yes, given a seed |
| Can compute a statistic | **No** | Yes |
| Can make an unexplained choice | **No** | **No** |

The last row is the interesting one. Executors do not make choices at all — they
apply them. When `cleaning_ops` imputes a column with the median, it does so
because a `CleaningDecision` said `strategy=MEDIAN` and carried a `rationale`
explaining why. There is no `if skewness > 1: use_median()` heuristic buried in the
executor, because a heuristic like that is an unexplained choice that no report
would ever surface.

This is why `rationale` is a **required** field rather than an optional one. It is
the product requirement expressed as a type: an agent physically cannot emit an
unexplained transformation, and a schema change that gives `rationale` a default
is caught by `tests/test_schemas.py::test_rationale_is_required`.

### What this buys, concretely

Ask "why is this number what it is?" about anything in the output and the answer
is always mechanical:

- **A statistic** — computed by `profiling/`, from the actual data, reproducibly.
  Re-run the profiler and get the same value.
- **A choice** — made by an agent, with the statistic it cited recorded alongside
  the decision in `run_summary.json`.
- **A model score** — computed by scikit-learn on a split whose strategy and
  rationale are recorded on `DataSplits`.

There is no third category of "the framework decided this for opaque reasons".

### Offline mode sits on the decision side of the line

`core/offline.py` replaces the *reasoner*, not the split. When
`settings.offline` is set, `BaseAgent.run` asks a rule engine for the same output
type instead of asking Claude, and the result still flows through `postprocess`
and `apply` — so every grounding check that guards a model's output guards this
one too.

This is the reason the rules live there and not in `execution/cleaning_ops.py`.
An `if skewness > 1: use_median()` buried in the executor would be exactly the
unexplained choice the table above forbids: invisible to the report, unattached
to a rationale, and impossible to override. The same branch expressed as a
`CleaningDecision` carrying *"skewness 2.425 (threshold 1.0), with mean 62,854
against median 50,560"* is a decision like any other — recorded, auditable, and
rendered into the report next to the model's own.

So the honest statement of what offline mode costs is narrow: the decisions get
worse, because rules read statistics and a model reads meaning. Nothing about the
architecture relaxes. Rule-derived reasoning is prefixed `[rule]` so a reader can
always tell the two apart, and `tests/test_offline.py` asserts that every
rationale the engine writes cites a measurement.

---

## Module map

```
automl_architect/
├── config.py              Settings: deployment wiring. Never per-run.
├── runner.py              Library facade: AutoMLArchitect, analyse()
├── cli.py                 Typer CLI with a live Rich display
│
├── core/                  Contracts and cross-cutting machinery
│   ├── schemas.py         EVERY data contract. Read this first.
│   ├── llm.py             LLMClient: the single place this talks to Claude
│   ├── agent.py           BaseAgent, HybridAgent: events, usage, grounding
│   ├── state.py           RunState (the blackboard), DataSplits
│   ├── events.py          EventBus: ordered, replayable, thread-safe
│   ├── context.py         DatasetProfile -> prompt text (the cached prefix)
│   └── errors.py          RetryableError vs FatalError, and the leaves
│
├── ingestion/             One connector per SourceKind, behind router.load_source
├── profiling/             Deterministic measurement: stats, semantic types, leakage
├── agents/                Thirteen specialists + Q&A, lazily registered
├── execution/             The deterministic half
├── orchestrator/          engine (drives), graph (step DAG), policies (retry/replan)
├── reporting/             charts, markdown, html, pdf, pptx, dashboard
├── storage/               repository (runs, events, fingerprints), memory, artifacts
└── api/                   FastAPI app, routes, SSE stream, RunManager threads
```

The dependency direction is strict: `core/` depends on nothing but `config.py`;
everything else depends on `core/`; `orchestrator/` depends on `agents/` and
`execution/`; nothing depends on `orchestrator/` except `runner.py` and `api/`.

`agents/__init__.py` registers agents **lazily** by dotted location. A syntax
error or a missing optional dependency in one of fourteen sibling modules would
otherwise break `import automl_architect.agents` for all of them, and with it the
CLI, the API, and the orchestrator.

---

## Data contracts

`core/schemas.py` is the single source of truth. Two rules govern it.

### 1. LLM-facing models stay inside Claude's structured-output schema subset

An agent's output model is compiled into a JSON schema the model must satisfy.
That compiler rejects:

- **Free-form objects.** `dict[str, Any]` becomes `{"type": "object"}` with no
  declared `properties`, which is not representable. Open-ended configuration
  therefore travels as `list[Param]` — typed key/value pairs — and
  `params_to_dict()` coerces values back to real ints, floats, bools, `None`,
  lists, and dicts at the executor boundary.
- **Recursion.** No model may reference itself, directly or transitively.

Both are enforced by tests that walk the generated JSON schema, so a new model is
covered automatically rather than by remembering to add a test.

```python
# Wrong: the schema compiler will reject this.
params: dict[str, Any]

# Right.
params: list[Param]
...
kwargs = params_to_dict(decision.parameters)   # {"n_estimators": 300, ...}
```

### 2. Every decision carries its reasoning

`rationale` is required on `ProblemDefinition`, `PlanStep`, `CleaningDecision`,
`FeatureDecision`, `ModelCandidate`, `TuningDecision`, `ChartSpec`, and
`DeploymentRecommendation`. Several models go further:

- `FeatureDecision` also requires a `hypothesis` (the mechanism) and a `risk`
  (leakage or overfitting exposure, and its mitigation).
- `CleaningPlan` has `skipped_considerations`: transformations deliberately *not*
  applied, and why. A plan that only records what it did hides half its reasoning.
- `ModelSelection` has both `summary` and `reasoning`, where `reasoning` is
  explicitly the comparative argument across candidates rather than a list of
  facts about each.
- `EvaluationVerdict` requires `verdict_rationale`, `recommended_action`, and
  `action_rationale`.

All models set `extra="forbid"`, so a hallucinated field is a loud validation
error rather than silently-dropped data.

---

## The run lifecycle

```
 load ──► profile ──► freeze context ──► consult memory
                                              │
                                              ▼
   understand ──► identify problem ──► PLAN ◄──────────┐
                                        │              │
                                        ▼              │
                          ┌─────── walk the plan ──────┤
                          │                            │
   clean ──► features ──► split ──► select ──► train ──┤
                          │                            │
                          ├──► tune ──► explain        │
                          │                            │
                          └──► EVALUATE ───────────────┘
                                   │      not acceptable,
                                   │      replans < max_replans
                              acceptable
                                   ▼
              insights ──► charts ──► report ──► persist
```

`orchestrator/graph.py` holds the canonical step definitions and their
dependencies. `orchestrator/engine.py` reconciles the Planner's proposed plan
against that graph — an agent-written plan is a *proposal*, and steps it invents,
misorders, or omits are reconciled against real dependencies rather than executed
blindly. `orchestrator/policies.py` owns retry and replan decisions.

Steps are recorded as `StepRecord`s with status, attempts, duration, and error, so
the summary shows not just what ran but what was retried and what was skipped.

---

## Prompt caching, and why context is frozen

This is the most easily-broken invariant in the codebase, so it is worth
understanding.

Anthropic's prompt caching is a **prefix match**. `core/llm.py` orders system
blocks stable-first:

```
block 0   PLATFORM_PREAMBLE      identical across every agent and every run   [cache]
block 1   run-context digest     identical across agents within one run       [cache]
block 2   agent instructions     unique per agent                             [no cache]
```

Blocks 0 and 1 carry cache breakpoints, so the second and subsequent agents in a
run read them from cache at a tenth of the input price. With thirteen agents and a
context digest that can run to thousands of tokens on a wide table, that is most
of the prompt cost.

**Putting anything volatile into block 0 or 1 silently destroys this.** A
timestamp, a step counter, a retry number, or a dict whose iteration order varies
changes the prefix bytes and every agent takes a cache miss. Nothing errors; the
bill just goes up. Hence:

- `core/context.py` is deterministic by construction — no timestamps, no
  set iteration, no run counters.
- `RunState.freeze_context()` builds the digest **once**, right after profiling,
  and is idempotent: the first call wins and later calls are ignored.
- Volatile content belongs in the **user turn**, which is not cached.

`tests/test_agents.py` asserts block ordering, breakpoint placement, and that two
calls to `system_blocks()` produce byte-identical stable blocks.

The second reason to freeze is correctness, not cost: all thirteen agents reason
over the *same* facts. Without freezing, the Evaluation Agent could be arguing
about a dataset shape the Cleaning Agent never saw.

### The TTL is 1 hour, and that is not a tuning preference

A byte-identical prefix is necessary but not sufficient — the entry also has to
still exist. Anthropic's default ephemeral TTL is **5 minutes**, and agent steps
in a real run do not fit inside it. A measured churn run spent 96s understanding
the data, 299s planning, 172s selecting models, and 389s writing the report; the
gap between consecutive cached prefixes routinely exceeds five minutes.

That run wrote **67,152 tokens to cache and read back 0**. Every call paid the
1.25× write premium and collected nothing — strictly worse than not caching at
all, and invisible unless you look at `cache_read_input_tokens`.

So `Settings.cache_ttl` defaults to `"1h"`. A 1-hour entry costs 2× to write
rather than 1.25×, but is read at 0.1× by every later agent. Break-even is three
calls; a run makes eleven or more.

The operational lesson generalises: **prompt caching is not verified by reading
the code that sets `cache_control`.** It is verified by asserting
`usage.cache_read_input_tokens > 0` on a second call, which is what
`scripts/check_cache_ttl.py` does against the real API.

---

## State: the blackboard

`RunState` is the single mutable object threaded through the orchestrator. Agents
read from it and write their typed output back; executors mutate the dataframes.

```python
state.raw_df          # as loaded — the audit copy, never mutated
state.working_df      # after cleaning
state.feature_frame   # after feature engineering (keeps target/group/time columns)
state.feature_names   # authoritative list of X columns
state.splits          # DataSplits, with the strategy and its rationale
state.preprocessor    # UNFITTED sklearn pipeline; the trainer fits it on train only
state.best_model      # fitted estimator
state.df              # property: the most-processed frame available
```

Keeping this in one object rather than passing tuples between steps is what makes
replanning possible: a revised plan re-runs steps against the same state, and each
step overwrites only its own slot. `state.to_summary()` serialises everything
except the fitted objects into a `RunSummary`.

`state.add_warning()` is the degradation channel. It appends to
`summary.warnings` *and* emits a `WARNING` event, so reduced capability is visible
in the report, in the live stream, and in the audit trail simultaneously.

---

## Grounding: never trust an agent's column name

The single most damaging failure mode in a system like this is an agent
confidently naming a column that does not exist, which then explodes three steps
later inside pandas with a `KeyError` that has no obvious connection to its cause.

`BaseAgent.keep_known_columns()` filters agent-supplied column references against
the real schema — the live dataframe if there is one, else the profile — and
records what it dropped:

```python
kept = self.keep_known_columns(decision.columns, state, context="cleaning plan")
# -> a warning on state, and a hallucination becomes a logged event, not a crash
```

Every agent applies this in `postprocess()`, and `tests/test_agents.py` verifies it
generically: for each discovered agent, a canned response is mutated to include
`__a_column_that_does_not_exist__` in every column-reference field, and the test
asserts that reference does not survive into the executor layer.

Executors filter again, on the principle that a defence applied once is a defence
you will eventually forget to apply.

---

## Leakage discipline

Leakage is handled at three separate points, because it arrives three different
ways.

**1. In the data (`profiling/leakage.py`).** Single-feature association against
the target — AUC for classification, correlation and mutual information for
regression. A near-perfect score on a non-trivial feature is the classic signature.
Findings carry a `score`, a `method`, a `severity`, and a `reason`, and the
Cleaning Agent decides what to do with them. The detector reports; it does not
drop. Whether a column is *legitimately* available at prediction time is a
business question.

**2. In the transform (`execution/feature_ops.py`).** Anything fitted from the
feature distribution — scaling, binning, power transforms, PCA/SVD, univariate
selection, variance thresholds, and one-hot category sets — is **deferred** into an
unfitted sklearn pipeline on `state.preprocessor`, which the trainer clones and
fits on the training partition only. Fitting a `StandardScaler` on the full column
before splitting leaks the test set's mean into the transform; the effect is small
but it is real, and it is free to avoid.

Target encoding is stronger still: K-fold out-of-fold with smoothing toward the
prior, and fitted on the training partition alone when one already exists. Target
encoding fitted on the full column writes the answer into the feature, and is
probably the most common cause of a validation score that collapses in production.

**3. In time (`execution/splitter.py`, `feature_ops`).** Lags, rolling windows,
diffs, and expanding statistics sort by the temporal column and are bounded by the
group key. Without both, a "lag" is an arbitrary neighbouring row and one entity's
history bleeds into the next entity's. Temporal splits never straddle time, and
grouped splits never put one entity on both sides.

`examples/churn.csv` ships a deliberate leak — `cancellation_tickets`, AUC 0.9975
against the target versus 0.627 for the best legitimate feature — so the detector
has a true positive to find. A detector tested only on datasets with no leakage has
been tested for false positives and nothing else.

---

## The self-improvement loop

The Evaluation Agent is a real gate. It returns `acceptable: bool` and a
`recommended_action` from a closed set:

```
accept · retry_feature_engineering · retry_model_selection · retry_cleaning
collect_more_data · reject
```

A rejection raises `ReplanRequested` — a control-flow signal, not an error — which
unwinds to the Planner with the verdict's `specific_improvements` as context. The
new plan gets `revision` incremented and a `revision_reason`; the old one is kept
in `plan_history`.

Three bounds keep this from being a liability:

- `max_replans` (default 2) caps the loop.
- `enable_self_improvement=False` disables it entirely.
- The engine keeps the **better** outcome across attempts, so a replan that makes
  things worse does not overwrite a good result with a bad one.

If the loop exhausts its budget with the model still unacceptable, the run
completes with an honest failing verdict rather than looping or pretending.

---

## Human-in-the-loop approvals

With `require_approval=True`, any step whose plan marks it `destructive=True`
suspends before executing. The engine creates an `ApprovalRequest` recording the
action summary, affected columns, an affected-row estimate, and a severity, emits
`APPROVAL_REQUESTED`, and raises `ApprovalRequired`.

The run's status becomes `AWAITING_APPROVAL` and the summary is persisted, so the
decision can arrive minutes or days later, through the CLI (interactive prompt),
the library (`architect.resume(request_id, approved)`), or the API
(`POST /api/runs/{id}/approvals/{request_id}`). A rejection raises
`ApprovalRejected`; the step is skipped or the run ends, and either way the
decision, the decider, and the note are recorded.

`destructive` is set by the Planner and the Cleaning Agent on anything that drops
columns or rows. It is deliberately a schema field rather than an executor
heuristic: what counts as destructive is a judgement about the data, so the agent
that understands the data makes it.

---

## Failure policy

`core/errors.py` splits on `RetryableError` vs `FatalError`, and the orchestrator's
retry policy branches on those base classes rather than on string matching — a new
error type slots in by choosing its parent.

The operating rule is **degrade, do not crash**. Concretely:

| Failure | Behaviour |
|---|---|
| Optional dependency missing | `MissingDependencyError` with an install hint; the feature is skipped, a warning is recorded, the run continues |
| One model family fails to fit | `ExperimentResult.failed=True` with the error; other families continue; only *all* families failing is fatal (`NoViableModelError`) |
| SHAP unavailable | `shap_available=False`; permutation importance is used instead |
| kaleido unavailable | Interactive HTML charts only; PNG paths stay `None` with a recorded error |
| Agent returns an unparseable object | `LLMClient` re-asks with the validator's complaint appended, up to 3 attempts |
| Response truncated at `max_tokens` | Retried with double the budget |
| Model declines the request | Re-served by the fallback model if `enable_refusal_fallback`; a warning notes the substitution |
| Rate limit / 5xx / timeout | SDK retries with backoff, then `LLMTransientError`, then the step's own retry policy |
| Chart or report format fails | Recorded in `bundle.warnings`; other formats still render. `write_report` never raises |

`tests/` covers each of these paths explicitly, because the whole point of a
degradation path is that it runs when nobody is watching.

---

## Events, and the three features built on them

`EventBus` is a first-class object rather than a set of logging calls because three
separate features are built on the same log:

1. **The live UI stream.** `api/routes.py` bridges the synchronous bus to SSE by
   pushing into a queue. Events carry a monotonic `sequence`, so a reconnecting
   `EventSource` resumes with `?after=<n>` rather than replaying everything.
2. **The audit trail in the report.** Which agent decided what, when, at what
   token cost.
3. **The Q&A agent's evidence base.** "Why did accuracy drop?" is answered by
   replaying events, not by recollection.

The bus is synchronous and thread-safe. A subscriber that raises is logged and
skipped — a broken listener must never fail a run.

---

## Dataset memory

After each run, `storage/memory.py` writes a `DatasetFingerprint`: shape, dtype
counts, missingness, duplicate fraction, task type, target kind, imbalance ratio,
column names, and the winning family and score.

Before planning, the engine looks for structurally similar past runs and hands the
Planner a `MemorySuggestion` — recommended families, feature ops that worked,
cautions from what did not. It is advisory context, not a decision: the Planner
still has to argue for whatever it chooses, and the suggestion is grounded in
recorded outcomes rather than in a prior.

This is what makes the system improve across runs on similar data rather than
starting cold each time.

---

## Testing strategy

The suite is fully offline. Two assets make that possible.

**`FakeLLMClient`** implements the real client's surface — `structured()`,
`text()`, `count_tokens()`, a `usage` accumulator — and resolves each structured
call by **output model type**. A test registers a canned `ProblemDefinition` and
every agent that asks for one gets it. Calls are recorded, so a test can assert on
the prompt an agent built without knowing anything about that agent's internals.
Usage accounting is deterministic (token counts derive from prompt length) and it
simulates cache reads, so cost assertions are meaningful.

**Schema autofill** (`conftest.synthesise`) constructs a minimal *schema-valid*
instance of any Pydantic model by walking its required fields. This is what keeps
the suite from being brittle to schema growth: a new optional field needs no test
change, a new required field is filled automatically, and an unregistered output
model still yields a usable value.

Above those sit realistic canned responses for all twelve agent output models,
written as a real run on `churn.csv` would produce them. So
`tests/test_orchestrator.py` exercises the entire pipeline — replanning,
approvals, persistence, reporting — with no credentials and no network.

What it still costs is compute: the reasoning is free but the training,
diagnostics, charting, and report rendering are real, so one full run is 20–65
seconds. `test_orchestrator.py` and `test_api.py` are therefore marked `slow`
(`make test-fast` skips them) and share one completed run across every assertion
that only reads the result. Tests that need a *fresh* run — replanning,
approvals, configuration overrides — pay for one and say why.

Tests that would hit the real API are marked `@pytest.mark.live` and skip without
`AUTOML_LIVE_TESTS=1`. An autouse fixture points the SDK at a dead local port so an
accidental live call fails in milliseconds instead of reaching the network.

The suite also tolerates modules being written in parallel: `import_or_skip`
accepts several candidate paths and skips rather than collapsing collection.

---

## Extending the system

### Add a model family

1. Add the member to `ModelFamily` in `core/schemas.py`.
2. In `execution/model_zoo.py`: register a builder, add it to the task-suitability
   map, add a default search space, and declare `supports_proba`.
3. If it needs an optional package, import it **inside** the builder and raise
   `MissingDependencyError` when absent. Never import it at module scope.

`tests/test_execution.py::TestModelZoo` then covers it automatically — it smoke-fits
every offered family and checks that every search-space key is a real parameter.

### Add a feature operation

1. Add the member to `FeatureOp`.
2. Implement the handler in `execution/feature_ops.py` and register it in the
   dispatch table.
3. Decide which of the three rules it falls under. If it learns anything from the
   feature distribution, it goes into `state.preprocessor`, not into the frame.
4. Mention it in the Feature Agent's instructions, or it will never be chosen.

### Add an agent

1. Add the member to `AgentName`, and its output model to `core/schemas.py`
   (`rationale` required, `list[Param]` for free-form config, no recursion).
2. Subclass `BaseAgent`, implementing `instructions`, `build_prompt`,
   `postprocess` (apply `keep_known_columns`), and `apply`.
3. Register the dotted location in `agents/__init__.py::_LOCATIONS`.
4. Add the step to `orchestrator/graph.py` with its dependencies.
5. Add a canned response to `tests/conftest.py::CANNED_BUILDERS` if the autofilled
   version is too bland to exercise the executor.

If the agent needs to interpret computed results, subclass `HybridAgent` and put
the deterministic work in `compute()` — that runs before `build_prompt`, so the
prompt can carry measured facts.

### Add a data source

1. Add the member to `SourceKind`.
2. Implement a connector in `ingestion/` and register it in `router.py`.
3. Import the driver lazily; raise `MissingDependencyError` with an install hint.
4. Return `(dataframe, IngestionResult)`, populating `schema_fields`,
   `bytes_in_memory`, and `truncated`.

Credentials are referenced by environment-variable *name* on
`DataSource.secret_env` — never by value. `DataSource` has no password field, and
`tests/test_schemas.py` asserts it never grows one.
