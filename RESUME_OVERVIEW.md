# AI Data Modeler — Project Overview

Reference sheet for resumes, portfolios, and interviews.
All figures below were measured from the repository and from real end-to-end runs.

---

## One-line summary

An autonomous multi-agent AI data scientist: it profiles a dataset, infers what
machine-learning problem it poses, plans a pipeline for *that* dataset, cleans and
engineers features, trains and compares models, tunes, explains, evaluates, and writes a
business report — recording a justification for every decision it makes.

## Project scale

| Component | Files | Lines |
| --- | ---: | ---: |
| Backend (Python) | 79 | 43,478 |
| Frontend (TypeScript / React) | 52 | 8,517 |
| Tests | 11 | 6,213 |
| **Total** | **142** | **~58,200** |

| Capability | Count |
| --- | ---: |
| Orchestrated reasoning agents | 13 (plus a Q&A agent) |
| Data-source connectors | 16 |
| Model families | 28 |
| Feature-engineering operations | 31 |
| Cleaning actions | 12 |
| ML task types recognised | 14 |
| Chart types | 21 |
| Tests passing | 799 (14 skipped, 0 failing) |

## The central idea

Most "AI does data science" tools let the language model produce the numbers, which makes
the output impossible to verify. This system enforces a hard separation:

> **Claude reasons. Python computes.**
> No agent ever calculates a statistic or trains a model.
> No executor ever makes an unexplained choice.

Agents receive *measured* facts — skewness computed by pandas, a class ratio counted from
the actual column, an AUC from scikit-learn — and return typed Pydantic decisions. Executor
modules then apply those decisions with real library code.

Two mechanisms enforce the split rather than merely requesting it:

1. **`rationale` is a required field** on every decision schema, so an agent structurally
   cannot emit a transformation it did not argue for. Explainability becomes a type
   constraint instead of a prompt instruction.
2. **Measured values are overwritten in post-processing.** For agents that interpret
   computed results, every numeric field is replaced with the value the executor actually
   measured, so the model cannot alter a score.

The consequence is concrete: when the report says *"imputed `annual_income` with the
median"*, you can read the skewness value that drove the choice, re-derive it from the CSV
yourself, and find the sentence the agent wrote explaining why the mean would have been
worse.

**Offline mode** demonstrates the same split from the other side. With `--offline` the
system runs end-to-end with zero model calls and no API key: a deterministic rule engine
supplies each agent's typed decision instead of Claude, deriving it — and its rationale —
from the measured profile. A real run on the 3,000-row churn example completes in ~42s for
$0.00, still catching the planted leakage column at 0.998 association and still choosing
the median for `annual_income` on its measured skewness of 2.43. Because only the reasoner
is swapped, the training, tuning, SHAP, and reporting paths are the same code either way —
which is what makes the boundary a real architectural seam rather than a diagram.

## Architecture

```
data source ──> ingestion ──> profiling (deterministic: MEASURE everything)
                                   │
                                   ▼
                        run-context digest, frozen once
                                   │
                    ┌──────────────┴──────────────┐
                    │        orchestrator         │  plan · retry · replan · approve
                    └───┬─────────────────────┬───┘
                        │                     │
              agents/ (REASON)        execution/ (COMPUTE)
              typed decisions  ──>    pandas · scikit-learn
              with rationales   <──   XGBoost · Optuna · SHAP
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                    reporting (md/html/pdf/pptx) · storage · API · UI
```

**Pipeline stages:** ingest → profile → understand → identify problem → plan → clean →
engineer features → split → select models → run experiments → tune → explain → evaluate →
insights → visualise → report.

## Technology

**Backend** — Python 3.11, FastAPI, Pydantic, SQLAlchemy, Claude API (structured outputs,
prompt caching, adaptive thinking)

**ML** — pandas, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, statsmodels

**Frontend** — Next.js 16, React 19, TypeScript, TailwindCSS, Plotly, Server-Sent Events

**Infrastructure** — SQLite/PostgreSQL, Docker, pytest, Typer CLI

---

## Resume bullets

### Short version

> Built an autonomous multi-agent AI data-science platform (58k LOC, 799 tests) that
> profiles datasets, plans ML pipelines, trains and evaluates models, and generates
> explained reports. Architected a strict reasoning/computation split — LLM agents make
> typed, schema-enforced decisions; scikit-learn, XGBoost, Optuna, and SHAP execute them —
> making every output reproducible and auditable.

### Detailed version

> - Designed a 13-agent orchestration engine (Python, FastAPI, Claude API) with retry
>   policies, human-in-the-loop approval gates, and a bounded self-improvement loop that
>   replans when evaluation rejects a model, keeping the better model if a retry regresses.
> - Enforced explainability structurally by making `rationale` a required field on every
>   decision schema; a live run produced 8 cleaning decisions, all citing measured evidence.
> - Implemented leakage-safe feature engineering — out-of-fold target encoding with
>   smoothing toward the prior, time-ordered splits, group-aware cross-validation — with
>   scalers and dimensionality reduction deferred into a Pipeline fitted inside each CV fold.
> - Correctly detected and dropped a planted target-leakage column at 0.9975 single-feature
>   AUC; the winning model reached 0.791 ROC AUC against a 0.500 baseline.
> - Diagnosed and fixed six defects found through live execution, including a prompt-cache
>   misconfiguration writing 67,000 tokens per run and reading zero, verified against the
>   live API.
> - Built a real-time React dashboard streaming agent reasoning over Server-Sent Events with
>   lossless reconnect via a sequence cursor.

### Stack line

> Python · FastAPI · Claude API · pandas · scikit-learn · XGBoost · LightGBM · Optuna ·
> SHAP · Plotly · SQLAlchemy · Next.js · TypeScript · TailwindCSS · pytest · Docker

---

## The 60-second spoken explanation

> The problem with letting an LLM do data science is that you can't trust the numbers — if
> the model reports an accuracy, you have no idea whether it computed it or invented it.
>
> So I split the system in two. The agents reason but never compute; they receive measured
> statistics and return typed decisions. Deterministic Python does all the actual work with
> scikit-learn and pandas. Every decision schema has a required `rationale` field, so an
> agent literally cannot emit a transformation without arguing for it — explainability is
> enforced by the type system rather than requested in a prompt.
>
> Thirteen agents handle the stages: one profiles the data, one infers the problem type, a
> planner writes an execution strategy adapted to that specific dataset, then cleaning,
> feature engineering, model selection, training, tuning, explainability, and evaluation.
> The evaluation agent is a real quality gate — if it rejects the model it triggers a
> bounded replan, and if the retry produces something worse the system keeps the better
> model and says so.
>
> The interesting engineering was mostly in the failure modes rather than the happy path.

---

## Three strongest interview stories

### 1. The layering lesson

Two separate bugs turned out to be the same mistake in opposite directions.

A column-validation guard ran **too early**: it checked feature-engineering inputs against
the source schema, which stripped references to columns the plan's own earlier steps were
about to create. Feature engineering is sequential, so a scaling step legitimately consumes
a column a log-transform produces — but at validation time that column doesn't exist yet.

Separately, feature scaling ran **too late**: it was left to the agent's judgement, so when
its scaling step got dropped, logistic regression trained on raw-magnitude inputs and failed
to converge.

The lesson: *put a guarantee at the layer that has the information to enforce it.* Column
existence can only be judged by the executor, after the creating steps have run. Scaling for
a penalised linear model or a distance metric is a model **requirement**, not a modelling
preference, so the trainer enforces it regardless of what the plan asked for.

### 2. Measure, don't assume

Prompt caching looked correct in the code. In production it wrote 67,152 tokens per run and
read back **zero**.

Agent steps take 60–390 seconds each, so the default five-minute cache TTL expired between
every call. The system was paying the cache-write premium eleven times per run and
collecting nothing — strictly worse than not caching at all, and completely invisible unless
you inspect `cache_read_input_tokens`.

Fixed by switching to a one-hour TTL, then verified against the live API: two calls
separated by 330 seconds, with 4,064 tokens read back from cache on the second.

### 3. Type overloading as a failure mode

A single parameter took down every model in a run, including the trivial baseline.

scikit-learn's `min_frequency` is overloaded on type: an integer means an absolute row count,
a float between 0 and 1 means a proportion. The code coerced everything to float, so a
perfectly reasonable "pool categories seen fewer than 20 times" became `20.0` — neither a
valid count nor a valid proportion. Because the encoder lived in the *shared* preprocessor,
one bad value failed all five candidates simultaneously and the run ended with no viable
model.

---

## Questions to be ready for

**"How do you stop the LLM hallucinating a metric?"**
The HybridAgent pattern. Deterministic code computes the results first; the agent then
interprets them, and post-processing overwrites every numeric field with the measured value
before the output is accepted. The model can shape the narrative but not the numbers.

**"How do you prevent data leakage?"**
Several layers. Target encoding is out-of-fold with smoothing toward the prior. Scalers,
PCA, and feature selection are deferred into an unfitted scikit-learn Pipeline that the
trainer fits inside each cross-validation fold, so no fold ever sees its own validation
rows. Splits are time-ordered when a temporal column exists and group-aware when an entity
key exists. The profiler independently scores every feature's association with the target
and flags near-perfect predictors as leakage candidates.

**"Does it actually work?"**
Yes, on a 3,000-row churn dataset. It caught a deliberately planted leakage column at 0.9975
single-feature AUC and dropped it with an evidence-citing explanation, produced eight
cleaning decisions all grounded in measured statistics, trained five models, and the winner
reached 0.791 ROC AUC against a 0.500 baseline. The tuner improved LightGBM and still
refused to crown it, because logistic regression still scored higher — reporting a
tuned-but-worse model as the winner would have been a measurement lie.

**"What would you do differently?"**
Test against the live API earlier. Every one of the six real defects was found by running
the system end to end, not by reading code or running the offline test suite — the offline
suite passed while the pipeline was failing in ways only a real run exposed.

---

## Honest framing

**Do not claim production deployment or users.** This is a working system with real
end-to-end runs, a 799-test suite, and measured results. That is already substantial. It has
not served production traffic.

**Be ready to say it was AI-assisted.** That is normal practice and not a weakness. The
credible version is owning the architecture and the judgement calls: the reasoning /
computation split, making `rationale` structurally required rather than requested, deciding
which layer owns which guarantee, and the debugging narratives above. Those are the parts an
interviewer will probe.

**Known limitations worth stating honestly:**

- Two fixes (a schema-compilation retry and feature-naming guidance) are unit-tested but
  have not yet been exercised by a full live run.
- A full run takes roughly 35 minutes and costs about $2–3.50 in API usage, dominated by
  reasoning latency rather than compute.
- Several connectors (Snowflake, Kaggle, cloud object storage) are implemented but not
  exercised against live services.
