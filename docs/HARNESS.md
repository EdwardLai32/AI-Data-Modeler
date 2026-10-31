# Agent harness

The harness is the code around a model that constrains its actions, validates its
outputs, records its work, and decides whether to retry, stop, or recover. This
project already had typed agent decisions and an orchestration loop. The additions
make those decisions more reproducible and execution easier to supervise.

## Connect Claude or GPT

Start the API and dashboard as described in the root README. Expand **Connect your
AI**, select Anthropic or OpenAI, choose a model from the dropdown, enter your API key,
then choose **Connect and test**. A successful test creates a random connection ID
for one hour. It also works when the server's default is offline. Your ChatGPT or
Claude chat subscription is not an API credential; supply a provider API key.

The browser holds the key only in the form until connection succeeds. The server
holds connection settings in memory. Keys are excluded from settings serialization,
run configurations, database summaries, checkpoints, and model exports. They are
not saved in browser storage. Reloading the page or restarting the server requires
reconnecting. Disconnecting prevents future use of that connection ID; an active
run keeps its already-resolved settings until it finishes.

For a server-owned OpenAI connection, use local environment settings:

```dotenv
AUTOML_PROVIDER=openai
OPENAI_API_KEY=your-local-key
AUTOML_MODEL=gpt-4.1-mini
AUTOML_OFFLINE=false
```

For Anthropic, use `AUTOML_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`. Model IDs
are configurable. They must support the adapter's structured-output options.
OpenAI uses the Responses API (`store=False`); Anthropic retains the existing
Messages adapter. Provider accounts, pricing, and model availability are external
dependencies. Offline CI uses mocked SDK transports and does not verify them.

## Limits and traces

`RunConfig` supports `max_llm_calls` (default 40), `max_llm_tokens` (500,000),
`max_cost_usd` (unset), and `strict_deadline` (true). These are also available in
the new-analysis form. Call limits include retry attempts. The harness reserves
input bytes plus framing/schema allowances and maximum output tokens before each
request, then reconciles successful calls against provider usage. Unknown usage
after a failed request keeps its reservation. Conservative reservation can stop
a request before its expected token usage would reach the limit.

A dollar cap requires both input and output prices per million tokens, entered in
the connection form or `AUTOML_INPUT_PRICE_PER_MILLION` and
`AUTOML_OUTPUT_PRICE_PER_MILLION`. Cache billing, service tiers, and provider
discounts can differ; this is an estimated cap, not an invoice guarantee. The
dashboard labels unconfigured pricing as unavailable. Connection tests and Q&A
are separate requests outside the analysis run's cumulative budgets.

Training and tuning run in a spawned process with one estimator worker. The
supervisor terminates it on cancellation or deadline. This is process isolation,
not a filesystem/network sandbox or a memory quota. Other operations check the
deadline between stages; in-process profiling and report rendering can overrun.
The SDK timeout is bounded by the remaining run time. Ordered events identify
stages, retries, provider/model, usage, warnings, artifacts, and recovery.

## Validation and exported models

After initial schema/problem discovery, raw data is partitioned before learned
cleaning or features. Subsequent planning uses a training-only profile. The test
partition is unavailable to the model-selection and replanning loop. The chosen
model receives one final test evaluation, exposed separately from validation
scores in the run summary and dashboard.

`PlanPreprocessor` fits imputation, clipping, encodings, selection, and sklearn
transformers inside each training/CV pipeline. Target encodings are cross-fitted;
prediction reuses frozen mappings. Raw prediction rows do not refit anything.
Exact duplicates are removed before splitting. Row-removal operations are excluded
from the fitted estimator to preserve feature/label alignment. Sequential filling
uses fitted median/mode instead. Lag/rolling/difference/expanding features are
excluded until exported inference has an explicit history contract; target
encoding is excluded for temporal/grouped problems.

Initial discovery still sees full-table summaries, so the whole system is not a
fully blind test protocol. Repeated validation-guided replanning can overfit the
validation set. Small datasets and unseen classes still require human judgment.

The final `models/best_model.joblib` includes preparation and the fitted model.
Its metadata sidecar records raw input columns, class labels, seed, and library
versions. Load only trusted model files and use the same environment:

```python
import joblib
import pandas as pd

pipeline = joblib.load("workspace/runs/RUN_ID/models/best_model.joblib")
predictions = pipeline.predict(pd.read_csv("new_rows.csv"))
```

Classification predictions may be encoded labels; consult `class_labels` in the
metadata. Required feature columns must be present. Extra target columns are ignored.

## Recover an interrupted run

After successful stages and finalization, the harness writes a compressed state
checkpoint, then atomically replaces its manifest. It preserves partitions,
fitted models, decisions, approvals, step status, budgets, elapsed time, and the
event cursor. The last two checkpoint blobs are retained. An HMAC authenticates
the manifest and binary before deserialization. Python/library versions and the
application source fingerprint must match.

```powershell
.\.venv\Scripts\amla.exe recover RUN_ID
```

Or reopen the run in the dashboard and choose **Resume checkpoint**. A BYOK run
needs a new connection after a server restart. Completed stages are skipped;
an interrupted stage starts again. Pending approvals remain pending. Resuming
does not reset elapsed time or spending limits, so an exhausted run needs a new
analysis with larger limits. Recovery is explicit; no scheduler automatically
restarts jobs. Avoid recovering one run concurrently in multiple processes.

Back up the workspace, database, and `.checkpoint-key` together. Checkpoints
contain dataset rows and model objects; keep the workspace private. This
authentication mechanism protects integrity, not confidentiality. Never load
uploaded joblib/pickle files as checkpoints.

## Evaluate changes

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not live"
.\.venv\Scripts\amla.exe evaluate --provider offline --rows 300 --output artifacts/evaluations/offline.json
# These explicitly opt into paid provider calls using local environment keys:
.\.venv\Scripts\amla.exe evaluate --provider openai --model gpt-4.1-mini --live --rows 300 --output artifacts/evaluations/openai.json
.\.venv\Scripts\amla.exe evaluate --provider anthropic --model YOUR_MODEL_ID --live --rows 300 --output artifacts/evaluations/anthropic.json
```

The runner records dataset hashes, seeds, completion rate, validation/final test
metrics, usage, elapsed time, retries, warnings, and checkpoint presence on the
same three synthetic datasets. Use the same row cap and repetitions for fair
comparisons. These examples are a regression benchmark, not evidence of broad
model quality. CI runs tests, the offline evaluation, TypeScript checks, and the
frontend production build without real provider credentials.

Tests cover provider schema handling, retries/refusals, budget exhaustion,
training-only statistics, batch-invariant prediction, model serialization,
checkpoint tampering and recovery, cancellation, secret handling, and API access.

## Portfolio wording

“Built a supervised AutoML agent harness with interchangeable Claude/OpenAI
reasoning, validated tool decisions, recoverable checkpoints, execution budgets,
and fold-local preprocessing; added repeatable offline evaluations and CI.”

Add measured test totals or provider comparisons only after running them. Avoid
claims of production readiness, guaranteed savings, or improved accuracy without
evidence from the evaluation reports.
