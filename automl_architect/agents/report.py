"""Report Agent — the deliverable.

Everything upstream of here is machinery; this is the artifact a human keeps. Two
design commitments shape the module:

*   **The audit trail is the product.** Each cleaning and feature section must
    carry the rationale that was recorded at the time. A report that lists the
    transformations without the reasons is a changelog, and a reviewer cannot
    audit a changelog.
*   **The section set is fixed.** Nine headings, in order, always. Downstream
    renderers (markdown, HTML, PDF, PPTX) and human readers both rely on the
    shape being the same every time, so :meth:`ReportAgent.postprocess`
    synthesises any section the agent skipped out of recorded state rather than
    shipping a document with a hole in it.

The other guard here is a hard one: when the evaluation gate judged the model
unacceptable, no wording in this report is allowed to read as deployment
approval. The agent is told that, and the postprocessor enforces it.
"""

from __future__ import annotations

import re

from ..core.agent import BaseAgent
from ..core.llm import Effort
from ..core.schemas import (
    AgentName,
    FinalReport,
    ReportSection,
    Severity,
)
from ..core.state import RunState
from . import _deliver_support as support

#: The nine sections the spec names, in order. Canonical headings, verbatim.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "Executive Summary",
    "Dataset Overview",
    "Methodology",
    "Cleaning Decisions",
    "Feature Engineering",
    "Models Tested",
    "Evaluation",
    "Business Recommendations",
    "Deployment Suggestions",
)

#: Normalised substrings that identify a section the agent may have titled
#: differently ("1. Data Cleaning", "Model Evaluation & Diagnostics"). Longest
#: matching alias wins, so "Deployment Recommendations" binds to deployment
#: rather than to business recommendations.
_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "Executive Summary": ("executivesummary", "summary"),
    "Dataset Overview": (
        "datasetoverview",
        "dataoverview",
        "datasetdescription",
        "dataunderstanding",
        "thedata",
    ),
    "Methodology": ("methodology", "method", "approach", "pipeline"),
    "Cleaning Decisions": (
        "cleaningdecisions",
        "datacleaning",
        "cleaning",
        "preprocessing",
    ),
    "Feature Engineering": ("featureengineering", "features", "featureplan"),
    "Models Tested": (
        "modelstested",
        "modelstried",
        "modelcandidates",
        "modelselection",
        "modelling",
        "modeling",
        "leaderboard",
    ),
    "Evaluation": ("evaluation", "modelevaluation", "results", "diagnostics"),
    "Business Recommendations": (
        "businessrecommendations",
        "recommendations",
        "businessimpact",
        "businessinsights",
    ),
    "Deployment Suggestions": (
        "deploymentsuggestions",
        "deploymentrecommendations",
        "deployment",
        "productionisation",
        "productionization",
        "operationalisation",
    ),
}

#: Phrases that assert readiness. Checked only to warn: rewriting an agent's
#: prose would destroy the audit trail, so the guard adds a banner instead.
_READINESS_CLAIMS: tuple[str, ...] = (
    "ready for production",
    "production ready",
    "production-ready",
    "ready to deploy",
    "ready for deployment",
    "deployment ready",
    "deployment-ready",
    "fit for deployment",
    "safe to deploy",
    "recommend deploying",
    "recommend deployment",
    "approved for production",
    "cleared for production",
)

_UNACCEPTABLE_BANNER = (
    "**Not cleared for deployment.** The evaluation gate judged this model "
    "unacceptable (grade {grade}; recommended action: {action}). The guidance "
    "below describes what would fit *if* the recorded weaknesses are resolved; "
    "it is not an approval to ship.\n\n"
)


def _norm(heading: str) -> str:
    """Strip a heading to comparable letters and digits."""
    return re.sub(r"[^a-z0-9]+", "", heading.lower())


def match_required_section(heading: str) -> str | None:
    """Map an arbitrary heading onto one of :data:`REQUIRED_SECTIONS`.

    Args:
        heading: The heading the agent produced.

    Returns:
        The canonical heading, or ``None`` when the section is an extra one.
    """
    normalised = _norm(heading)
    best: tuple[int, str] | None = None
    for canonical, aliases in _SECTION_ALIASES.items():
        for alias in aliases:
            if alias in normalised and (best is None or len(alias) > best[0]):
                best = (len(alias), canonical)
    return best[1] if best else None


class ReportAgent(BaseAgent[FinalReport]):
    """Assembles the final report and the deployment recommendation."""

    name = AgentName.REPORT
    title = "Report Agent"
    output_model = FinalReport
    effort: Effort = "xhigh"
    max_tokens = 32_000

    # -- prompts -----------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        """The system prompt: document structure, tone discipline, deployment method."""
        return f"""\
## YOUR ROLE

You are the lead data scientist writing the deliverable for a completed modelling
engagement. Two audiences read it: an executive who will read the first page and
nothing else, and a data scientist who will audit every decision you made. The
document has to serve both without compromising for either.

## REQUIRED STRUCTURE

Produce exactly these nine sections, in this order, with these headings:

{chr(10).join(f"{i}. {name}" for i, name in enumerate(REQUIRED_SECTIONS, start=1))}

Set `order` 1-9 to match. No extra sections, no missing sections, no renaming.
Section bodies are markdown.

## USE REAL MARKDOWN TABLES

Three things read far better as tables than as prose, and you must render them as
tables:

*   **Models Tested** — the leaderboard: model, primary metric, cross-validated
    mean and spread, training time, prediction time, size.
*   **Cleaning Decisions** — the decision log: action, columns, strategy,
    rationale, expected impact. One row per decision.
*   **Feature Engineering** — the feature list: operation, inputs, output,
    rationale, leakage/overfitting risk.

Every row of the cleaning and feature tables must carry the rationale that was
recorded for that decision. Do not summarise the rationales away and do not
invent new ones: reproduce the recorded reasoning, tightened for readability. The
audit trail is the deliverable. A reviewer who disagrees with a choice must be
able to see the evidence it was made on.

## THE EXECUTIVE SUMMARY

Written for someone who will read nothing else. Lead with the outcome and the
decision it enables — what the model can now predict, how well, against what
baseline, and what that lets the business do differently. Methodology does not
appear in the first two sentences. Neither does a metric name without its
meaning: "identifies 3 in 4 of the accounts that will churn" beats "recall of
0.74" for this audience, though you may give the number alongside it. If the
model failed the evaluation gate, the summary says so plainly, in the first
sentence, and states what would have to change. A summary that reads like a
success story on a failing model destroys the credibility of everything else in
the document.

Use the `executive_summary` field for this, and open the Executive Summary
section with the same content — the field is what the API and slide renderers
pick up.

## SECTION-BY-SECTION EXPECTATIONS

*   **Dataset Overview** — what one row means, the shape, integrity (missingness,
    duplicates), the target's distribution, and the quality issues that mattered.
    Numbers only from the measured facts.
*   **Methodology** — the pipeline as executed, including the split strategy and
    why it was chosen (temporal, grouped, stratified) and the validation scheme.
    A reader must be able to judge whether the reported scores are trustworthy.
*   **Models Tested** — the leaderboard plus the comparative argument: why the
    winner won, what the baseline establishes, and what the failures tell you.
    Name families that failed to fit and why.
*   **Evaluation** — the verdict, the bias/variance reading, calibration,
    confidence intervals, fairness slices, and the weaknesses. Report the
    weaknesses as prominently as the wins.
*   **Business Recommendations** — the actions, each tied to the evidence and
    tagged with who owns it. Where a recommendation rests on a small slice or a
    `mixed`-direction feature, say so in the same sentence.
*   **Deployment Suggestions** — see below.

## THE DEPLOYMENT RECOMMENDATION

Ground the `deployment` object in the MEASURED numbers you were given, not in
generic architecture advice. The reasoning chain:

1.  **Latency and volume decide the pattern.** Measured per-row prediction time
    plus the required response time is the first branch. Predictions consumed
    daily or hourly by a downstream process, and per-row latency that is
    irrelevant at that cadence, argue for `batch_inference`. An interactive
    consumer needing a prediction inside a request argues for `rest_api`.
    Event-driven, continuous arrival argues for `streaming`.
2.  **Model size gates the serverless option.** A large artifact loaded per
    invocation makes cold starts dominate latency, so a big model plus a tight
    latency requirement argues AGAINST `serverless` — say that explicitly rather
    than defaulting to it. A small artifact with bursty, low-volume traffic is
    exactly what `serverless` suits.
3.  **A tiny model with high throughput can move to the consumer.** Very small
    artifacts and simple decision functions suit `edge` or `embedded_sql`, which
    removes a network hop and a service to operate. A linear or shallow-tree model
    of a few kilobytes is the case for it; a large ensemble is not.
4.  **Fill the numeric fields from the measurements.** `estimated_latency_ms` and
    `estimated_throughput_rps` come from the measured prediction time, and
    `model_size_mb` from the measured artifact size. If a figure was not
    measured, leave it null rather than guessing.

`monitoring_plan` must name the specific signals to watch for THIS dataset:
the columns whose drift would hurt most (the top attributions), the specific
distributions to track (the target's base rate, the class balance, the missing
rate of the columns that needed imputation), and the metric to recompute once
labels arrive. "Monitor for data drift" is not a monitoring plan; "alert if the
month-to-month share of `contract_type` moves more than 5 points from the 55%
training share" is.

`retraining_cadence` must be justified by the data's temporal span and the drift
risk recorded by the evaluator. A dataset spanning fourteen months of weekly
behaviour cannot support a claim about annual seasonality; say what the span does
and does not support, and set the cadence to something the span justifies.

`rollout_strategy` should say how to de-risk the first release (shadow mode,
holdback, staged traffic) and what would trigger a rollback. `risks` names what
could go wrong operationally, not model weaknesses already covered in Evaluation.

## HARD RULES

*   Never claim, imply, or hint at deployment readiness if the evaluation gate
    judged the model unacceptable. Frame the deployment section conditionally.
*   Never state a number that was not measured. Every figure in this document
    must be traceable to the facts you were given.
*   `chart_refs` on a section may reference only chart titles from the planned
    figure set you were given.
*   `appendix_notes` is for the caveats, the degradations recorded during the run,
    and the reproducibility details (random seed, library versions if given, row
    caps applied)."""

    def build_prompt(self, state: RunState) -> str:
        """The user turn: every recorded fact the document has to be built from."""
        sections = [
            "# THE COMPLETE RUN RECORD",
            "",
            "## Framing",
            support.run_facts_text(state),
            "",
            "## Problem definition",
            self._problem_text(state),
            "",
            "## Dataset understanding recorded at the start",
            self._understanding_text(state),
            "",
            "## Execution plan as run (with revisions)",
            support.plan_text(state.plan, state.plan_history),
            "",
            "## Data splitting",
            support.splits_text(state),
            "",
            "## Cleaning decisions and their recorded rationales",
            support.cleaning_text(state.cleaning),
            f"cleaning operations the executor actually applied: "
            f"{state.applied_cleaning or 'none recorded'}",
            f"columns dropped during the run: {state.dropped_columns or 'none'}",
            "",
            "## Feature engineering decisions and their recorded rationales",
            support.features_text(state.features),
            f"feature transformations the executor actually applied: "
            f"{state.applied_features or 'none recorded'}",
            f"final feature count: {len(state.feature_names)}",
            "",
            "## Model selection reasoning",
            self._selection_text(state),
            "",
            "## Leaderboard (measured)",
            support.leaderboard_text(state.experiments, limit=20),
            "",
            "## Winner versus baseline",
            support.winner_vs_baseline_text(state),
            "",
            "## Hyperparameter tuning",
            support.tuning_text(state.tuning_decision, state.tuning),
            "",
            "## Feature attributions",
            support.attributions_text(state.explainability, limit=20),
            "",
            "## Evaluation verdict",
            support.evaluation_text(state.evaluation),
            "",
            "## Business insights already produced",
            support.insights_text(state.insights),
            "",
            "## Target distribution",
            support.target_text(state.profile),
            "",
            "## MEASURED DEPLOYMENT FACTS (the basis for the deployment recommendation)",
            self._deployment_facts(state),
            "",
            "## Planned figures (the only titles `chart_refs` may cite)",
            support.charts_text(state.visualization_plan),
            "",
            "## Data-quality issues measured in the source",
            self._quality_text(state),
            "",
            "## Degradations and warnings recorded during the run",
            support.warnings_text(state, limit=20),
        ]

        bundles = support.executor_bundles_text(state, char_budget=4000)
        if bundles:
            sections += ["", "## Measured diagnostics", bundles]

        sections += [
            "",
            "# YOUR TASK",
            "",
            "Write the final report for this run: the nine required sections in "
            "order, with the leaderboard, cleaning log, and feature list as markdown "
            "tables carrying the recorded rationales, plus the deployment "
            "recommendation grounded in the measured latency, size, and temporal span "
            "above.",
            "",
            "Give the document a title and subtitle that name the dataset's domain "
            "and the outcome, not the toolchain.",
        ]
        return "\n".join(sections)

    # -- prompt fragments --------------------------------------------------

    @staticmethod
    def _problem_text(state: RunState) -> str:
        problem = state.problem
        if problem is None:
            return "(no problem definition was recorded)"
        return "\n".join(
            [
                f"task type: {problem.task_type.value} (confidence={problem.confidence})",
                f"task rationale: {support.clip(problem.rationale, 600)}",
                f"primary metric: {problem.primary_metric} — "
                f"{support.clip(problem.metric_rationale, 400)}",
                f"secondary metrics: {problem.secondary_metrics or 'none'}",
                f"alternatives considered: {problem.alternatives_considered or 'none'}",
                f"horizon: {problem.horizon if problem.horizon is not None else 'n/a'} | "
                f"temporal column: {problem.temporal_column or 'none'} | "
                f"group column: {problem.group_column or 'none'}",
            ]
        )

    @staticmethod
    def _understanding_text(state: RunState) -> str:
        understanding = state.understanding
        if understanding is None:
            return "(no dataset understanding was recorded)"
        return "\n".join(
            [
                f"headline: {support.clip(understanding.headline, 240)}",
                f"domain: {understanding.likely_domain} | grain: {understanding.grain}",
                f"readiness: {understanding.data_readiness} — "
                f"{support.clip(understanding.readiness_rationale, 300)}",
                "key findings:",
                support.bullets(understanding.key_findings, 10, "  - "),
                "risks:",
                support.bullets(understanding.risks, 8, "  - "),
                f"narrative: {support.clip(understanding.narrative, 1500)}",
            ]
        )

    @staticmethod
    def _selection_text(state: RunState) -> str:
        selection = state.model_selection
        if selection is None:
            return "(no model-selection record)"
        lines = [
            f"summary: {support.clip(selection.summary, 400)}",
            f"comparative reasoning: {support.clip(selection.reasoning, 900)}",
            f"validation strategy: {selection.validation_strategy} — "
            f"{support.clip(selection.validation_rationale, 300)}",
            "candidates:",
        ]
        for candidate in sorted(selection.candidates, key=lambda c: c.rank):
            lines.append(
                f"  {candidate.rank}. {candidate.family.value} "
                f"(suitability={candidate.suitability}, baseline={str(candidate.is_baseline).lower()}): "
                f"{support.clip(candidate.rationale, 260)}"
            )
        if selection.excluded_families:
            lines.append(f"excluded: {selection.excluded_families}")
            lines.append(support.bullets(selection.exclusion_rationale, 6, "  - "))
        return "\n".join(lines)

    @staticmethod
    def _quality_text(state: RunState) -> str:
        profile = state.profile
        if profile is None or not profile.quality_issues:
            return "(no data-quality issues were recorded)"
        rank = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        ordered = sorted(profile.quality_issues, key=lambda i: rank[i.severity])
        return "\n".join(
            f"- [{issue.severity.value.upper()}] {issue.code}: "
            f"{support.clip(issue.detail, 240)}"
            f"{f' (columns={issue.columns})' if issue.columns else ''}"
            for issue in ordered[:20]
        )

    @staticmethod
    def _deployment_facts(state: RunState) -> str:
        """Measured serving costs, derived arithmetically — never estimated."""
        winner = support.best_experiment(state)
        if winner is None:
            return (
                "(no model was trained, so there are no measured serving costs; "
                "do not invent latency or size figures)"
            )
        sizes = state.splits.sizes()
        scored_rows = sizes["test"] or sizes["validation"] or sizes["train"]
        lines = [
            f"winning model: {winner.family.value}"
            f"{' (tuned)' if winner.tuned else ''}",
            f"measured artifact size: {support.mb(winner.model_size_bytes)} "
            f"({winner.model_size_bytes:,} bytes)",
            f"measured training time: {support.num(winner.train_seconds, 3)}s",
            f"measured prediction time: {support.num(winner.predict_seconds, 3)}s "
            f"for {scored_rows:,} rows",
            f"measured peak training memory: {support.num(winner.peak_memory_mb, 3)} MB",
            f"features at inference time: {winner.n_features_in}",
        ]
        if winner.predict_seconds > 0 and scored_rows > 0:
            per_row_ms = winner.predict_seconds / scored_rows * 1000
            lines.append(
                f"derived per-row latency: {per_row_ms:.4f} ms/row "
                f"(batch-mode throughput ≈ {scored_rows / winner.predict_seconds:,.0f} rows/s "
                "on the machine that ran this experiment; a single-row request pays "
                "per-call overhead on top of this)"
            )
        else:
            lines.append(
                "derived per-row latency: not computable from the recorded timings"
            )
        lines.append(
            f"dataset volume: {state.profile.n_rows:,} rows x "
            f"{state.profile.n_columns} columns"
            if state.profile
            else "dataset volume: not profiled"
        )
        lines.append("temporal span of the data (basis for retraining cadence):")
        lines.append(support.temporal_span_text(state.profile))
        if state.evaluation:
            lines.append(
                f"drift risk assessed by the evaluator: {state.evaluation.drift_risk} — "
                f"{support.clip(state.evaluation.drift_rationale, 300)}"
            )
            lines.append(
                f"model acceptable for deployment: "
                f"{str(state.evaluation.acceptable).lower()}"
            )
        if state.explainability and state.explainability.global_attributions:
            top = [a.feature for a in state.explainability.global_attributions[:8]]
            lines.append(f"columns whose drift would matter most: {top}")
        imputed = [
            c
            for d in (state.cleaning.decisions if state.cleaning else [])
            for c in d.columns
            if d.strategy is not None
        ]
        if imputed:
            lines.append(
                f"columns that required imputation (watch their missing rate): "
                f"{sorted(set(imputed))[:12]}"
            )
        return "\n".join(lines)

    # -- grounding ---------------------------------------------------------

    def postprocess(self, value: FinalReport, state: RunState) -> FinalReport:
        """Enforce the nine-section contract and the deployment-readiness guard.

        Steps, in order: bind each returned section to a canonical heading;
        synthesise from recorded state anything missing or empty; order the
        canonical nine first and any extras after; renumber densely from 1; drop
        chart references that name no planned figure; and, if evaluation failed,
        prefix the summary and the deployment rationale with an explicit
        not-cleared notice.

        Args:
            value: The report as returned by the model.
            state: The run blackboard, the source for any synthesised section.

        Returns:
            The repaired report.
        """
        matched: dict[str, ReportSection] = {}
        extras: list[ReportSection] = []

        for section in sorted(value.sections, key=lambda s: s.order):
            canonical = match_required_section(section.heading)
            if canonical and canonical not in matched:
                section.heading = canonical
                matched[canonical] = section
            else:
                extras.append(section)

        # The `executive_summary` field — not the section — is what the HTML,
        # PDF, PPTX, and markdown renderers and the API digest read, so a blank
        # one ships an empty first page. Backfill it from the section the agent
        # did write, else from the measured outcome.
        if not value.executive_summary.strip():
            summary_section = matched.get("Executive Summary")
            body = summary_section.body_markdown.strip() if summary_section else ""
            value.executive_summary = body or self._fallback_summary(state)
            state.add_warning(
                f"{self.title} left `executive_summary` empty; every renderer reads "
                "that field directly, so it was backfilled from "
                f"{'the Executive Summary section' if body else 'the measured run outcome'}."
            )

        synthesised: list[str] = []
        for canonical in REQUIRED_SECTIONS:
            section = matched.get(canonical)
            if section is not None and section.body_markdown.strip():
                continue
            body = self._synthesise_section(canonical, state, value)
            if section is None:
                matched[canonical] = ReportSection(
                    heading=canonical, order=0, body_markdown=body
                )
            else:
                section.body_markdown = body
            synthesised.append(canonical)

        # Offline runs route section assembly through these same builders by
        # design, so synthesis there is the expected path rather than a gap the
        # agent left. Warning about it every run would train the reader to ignore
        # the warning list, which is where real problems appear.
        if synthesised and not getattr(state.settings, "offline", False):
            state.add_warning(
                f"{self.title} omitted or left empty {len(synthesised)} required "
                f"section(s) {synthesised}; they were synthesised from the recorded "
                "run state so the report has no gaps."
            )

        ordered = [matched[name] for name in REQUIRED_SECTIONS] + extras
        for index, section in enumerate(ordered, start=1):
            section.order = index
            section.chart_refs = self._filter_chart_refs(section.chart_refs, state)
        value.sections = ordered

        self._enforce_readiness_guard(value, state)
        return value

    @staticmethod
    def _filter_chart_refs(refs: list[str], state: RunState) -> list[str]:
        """Keep only references that name a planned chart title or kind."""
        plan = state.visualization_plan
        if plan is None or not refs:
            return refs
        known = {c.title.strip().lower() for c in plan.charts}
        known |= {c.kind.value for c in plan.charts}
        return [r for r in refs if r.strip().lower() in known or r in known]

    def _enforce_readiness_guard(self, value: FinalReport, state: RunState) -> None:
        """Make an unacceptable verdict impossible to read as an approval."""
        verdict = state.evaluation
        if verdict is None or verdict.acceptable:
            return

        banner = _UNACCEPTABLE_BANNER.format(
            grade=verdict.overall_grade, action=verdict.recommended_action
        )
        if "not cleared for deployment" not in value.executive_summary.lower():
            value.executive_summary = banner + value.executive_summary

        summary_section = next(
            (s for s in value.sections if s.heading == "Executive Summary"), None
        )
        if summary_section and "not cleared for deployment" not in summary_section.body_markdown.lower():
            summary_section.body_markdown = banner + summary_section.body_markdown

        deployment = value.deployment
        if not deployment.rationale.lower().startswith("[conditional"):
            deployment.rationale = (
                "[Conditional — the model did not pass evaluation and must not be "
                f"deployed as-is] {deployment.rationale}"
            )
        deployment.risks.append(
            f"The evaluation gate judged this model unacceptable (grade "
            f"{verdict.overall_grade}; recommended action: "
            f"{verdict.recommended_action}). Deploying it in any pattern would ship "
            "known-inadequate predictions; resolve the recorded weaknesses first."
        )

        haystacks = [value.executive_summary.lower(), deployment.rationale.lower()]
        haystacks += [s.body_markdown.lower() for s in value.sections]
        claimed = sorted(
            {phrase for phrase in _READINESS_CLAIMS for h in haystacks if phrase in h}
        )
        # Readiness language surviving the banner is still worth surfacing: the
        # reviewer should know the prose and the verdict disagree.
        if claimed:
            state.add_warning(
                f"{self.title} used readiness language {claimed} while the "
                "evaluation verdict was unacceptable; a not-cleared notice was "
                "prepended to the summary and the deployment rationale."
            )

    # -- section synthesis -------------------------------------------------

    def _synthesise_section(
        self, canonical: str, state: RunState, value: FinalReport
    ) -> str:
        """Build a required section's body from recorded state.

        Args:
            canonical: One of :data:`REQUIRED_SECTIONS`.
            state: The run blackboard.
            value: The report being repaired, for cross-referencing.

        Returns:
            Markdown for the section, prefixed with a provenance note so a reader
            knows the agent did not write it.
        """
        builders = {
            "Executive Summary": lambda: value.executive_summary
            or self._fallback_summary(state),
            "Dataset Overview": lambda: self._dataset_overview(state),
            "Methodology": lambda: self._methodology(state),
            "Cleaning Decisions": lambda: support.cleaning_markdown(
                state.cleaning, state.applied_cleaning
            ),
            "Feature Engineering": lambda: support.features_markdown(
                state.features, state.applied_features
            ),
            "Models Tested": lambda: support.leaderboard_markdown(state.experiments),
            "Evaluation": lambda: self._evaluation_markdown(state),
            "Business Recommendations": lambda: self._recommendations_markdown(state),
            "Deployment Suggestions": lambda: self._deployment_markdown(value),
        }
        body = builders[canonical]()
        note = (
            "_Assembled deterministically from the recorded run state (offline "
            "mode: no model calls were made)._\n\n"
            if getattr(state.settings, "offline", False)
            else "_This section was assembled by the orchestrator from the recorded "
            "run state because the reporting agent did not supply it._\n\n"
        )
        return note + body

    @staticmethod
    def _fallback_summary(state: RunState) -> str:
        winner = support.best_experiment(state)
        if winner is None:
            return (
                "No model completed training in this run, so there is no result to "
                "summarise. See the Evaluation section for what was recorded."
            )
        verdict = state.evaluation
        status = (
            "passed the evaluation gate"
            if verdict and verdict.acceptable
            else "did NOT pass the evaluation gate"
        )
        return (
            f"The best model was `{winner.family.value}`, scoring "
            f"{support.num(winner.primary_score)} on {state.primary_metric}. It "
            f"{status}"
            f"{f' (grade {verdict.overall_grade})' if verdict else ''}.\n\n"
            f"{support.winner_vs_baseline_text(state)}"
        )

    @staticmethod
    def _dataset_overview(state: RunState) -> str:
        profile = state.profile
        if profile is None:
            return "_No dataset profile was recorded for this run._"
        lines = [
            f"- **Rows:** {profile.n_rows:,}",
            f"- **Columns:** {profile.n_columns:,}",
            f"- **Memory:** {support.mb(profile.memory_bytes)}",
            f"- **Duplicate rows:** {profile.n_duplicate_rows:,} "
            f"({support.pct(profile.duplicate_fraction)})",
            f"- **Missing cells:** {profile.total_missing_cells:,} "
            f"({support.pct(profile.missing_cell_fraction)} of all cells)",
            f"- **Temporal columns:** {profile.temporal_columns or 'none detected'}",
            f"- **Identifier-like columns:** {profile.identifier_columns or 'none'}",
            f"- **Free-text columns:** {profile.text_columns or 'none'}",
            f"- **Constant columns:** {profile.constant_columns or 'none'}",
        ]
        if state.understanding:
            lines = [
                f"{state.understanding.headline}",
                "",
                f"One row represents: {state.understanding.grain}. "
                f"Domain: {state.understanding.likely_domain}.",
                "",
                *lines,
            ]
        target = "\n\n**Target**\n\n" + support.target_text(profile)
        issues = ""
        if profile.quality_issues:
            rows = [
                [i.severity.value.upper(), i.code, ", ".join(i.columns) or "—", i.detail]
                for i in profile.quality_issues[:15]
            ]
            issues = "\n\n**Data-quality issues**\n\n" + support.md_table(
                ["Severity", "Code", "Columns", "Detail"], rows
            )
        return "\n".join(lines) + target + issues

    @staticmethod
    def _methodology(state: RunState) -> str:
        parts: list[str] = []
        if state.plan:
            parts.append(support.clip(state.plan.summary, 1200))
            rows = [
                [str(step.order), step.agent.value, step.title, step.rationale]
                for step in state.plan.ordered()
            ]
            table = support.md_table(["#", "Agent", "Step", "Why it was needed"], rows)
            if table:
                parts += ["", table]
            if state.plan.dataset_specific_adaptations:
                parts += [
                    "",
                    "**Adaptations made for this dataset**",
                    "",
                    support.bullets(state.plan.dataset_specific_adaptations, 12),
                ]
        parts += ["", "**Data splitting and validation**", "", support.splits_text(state)]
        if state.problem:
            parts += [
                "",
                f"**Primary metric:** `{state.problem.primary_metric}` — "
                f"{support.clip(state.problem.metric_rationale, 500)}",
            ]
        return "\n".join(parts) if parts else "_No methodology was recorded._"

    @staticmethod
    def _evaluation_markdown(state: RunState) -> str:
        verdict = state.evaluation
        if verdict is None:
            return (
                "_Evaluation did not run for this model, so its quality is "
                "unverified. Treat every score in this report as provisional._"
            )
        parts = [
            f"**Verdict:** grade {verdict.overall_grade}; "
            f"{'acceptable' if verdict.acceptable else 'NOT acceptable'} for "
            f"deployment. Recommended action: `{verdict.recommended_action}`.",
            "",
            support.clip(verdict.verdict_rationale, 1500),
        ]
        bv = verdict.bias_variance
        parts += [
            "",
            "**Fit diagnosis**",
            "",
            f"- Verdict: {bv.verdict}",
            f"- Train: {support.num(bv.train_score)} | "
            f"Validation: {support.num(bv.validation_score)} | "
            f"Test: {support.num(bv.test_score)} | Gap: {support.num(bv.gap)}",
        ]
        if bv.detail:
            parts.append(f"- {support.clip(bv.detail, 400)}")
        if verdict.confidence_intervals:
            rows = [
                [
                    ci.metric,
                    support.num(ci.point_estimate),
                    f"[{support.num(ci.lower)}, {support.num(ci.upper)}]",
                    support.pct(ci.level, 0),
                    ci.method,
                ]
                for ci in verdict.confidence_intervals
            ]
            parts += [
                "",
                "**Confidence intervals**",
                "",
                support.md_table(
                    ["Metric", "Estimate", "Interval", "Level", "Method"], rows
                ),
            ]
        if verdict.fairness_slices:
            rows = [
                [
                    s.attribute,
                    s.slice_value,
                    f"{s.n_rows:,}",
                    s.metric_name,
                    support.num(s.metric_value),
                    support.num(s.delta_vs_overall),
                ]
                for s in verdict.fairness_slices
            ]
            parts += [
                "",
                "**Fairness slices**",
                "",
                support.md_table(
                    ["Attribute", "Slice", "Rows", "Metric", "Value", "vs overall"], rows
                ),
            ]
        parts += ["", "**Weaknesses**", "", support.bullets(verdict.weaknesses, 12)]
        if verdict.specific_improvements:
            parts += [
                "",
                "**Improvements identified**",
                "",
                support.bullets(verdict.specific_improvements, 10),
            ]
        return "\n".join(parts)

    @staticmethod
    def _recommendations_markdown(state: RunState) -> str:
        insights = state.insights
        if insights is None or not insights.insights:
            return (
                "_No business insights were recorded. The measured drivers are in "
                "the Evaluation section; no recommendation can be made without "
                "them being interpreted._"
            )
        parts = [support.clip(insights.executive_summary, 1200), ""]
        for insight in insights.insights:
            parts += [
                f"### {insight.headline}",
                "",
                support.clip(insight.detail, 800),
                "",
                f"- **Evidence:** {support.clip(insight.supporting_evidence, 400)}",
                f"- **Action:** {support.clip(insight.recommended_action, 400)}",
                f"- **Owner:** {insight.audience}",
                f"- **Confidence:** {insight.confidence}",
            ]
            if insight.expected_value:
                parts.append(f"- **Expected value:** {support.clip(insight.expected_value, 300)}")
            parts.append("")
        if insights.caveats:
            parts += ["**Caveats**", "", support.bullets(insights.caveats, 12)]
        return "\n".join(parts)

    @staticmethod
    def _deployment_markdown(value: FinalReport) -> str:
        deployment = value.deployment
        parts = [
            f"**Recommended pattern:** `{deployment.pattern.value}`",
            "",
            support.clip(deployment.rationale, 1200),
            "",
            f"- Estimated latency: {support.num(deployment.estimated_latency_ms)} ms",
            f"- Estimated throughput: "
            f"{support.num(deployment.estimated_throughput_rps)} req/s",
            f"- Model size: {support.num(deployment.model_size_mb)} MB",
            f"- Retraining cadence: {deployment.retraining_cadence or 'not specified'}",
        ]
        if deployment.infrastructure_notes:
            parts += ["", support.clip(deployment.infrastructure_notes, 800)]
        if deployment.monitoring_plan:
            parts += ["", "**Monitoring plan**", "", support.bullets(deployment.monitoring_plan, 15)]
        if deployment.rollout_strategy:
            parts += ["", f"**Rollout:** {support.clip(deployment.rollout_strategy, 600)}"]
        if deployment.risks:
            parts += ["", "**Risks**", "", support.bullets(deployment.risks, 12)]
        return "\n".join(parts)

    # -- state -------------------------------------------------------------

    def apply(self, state: RunState, value: FinalReport) -> None:
        """Record the final report on the blackboard."""
        state.report = value

    def decision_summary(self, value: FinalReport) -> str:
        """One line for the event stream."""
        return (
            f"report '{support.clip(value.title, 80)}' with "
            f"{len(value.sections)} section(s); deployment pattern "
            f"{value.deployment.pattern.value}; "
            f"{len(value.deployment.monitoring_plan)} monitoring signal(s)"
        )


__all__ = ["REQUIRED_SECTIONS", "ReportAgent", "match_required_section"]
