"""Library usage, walked through end to end.

Run it:

    python examples/quickstart.py                       # churn, the default
    python examples/quickstart.py --dataset house       # regression
    python examples/quickstart.py --dataset sales       # time series
    python examples/quickstart.py --dry-run             # profile only, no cost

Needs a credential in the environment (``ANTHROPIC_API_KEY``,
``ANTHROPIC_AUTH_TOKEN``, or an OAuth profile). The ``--dry-run`` path makes no
model calls at all, so it works without one — a useful first check that ingestion
and profiling are wired up before spending anything.

The point of this file is not the API surface, which is three functions. It is
what you do with the result: every decision in a run carries the reasoning that
produced it, and the sections below print those out rather than only the score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

EXAMPLES = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES.parent

# Running a script puts *its own* directory on sys.path, not the repo root, so a
# fresh clone without `pip install -e .` cannot import the package. Add the root
# rather than failing with a ModuleNotFoundError that says nothing useful.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS: dict[str, tuple[str, str]] = {
    "churn": ("churn.csv", "churned"),
    "house": ("house_prices.csv", "sale_price"),
    "sales": ("sales_timeseries.csv", "units_sold"),
}


def rule(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _header(path: Path) -> list[str]:
    """Column names from a CSV's first line, without loading the file."""
    with path.open("r", encoding="utf-8") as handle:
        return [name.strip() for name in handle.readline().split(",")]


def _num(value: float | None, spec: str = ".4f", width: int = 0) -> str:
    """Format an optional number, rendering ``None`` as ``n/a``.

    Several score fields are legitimately ``None`` — the trainer records an
    experiment that fitted but whose primary metric was not computable, the tuner
    records ``ran=True`` with no scored trial, and a non-numeric target has no
    mean. Formatting those with a numeric spec raises ``TypeError``, so every
    optional number printed here goes through this helper. Mirrors the ``_fmt``
    convention in ``execution.trainer``.
    """
    text = "n/a" if value is None else f"{value:{spec}}"
    return f"{text:>{width}}" if width else text


# ---------------------------------------------------------------------------
# 1. Profile without spending anything
# ---------------------------------------------------------------------------


def show_profile(path: Path, target: str) -> None:
    """Measure the data and print the facts the agents will reason over.

    This is pure computation — pandas, numpy, scipy. No model is called, so it
    costs nothing and is the right way to sanity-check a new dataset.

    Args:
        path: CSV to profile.
        target: Target column name.
    """
    import pandas as pd

    from automl_architect.profiling.profiler import profile_dataframe

    frame = pd.read_csv(path)
    profile = profile_dataframe(frame, target=target)

    rule(f"MEASURED FACTS — {path.name}")
    print(f"shape           {profile.n_rows:,} rows x {profile.n_columns} columns")
    print(f"duplicates      {profile.n_duplicate_rows:,} ({profile.duplicate_fraction:.2%})")
    print(f"missing cells   {profile.total_missing_cells:,} ({profile.missing_cell_fraction:.2%})")

    if profile.target:
        summary = profile.target
        print(f"\ntarget `{summary.name}` [{summary.kind.value}]")
        if summary.class_counts:
            for count in summary.class_counts:
                print(f"    {count.value:>12}  {count.count:>7,}  ({count.fraction:.2%})")
            if summary.imbalance_ratio is not None:
                verdict = "IMBALANCED" if summary.is_imbalanced else "reasonably balanced"
                print(f"    imbalance ratio {summary.imbalance_ratio:.3f} -> {verdict}")
        else:
            print(
                f"    mean={_num(summary.mean, ',.2f')}  "
                f"std={_num(summary.std, ',.2f')}  "
                f"skew={_num(summary.skewness, '.3f')}"
            )

    # Leakage is the finding that matters most, because it invalidates everything
    # measured downstream of it if missed.
    if profile.leakage_findings:
        print("\nleakage candidates")
        for finding in profile.leakage_findings:
            print(
                f"    {finding.column:<28} score={finding.score:.4f} "
                f"[{finding.severity.value}] {finding.reason}"
            )
    else:
        print("\nleakage candidates: none")

    if profile.quality_issues:
        print("\nquality issues")
        for issue in profile.quality_issues[:8]:
            print(f"    [{issue.severity.value.upper():<8}] {issue.code}: {issue.detail}")

    skewed = sorted(
        (c for c in profile.columns if c.skewness is not None and abs(c.skewness) > 1.0),
        key=lambda c: abs(c.skewness or 0.0),
        reverse=True,
    )
    if skewed:
        print("\nmost skewed columns (this is what decides mean vs median imputation)")
        for column in skewed[:5]:
            print(
                f"    {column.name:<28} skew={column.skewness:>8.3f}  "
                f"missing={column.missing_fraction:.2%}"
            )


# ---------------------------------------------------------------------------
# 2. Run the pipeline
# ---------------------------------------------------------------------------


def run_pipeline(path: Path, target: str) -> Any:
    """Run the full pipeline and return the summary.

    Args:
        path: CSV to analyse.
        target: Column to predict.

    Returns:
        A ``(summary, architect)`` pair. The architect is returned too because
        :func:`ask_questions` needs the same instance — its repository is what
        holds the finished run.
    """
    from automl_architect import AutoMLArchitect

    rule("RUNNING")
    print("Thirteen agents reason; pandas and scikit-learn compute. This takes a few")
    print("minutes and makes real model calls.\n")

    architect = AutoMLArchitect()
    summary = architect.analyse(
        str(path),
        target=target,
        project="quickstart",
        time_budget_seconds=900,
        max_experiments=5,
        report_formats=["markdown", "html", "json"],
        # Slice metrics by region where the column exists, so the fairness audit
        # has something real to audit rather than an empty list.
        fairness_attributes=["region"] if "region" in _header(path) else [],
        notes="Quickstart example run.",
    )
    print(f"status: {summary.status.value}")
    if summary.error:
        print(f"error:  {summary.error}")
    return summary, architect


# ---------------------------------------------------------------------------
# 3. Read the reasoning, not just the score
# ---------------------------------------------------------------------------


def show_decisions(summary: Any) -> None:
    """Print each stage's decision alongside the reason recorded for it."""
    rule("THE PROBLEM, AS DIAGNOSED")
    problem = summary.problem
    if problem:
        print(f"task            {problem.task_type.value}  (confidence: {problem.confidence})")
        print(f"target          {problem.target_column}")
        print(f"primary metric  {problem.primary_metric}")
        print(f"\nwhy this task   {problem.rationale}")
        print(f"\nwhy this metric {problem.metric_rationale}")
        print(f"\nobjective       {problem.business_objective}")
        for alternative in problem.alternatives_considered:
            print(f"  considered:   {alternative}")

    rule("PLAN")
    plan = summary.plan
    if plan:
        print(f"{plan.summary}\n")
        for step in plan.ordered():
            flag = " [destructive]" if step.destructive else ""
            print(f"{step.order:>2}. {step.title}{flag}")
            print(f"    why: {step.rationale}")
        if plan.dataset_specific_adaptations:
            print("\nfitted to this dataset by:")
            for adaptation in plan.dataset_specific_adaptations:
                print(f"  - {adaptation}")

    rule("CLEANING — every transformation with its evidence")
    cleaning = summary.cleaning
    if cleaning:
        for decision in cleaning.decisions:
            columns = ", ".join(decision.columns) or "(whole table)"
            strategy = f" strategy={decision.strategy.value}" if decision.strategy else ""
            print(f"{decision.action.value}{strategy} on [{columns}]")
            print(f"    why: {decision.rationale}")
        # The transformations deliberately NOT applied are half the reasoning.
        if cleaning.skipped_considerations:
            print("\ndeliberately NOT done:")
            for skipped in cleaning.skipped_considerations:
                print(f"  - {skipped}")

    rule("FEATURES — each with a mechanism and a risk")
    features = summary.features
    if features:
        for decision in features.decisions:
            print(f"{decision.op.value} <- [{', '.join(decision.input_columns)}]")
            print(f"    why:        {decision.rationale}")
            if decision.hypothesis:
                print(f"    mechanism:  {decision.hypothesis}")
            if decision.risk:
                print(f"    risk:       {decision.risk}")

    rule("MODEL SELECTION")
    selection = summary.model_selection
    if selection:
        print(f"{selection.reasoning}\n")
        for candidate in sorted(selection.candidates, key=lambda c: c.rank):
            baseline = " (baseline)" if candidate.is_baseline else ""
            print(f"{candidate.rank}. {candidate.family.value}{baseline} — {candidate.suitability}")
            print(f"    why: {candidate.rationale}")
        for family, reason in zip(
            selection.excluded_families, selection.exclusion_rationale, strict=False
        ):
            print(f"excluded {family}: {reason}")
        print(f"\nvalidation: {selection.validation_strategy}")
        print(f"    why: {selection.validation_rationale}")


def show_results(summary: Any) -> None:
    """Print the leaderboard, the verdict, and the business translation."""
    rule("LEADERBOARD")
    log = summary.experiments
    if log:
        metric = log.primary_metric
        print(f"{'family':<28}{metric:>12}{'train_s':>10}   status")
        for result in sorted(
            log.results,
            key=lambda r: (r.primary_score is None, -(r.primary_score or 0.0)),
        ):
            marker = "  <- best" if result.experiment_id == log.best_experiment_id else ""
            if result.failed:
                print(f"{result.family.value:<28}{'FAILED':>12}{'':>10}   {result.error}")
                continue
            print(
                f"{result.family.value:<28}{_num(result.primary_score, '.4f', 12)}"
                f"{result.train_seconds:>10.2f}{marker}"
            )
        if log.leaderboard_notes:
            print(f"\n{log.leaderboard_notes}")

    tuning = summary.tuning
    if tuning:
        rule("TUNING")
        if tuning.ran:
            print(f"{tuning.method.value} on {tuning.family.value}: "
                  f"{tuning.n_trials_completed} trials, best={_num(tuning.best_score)}")
            print(f"improvement over baseline: {tuning.improvement}")
        else:
            print(f"skipped — {tuning.skipped_reason}")

    explain = summary.explainability
    if explain and explain.global_attributions:
        rule("WHAT THE MODEL LEARNED")
        for attribution in explain.global_attributions[:8]:
            print(
                f"    {attribution.feature:<34}{attribution.importance:>7.1%}  "
                f"{attribution.direction}"
            )
        print(f"\nmethod: {explain.method_notes}")
        for sentence in explain.plain_language_explanations[:4]:
            print(f"  - {sentence}")

    rule("VERDICT — the quality gate, not a formality")
    verdict = summary.evaluation
    if verdict:
        print(f"acceptable: {verdict.acceptable}   grade: {verdict.overall_grade}")
        print(f"action:     {verdict.recommended_action}")
        print(f"\n{verdict.verdict_rationale}\n")
        bias = verdict.bias_variance
        if bias.train_score is not None:
            print(
                f"fit:         train={bias.train_score:.4f}  "
                f"validation={bias.validation_score}  test={bias.test_score}  "
                f"gap={bias.gap}  -> {bias.verdict}"
            )
            print(f"             {bias.detail}")
        if verdict.calibration.applicable:
            print(f"calibration: {verdict.calibration.verdict}")
        print(f"drift risk:  {verdict.drift_risk} — {verdict.drift_rationale}")
        for weakness in verdict.weaknesses:
            print(f"weakness:    {weakness}")
        for slice_ in verdict.fairness_slices[:6]:
            print(
                f"fairness:    {slice_.attribute}={slice_.slice_value} "
                f"n={slice_.n_rows} {slice_.metric_name}={slice_.metric_value:.4f} "
                f"(delta {slice_.delta_vs_overall:+.4f})"
            )

    rule("WHAT THE BUSINESS SHOULD DO")
    insights = summary.insights
    if insights:
        print(f"{insights.executive_summary}\n")
        for insight in insights.insights:
            print(f"* {insight.headline}   [{insight.confidence} confidence]")
            print(f"    {insight.detail}")
            print(f"    evidence: {insight.supporting_evidence}")
            print(f"    action:   {insight.recommended_action}")
            if insight.expected_value:
                print(f"    value:    {insight.expected_value}")
        if insights.caveats:
            print("\ncaveats — read these before acting")
            for caveat in insights.caveats:
                print(f"  - {caveat}")

    rule("DEPLOYMENT RECOMMENDATION")
    report = summary.report
    if report:
        deployment = report.deployment
        print(f"pattern: {deployment.pattern.value}")
        print(f"    why: {deployment.rationale}")
        if deployment.estimated_latency_ms is not None:
            print(f"    measured latency: {deployment.estimated_latency_ms:.2f} ms/row")
        print(f"    retraining: {deployment.retraining_cadence}")
        for item in deployment.monitoring_plan:
            print(f"    monitor: {item}")


def show_accounting(summary: Any) -> None:
    """Print artifacts, warnings, and what the run actually cost."""
    rule("ARTIFACTS")
    bundle = summary.report_bundle
    if bundle:
        for label, path in (
            ("markdown", bundle.markdown_path),
            ("html", bundle.html_path),
            ("pdf", bundle.pdf_path),
            ("pptx", bundle.pptx_path),
            ("json", bundle.json_path),
        ):
            if path:
                print(f"    {label:<10} {path}")
        for warning in bundle.warnings:
            print(f"    warning: {warning}")
    print(f"    run dir    {summary.artifact_dir}")

    if summary.warnings:
        rule("WARNINGS — reduced capability, surfaced rather than swallowed")
        for warning in summary.warnings:
            print(f"  - {warning}")

    rule("COST")
    usage = summary.usage
    print(f"    llm calls          {usage.llm_calls}")
    print(f"    input tokens       {usage.input_tokens:,}")
    print(f"    output tokens      {usage.output_tokens:,}")
    print(f"    cache read tokens  {usage.cache_read_tokens:,}  (the frozen run context)")
    print(f"    estimated cost     ${usage.cost_usd:.4f}")
    print(f"    wall clock         {summary.duration_seconds:.1f}s")
    if summary.replans:
        print(f"    replans            {summary.replans}  (evaluation sent the run back)")


# ---------------------------------------------------------------------------
# 4. Ask about the run afterwards
# ---------------------------------------------------------------------------


def ask_questions(architect: Any, run_id: str) -> None:
    """Query the finished run in natural language.

    Answers are grounded in what the orchestrator recorded — events, metrics,
    decisions — rather than in recollection, and each carries an ``evidence`` list
    of the specific records it drew on.
    """
    rule("ASKING ABOUT THE RUN")
    questions = [
        "Why was this metric chosen over accuracy?",
        "Which single decision most affected the final score?",
        "What would you try next to improve it?",
    ]
    for question in questions:
        answer = architect.ask(run_id, question)
        print(f"\nQ: {question}")
        print(f"A: {answer.answer}")
        for evidence in answer.evidence[:3]:
            print(f"   evidence: {evidence}")
        if answer.caveats:
            print(f"   caveat: {answer.caveats[0]}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments and walk the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="churn",
        help="Which example dataset to use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Profile only. No model calls, no cost, no credential needed.",
    )
    parser.add_argument("--no-ask", action="store_true", help="Skip the Q&A section.")
    args = parser.parse_args()

    filename, target = DATASETS[args.dataset]
    path = EXAMPLES / filename
    if not path.exists():
        print(f"{path} is missing. Generate it with:\n"
              f"    python examples/generate_datasets.py")
        return 1

    show_profile(path, target)

    if args.dry_run:
        print("\n--dry-run: stopping before any model call.")
        return 0

    from automl_architect import get_settings

    if not get_settings().has_api_key():
        print(
            "\nNo ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN found. The SDK may still\n"
            "resolve an OAuth profile; if the run fails on credentials, set one in .env\n"
            "or re-run with --dry-run."
        )

    summary, architect = run_pipeline(path, target)

    show_decisions(summary)
    show_results(summary)
    show_accounting(summary)

    if not args.no_ask and architect.repository is not None:
        ask_questions(architect, summary.run_id)

    return 0 if summary.status.value == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
