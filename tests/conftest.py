"""Shared test fixtures, and the offline test harness this suite is built on.

Three things live here, in order of importance.

**1. :class:`FakeLLMClient`.** The whole pipeline — including the orchestrator —
must be testable without network access, credentials, or nondeterminism. The fake
implements the same call surface as ``core.llm.LLMClient`` (``structured``,
``text``, ``count_tokens``, a ``usage`` accumulator) and resolves each
``structured`` call by *output model type*: a test registers a canned instance
for, say, ``ProblemDefinition`` and every agent that asks for one gets it. Calls
are recorded, so a test can assert on the prompt an agent built without knowing
anything about that agent's internals.

**2. Schema autofill.** Registering 13 canned agent outputs by hand for every
test would make the suite brittle to schema edits. :func:`synthesise` walks a
Pydantic model's fields and constructs a minimal *schema-valid* instance of
anything, so an unregistered output model still yields a usable value. Realistic
canned responses (:data:`CANNED_BUILDERS`) take precedence where they exist.

**3. Sibling-module tolerance.** Modules of this project are written in parallel,
so a test file must be able to say "skip me if the module I exercise does not
exist yet" rather than collapsing collection. :func:`import_or_skip` and
:func:`attr_or_skip` do that, and they accept several candidate paths because a
sibling's exact module name is a guess until it lands.

Network access is blocked for the whole session by :func:`_offline_env`, which
points the Anthropic SDK at a dead local port. A test that genuinely needs the
API must be marked ``@pytest.mark.live`` and skip itself without credentials.
"""

from __future__ import annotations

import datetime as _dt
import enum
import importlib
import json
import os
import types
import typing
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import pandas as pd
import pytest
from pydantic import BaseModel

from automl_architect import config as config_module
from automl_architect.config import Settings, estimate_cost_usd, reset_settings_cache
from automl_architect.core import llm as llm_module
from automl_architect.core.context import build_run_context
from automl_architect.core.events import EventBus
from automl_architect.core.llm import LLMResult, PromptBlock, StructuredResult, Usage
from automl_architect.core.schemas import (
    AgentName,
    BiasVarianceDiagnosis,
    BusinessInsight,
    CalibrationDiagnosis,
    CategoryCount,
    ChartKind,
    ChartSpec,
    CleaningAction,
    CleaningDecision,
    CleaningPlan,
    ColumnAssessment,
    ColumnKind,
    ColumnProfile,
    ColumnRole,
    DataSource,
    DatasetProfile,
    DatasetUnderstanding,
    DeploymentPattern,
    DeploymentRecommendation,
    EvaluationVerdict,
    ExecutionPlan,
    ExplainabilityReport,
    FeatureAttribution,
    FeatureDecision,
    FeatureOp,
    FeaturePlan,
    FinalReport,
    InsightReport,
    LeakageFinding,
    MissingStrategy,
    ModelCandidate,
    ModelFamily,
    ModelSelection,
    PlanStep,
    ProblemDefinition,
    Quantiles,
    ReportSection,
    RunConfig,
    SearchSpaceEntry,
    Severity,
    SourceKind,
    TargetSummary,
    TaskType,
    TuningDecision,
    TuningMethod,
    VisualizationPlan,
    dict_to_params,
)
from automl_architect.core.state import RunState

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

CHURN_CSV = EXAMPLES_DIR / "churn.csv"
HOUSE_CSV = EXAMPLES_DIR / "house_prices.csv"
SALES_CSV = EXAMPLES_DIR / "sales_timeseries.csv"

CHURN_TARGET = "churned"
CHURN_LEAK = "cancellation_tickets"
HOUSE_TARGET = "sale_price"
SALES_TARGET = "units_sold"


# ===========================================================================
# Sibling-module tolerance
# ===========================================================================


def try_import(*candidates: str) -> types.ModuleType | None:
    """Import the first importable module from ``candidates``, else ``None``.

    Args:
        *candidates: Dotted module paths, most likely first.

    Returns:
        The imported module, or ``None`` if none of them import.
    """
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def import_or_skip(*candidates: str, feature: str = "") -> types.ModuleType:
    """Import the first available candidate module, or skip the test.

    Args:
        *candidates: Dotted module paths to try in order.
        feature: Human name of the capability, used in the skip message.

    Returns:
        The imported module.

    Raises:
        pytest.skip.Exception: If no candidate imports, which in this project
            means the sibling module has not been written yet.
    """
    module = try_import(*candidates)
    if module is None:
        label = feature or candidates[0]
        pytest.skip(f"{label} not available yet (tried: {', '.join(candidates)})")
    return module


def attr_or_skip(module: types.ModuleType, *names: str) -> Any:
    """Fetch the first present attribute from ``module``, or skip the test."""
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    pytest.skip(f"{module.__name__} exposes none of {names}")


# ===========================================================================
# Schema autofill
# ===========================================================================

_MAX_DEPTH = 6


def _placeholder_text(field_name: str) -> str:
    """Readable filler that still reads as a sentence in a rendered report."""
    words = field_name.replace("_", " ")
    return f"synthetic {words} generated by the test harness"


def _value_for(annotation: Any, field_name: str, depth: int) -> Any:
    """Build one schema-valid value for a type annotation.

    Args:
        annotation: The field's type annotation.
        field_name: Used to make string placeholders self-describing.
        depth: Recursion guard so a pathological schema cannot hang the suite.

    Returns:
        A value that will pass Pydantic validation for ``annotation``.
    """
    if depth > _MAX_DEPTH:
        return None

    origin = get_origin(annotation)

    if origin is Literal:
        return get_args(annotation)[0]

    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) < len(get_args(annotation)):
            return None  # Optional: null is always valid
        return _value_for(args[0], field_name, depth + 1)

    if origin in (list, set, tuple, frozenset):
        args = get_args(annotation)
        if not args:
            return []
        # Exactly one element: enough to exercise list handling downstream
        # without inflating an autofilled object into something unreadable.
        return [_value_for(args[0], field_name, depth + 1)]

    if origin is dict:
        return {}

    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            return next(iter(annotation))
        if issubclass(annotation, BaseModel):
            return synthesise(annotation, depth=depth + 1)
        if issubclass(annotation, bool):
            return False
        if issubclass(annotation, int):
            return 1
        if issubclass(annotation, float):
            return 1.0
        if issubclass(annotation, str):
            return _placeholder_text(field_name)
        if issubclass(annotation, _dt.datetime):
            return _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)

    return None


def synthesise(model: type[BaseModel], *, depth: int = 0, **overrides: Any) -> Any:
    """Construct a minimal schema-valid instance of any Pydantic model.

    Only *required* fields are populated; anything with a default keeps it. This
    is what lets the suite survive schema growth: a new optional field needs no
    test change, and a new required field is filled automatically.

    Args:
        model: The model class to instantiate.
        depth: Internal recursion depth.
        **overrides: Field values to force, applied after autofill.

    Returns:
        A validated instance of ``model``.
    """
    payload: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if not field.is_required():
            continue
        payload[name] = _value_for(field.annotation, name, depth)
    payload.update(overrides)
    return model.model_validate(payload)


# ===========================================================================
# Realistic canned agent responses
# ===========================================================================


def churn_understanding() -> DatasetUnderstanding:
    """A Dataset Agent output shaped like one a real run produces on churn.csv."""
    return DatasetUnderstanding(
        headline="A 3,000-row telecom subscriber table with a 26% churn label.",
        narrative=(
            "The grain is one subscriber. Sixteen columns mix contract terms, billing "
            "amounts, service usage, and a binary churn outcome. Two things dominate "
            "the read. First, annual_income is log-normally distributed with 8.6% "
            "missing, so mean imputation would pull imputed subscribers toward a value "
            "no real subscriber holds. Second, cancellation_tickets separates the "
            "target almost perfectly, which is not a finding about churn but a finding "
            "about the table: cancellation tickets are filed after a subscriber "
            "decides to leave, so the column cannot exist at prediction time."
        ),
        likely_domain="telecom subscriber churn",
        grain="one subscriber",
        column_assessments=[
            ColumnAssessment(
                name="customer_id",
                role=ColumnRole.IDENTIFIER,
                predictive_potential="none",
                concerns=["3,000 distinct values across 3,000 rows"],
                notes="A surrogate key. Carries no signal and would be memorised by a tree.",
            ),
            ColumnAssessment(
                name=CHURN_LEAK,
                role=ColumnRole.LEAKAGE_SUSPECT,
                predictive_potential="high",
                concerns=["near-perfect target separation", "post-outcome timing"],
                notes=(
                    "Association with the target is 0.99 by AUC against 0.63 for the "
                    "next-best feature. Recorded after the churn decision; must be dropped."
                ),
            ),
            ColumnAssessment(
                name="tenure_months",
                role=ColumnRole.FEATURE,
                predictive_potential="high",
                concerns=[],
                notes="Longer tenure tracks with retention; the strongest legitimate signal here.",
            ),
            ColumnAssessment(
                name="annual_income",
                role=ColumnRole.FEATURE,
                predictive_potential="low",
                concerns=["8.6% missing", "heavy right skew"],
                notes="Weakly associated with churn but worth keeping with median imputation.",
            ),
            ColumnAssessment(
                name=CHURN_TARGET,
                role=ColumnRole.TARGET,
                predictive_potential="none",
                concerns=[],
                notes="The label: 1 = churned. 26% positive.",
            ),
        ],
        key_findings=[
            "Churn rate is 26%, imbalanced enough that accuracy will mislead.",
            "cancellation_tickets leaks the outcome and must be removed before training.",
            "total_charges correlates 0.81 with tenure_months, so both carry the same signal.",
        ],
        risks=[
            "crypto_wallet appears in 0.5% of rows; one-hot encoding it risks a near-constant column.",
            "Dropping the leakage column will lower headline scores, correctly.",
        ],
        suggested_target_columns=[CHURN_TARGET],
        data_readiness="needs_cleaning",
        readiness_rationale=(
            "One leakage column to drop and one skewed column to impute; no structural problems."
        ),
    )


def churn_problem() -> ProblemDefinition:
    """A Problem Agent output for churn.csv."""
    return ProblemDefinition(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column=CHURN_TARGET,
        positive_class="1",
        temporal_column=None,
        group_column=None,
        horizon=None,
        rationale=(
            "churned takes exactly two integer values with 26% positive, so this is "
            "binary classification on an imbalanced target."
        ),
        alternatives_considered=[
            "Survival analysis, which the data cannot support: there is no observed "
            "time-to-event column, only a static tenure snapshot."
        ],
        confidence="high",
        primary_metric="roc_auc",
        secondary_metrics=["average_precision", "f1", "recall"],
        metric_rationale=(
            "At 26% positive, accuracy is beaten by predicting the majority class. "
            "ROC AUC is threshold-free and average precision tracks the minority class."
        ),
        business_objective=(
            "Rank subscribers by churn risk so retention offers go to the ~750 accounts "
            "most likely to leave."
        ),
        constraints=["Scores must be interpretable enough to justify a retention offer."],
    )


def churn_plan() -> ExecutionPlan:
    """A Planner output whose steps name real columns from churn.csv."""
    steps = [
        ("clean_data", 1, "Remove leakage and impute income", AgentName.CLEANING, True),
        ("engineer_features", 2, "Encode categoricals, add charge ratios", AgentName.FEATURES, False),
        ("select_models", 3, "Choose candidate families", AgentName.MODEL_SELECTION, False),
        ("run_experiments", 4, "Train and score every candidate", AgentName.EXPERIMENT, False),
        ("tune", 5, "Tune the leading family if worthwhile", AgentName.TUNING, False),
        ("explain", 6, "Attribute the winning model", AgentName.EXPLAIN, False),
        ("evaluate", 7, "Judge fitness for deployment", AgentName.EVALUATION, False),
        ("insights", 8, "Translate results for the business", AgentName.INSIGHT, False),
        ("visualise", 9, "Render the charts that carry the argument", AgentName.VISUALIZATION, False),
        ("report", 10, "Assemble the final report", AgentName.REPORT, False),
    ]
    return ExecutionPlan(
        summary=(
            "Drop the leakage column first so every later measurement is honest, then "
            "impute the skewed income column with its median, encode the five "
            "categoricals, and compare a dummy baseline against linear and tree models "
            "under stratified 5-fold CV on ROC AUC."
        ),
        steps=[
            PlanStep(
                step_id=step_id,
                order=order,
                title=title,
                agent=agent,
                objective=title,
                rationale=f"{title} is required for this dataset because of its measured properties.",
                depends_on=[steps[i - 1][0]] if order > 1 else [],
                optional=False,
                destructive=destructive,
                estimated_seconds=30,
                success_criteria="The step completes without reducing available signal.",
            )
            for i, (step_id, order, title, agent, destructive) in enumerate(steps)
        ],
        dataset_specific_adaptations=[
            "Leakage removal is step one rather than a later hygiene pass, because every "
            "downstream score computed with cancellation_tickets present would be void.",
            "Stratified folds, not plain K-fold: at 26% positive a random fold can shift "
            "the class ratio enough to move AUC.",
        ],
        risks=["Removing the leak will drop apparent AUC from ~0.99 to ~0.70."],
        fallback_strategy=(
            "If AUC lands under 0.65, revisit feature engineering for interaction terms "
            "between contract_type and tenure_months before adding model families."
        ),
        revision=0,
        revision_reason=None,
    )


def churn_cleaning() -> CleaningPlan:
    """A Cleaning Agent output: drop the leak, median-impute the skewed column."""
    return CleaningPlan(
        decisions=[
            CleaningDecision(
                action=CleaningAction.DROP_LEAKAGE_COLUMN,
                columns=[CHURN_LEAK],
                strategy=None,
                parameters=[],
                rationale=(
                    "AUC of 0.9975 against the target versus 0.627 for the next-best "
                    "feature, and the column records an event that happens after churn."
                ),
                expected_impact="Headline AUC falls to a truthful level.",
                destructive=True,
                severity_if_skipped=Severity.CRITICAL,
            ),
            CleaningDecision(
                action=CleaningAction.IMPUTE_MISSING,
                columns=["annual_income"],
                strategy=MissingStrategy.MEDIAN,
                parameters=[],
                rationale=(
                    "8.6% missing on a log-normal column (skewness 2.43): the mean sits "
                    "at the 63rd percentile, so mean imputation would invent atypically "
                    "wealthy subscribers while the median lands in the bulk."
                ),
                expected_impact="Retains 259 rows that row-dropping would discard.",
                destructive=False,
                severity_if_skipped=Severity.MEDIUM,
            ),
            CleaningDecision(
                action=CleaningAction.DROP_COLUMN,
                columns=["customer_id"],
                strategy=None,
                parameters=[],
                rationale="3,000 unique values over 3,000 rows: a surrogate key with no signal.",
                expected_impact="Removes a column a tree would otherwise memorise.",
                destructive=True,
                severity_if_skipped=Severity.HIGH,
            ),
        ],
        summary="One leakage drop, one identifier drop, one median imputation.",
        columns_to_drop=[CHURN_LEAK, "customer_id"],
        drop_rationale=["post-outcome leakage", "surrogate key"],
        skipped_considerations=[
            "Outlier clipping on total_charges: the tail is real billing history, not error.",
            "Dropping crypto_wallet rows: 15 rows is not worth losing a payment channel.",
        ],
    )


def churn_features() -> FeaturePlan:
    """A Feature Agent output using only columns that exist post-cleaning."""
    return FeaturePlan(
        decisions=[
            FeatureDecision(
                op=FeatureOp.ONE_HOT_ENCODE,
                input_columns=["contract_type", "internet_service", "region"],
                output_name_hint="",
                parameters=dict_to_params({"handle_unknown": "infrequent_if_exist", "min_frequency": 0.01}),
                rationale="Three nominal columns at 3-4 levels each: 10 dummy columns total.",
                hypothesis="Month-to-month contracts churn at a different base rate.",
                risk="None; cardinality is far too low to inflate dimensionality.",
                priority="high",
            ),
            FeatureDecision(
                op=FeatureOp.FREQUENCY_ENCODE,
                input_columns=["payment_method"],
                output_name_hint="payment_method_freq",
                parameters=[],
                rationale=(
                    "crypto_wallet holds 0.5% of rows; one-hot would produce a "
                    "near-constant column, so frequency encoding preserves the level."
                ),
                hypothesis="Rare payment channels correlate with lower account commitment.",
                risk="Collapses distinct levels that share a frequency; acceptable at 5 levels.",
                priority="medium",
            ),
            FeatureDecision(
                op=FeatureOp.RATIO,
                input_columns=["total_charges", "tenure_months"],
                output_name_hint="avg_charge_per_month",
                parameters=[],
                rationale=(
                    "total_charges correlates 0.81 with tenure_months; the ratio isolates "
                    "spend level from account age."
                ),
                hypothesis="Rising spend per month of tenure signals plan dissatisfaction.",
                risk="Division by zero at tenure 0; the minimum observed tenure is 1.",
                priority="high",
            ),
            FeatureDecision(
                op=FeatureOp.LOG_TRANSFORM,
                input_columns=["annual_income"],
                output_name_hint="annual_income_log",
                parameters=[],
                rationale="Log-normal by construction; the log makes it near-Gaussian for linear models.",
                hypothesis="Income effects on churn are multiplicative, not additive.",
                risk="None: all values are strictly positive.",
                priority="low",
            ),
            FeatureDecision(
                op=FeatureOp.STANDARD_SCALE,
                input_columns=["tenure_months", "monthly_charges", "total_charges", "satisfaction_score"],
                output_name_hint="",
                parameters=[],
                rationale="Logistic regression and SVM need comparable scales; trees ignore it.",
                hypothesis="",
                risk="",
                priority="medium",
            ),
        ],
        summary="Low-cardinality one-hot, frequency encoding for the rare payment level, two derived ratios, and scaling.",
        expected_feature_count_delta=12,
        dimensionality_strategy="No reduction needed: ~25 features against 3,000 rows.",
        selection_strategy="Keep everything; report permutation importance instead of pre-filtering.",
    )


def churn_model_selection() -> ModelSelection:
    """A Model Selection Agent output ranked for a 3k-row tabular problem."""
    return ModelSelection(
        candidates=[
            ModelCandidate(
                family=ModelFamily.BASELINE_DUMMY,
                rank=4,
                suitability="poor",
                rationale="Not a contender; it fixes the floor every other score is judged against.",
                expected_strengths=["Establishes that 74% accuracy is worthless here"],
                expected_weaknesses=["AUC 0.5 by construction"],
                initial_params=dict_to_params({"strategy": "prior"}),
                is_baseline=True,
                tune_priority="none",
            ),
            ModelCandidate(
                family=ModelFamily.LOGISTIC,
                rank=3,
                suitability="good",
                rationale="Linear, calibrated, and directly interpretable as odds per feature.",
                expected_strengths=["Coefficients survive a compliance review"],
                expected_weaknesses=["Misses the tenure-by-contract interaction unless given it"],
                initial_params=dict_to_params({"max_iter": 2000, "class_weight": "balanced"}),
                is_baseline=False,
                tune_priority="low",
            ),
            ModelCandidate(
                family=ModelFamily.RANDOM_FOREST,
                rank=2,
                suitability="good",
                rationale="Handles the mixed dtypes and interactions with almost no tuning.",
                expected_strengths=["Robust defaults", "Native feature importance"],
                expected_weaknesses=["Probabilities need calibration before use as risk scores"],
                initial_params=dict_to_params({"n_estimators": 300, "class_weight": "balanced_subsample"}),
                is_baseline=False,
                tune_priority="medium",
            ),
            ModelCandidate(
                family=ModelFamily.HIST_GRADIENT_BOOSTING,
                rank=1,
                suitability="excellent",
                rationale=(
                    "3,000 rows and ~25 features is where histogram boosting reliably wins "
                    "on tabular data, and it needs no imputation of its own."
                ),
                expected_strengths=["Best expected AUC", "Fast on this size"],
                expected_weaknesses=["Can overfit 3k rows without a depth cap"],
                initial_params=dict_to_params({"max_iter": 300, "learning_rate": 0.08}),
                is_baseline=False,
                tune_priority="high",
            ),
        ],
        summary="A dummy floor, a linear reference, and two tree ensembles.",
        reasoning=(
            "The comparison that matters is boosting against logistic regression: if "
            "boosting wins by less than a couple of AUC points, the interpretable model "
            "is the better product for a retention team that must justify each offer. "
            "Random forest is included as a variance check on the boosted result rather "
            "than because it is expected to win."
        ),
        excluded_families=["neural_network", "knn"],
        exclusion_rationale=[
            "3,000 rows is far too few to train an MLP that beats boosting.",
            "KNN degrades on the one-hot columns' sparse geometry.",
        ],
        validation_strategy="stratified 5-fold cross-validation",
        validation_rationale="At 26% positive, unstratified folds move AUC by more than the model differences.",
    )


def churn_tuning_decision() -> TuningDecision:
    """A Tuning Agent output that says yes, with a bounded budget."""
    return TuningDecision(
        worthwhile=True,
        rationale=(
            "Boosting leads logistic regression by 0.04 AUC on defaults, which is small "
            "enough that tuning could change the ranking. 3,000 rows means 30 trials cost "
            "under a minute."
        ),
        method=TuningMethod.RANDOM_SEARCH,
        method_rationale="Random search covers a 4-parameter space adequately without Optuna's overhead.",
        target_family=ModelFamily.HIST_GRADIENT_BOOSTING,
        n_trials=12,
        timeout_seconds=60,
        early_stopping=True,
        search_space=[
            SearchSpaceEntry(
                name="learning_rate",
                kind="log_float",
                low=0.01,
                high=0.3,
                rationale="Log scale: the useful range spans an order of magnitude.",
            ),
            SearchSpaceEntry(
                name="max_leaf_nodes",
                kind="int",
                low=8,
                high=63,
                rationale="Capped below the sklearn default of 31x2 because 3,000 rows overfit quickly.",
            ),
            SearchSpaceEntry(
                name="min_samples_leaf",
                kind="int",
                low=10,
                high=80,
                rationale="A floor of 10 keeps leaves above 0.3% of the training rows.",
            ),
            SearchSpaceEntry(
                name="l2_regularization",
                kind="log_float",
                low=1e-3,
                high=10.0,
                rationale="Regularisation is the main defence against the small row count.",
            ),
        ],
        expected_gain="0.01-0.02 AUC; enough to settle the ranking, not enough to change the recommendation.",
    )


def explainability_narration() -> ExplainabilityReport:
    """An Explainability Agent narration over pre-computed attributions."""
    return ExplainabilityReport(
        global_attributions=[
            FeatureAttribution(feature="tenure_months", importance=0.31, direction="decreases", method="shap"),
            FeatureAttribution(feature="contract_type_month_to_month", importance=0.24, direction="increases", method="shap"),
            FeatureAttribution(feature="satisfaction_score", importance=0.18, direction="decreases", method="shap"),
            FeatureAttribution(feature="avg_charge_per_month", importance=0.14, direction="increases", method="shap"),
            FeatureAttribution(feature="support_tickets", importance=0.13, direction="increases", method="shap"),
        ],
        permutation_importance=[
            FeatureAttribution(feature="tenure_months", importance=0.34, direction="decreases", method="permutation"),
            FeatureAttribution(feature="contract_type_month_to_month", importance=0.22, direction="increases", method="permutation"),
        ],
        shap_available=True,
        shap_summary_path=None,
        partial_dependence_paths=[],
        counterfactuals=[],
        plain_language_explanations=[
            "Account age accounts for roughly 31% of the model's churn signal, and more of it lowers risk.",
            "Being on a month-to-month contract is the second-largest driver and raises risk.",
            "Support ticket volume matters, but a third as much as contract type.",
        ],
        narrative=(
            "The model has learned the commitment story rather than a billing story: "
            "tenure and contract type together carry 55% of the attribution, while every "
            "charge-related feature combined carries 14%."
        ),
        method_notes="TreeSHAP on the full test split; permutation importance over 10 repeats as a cross-check.",
    )


def evaluation_verdict(*, acceptable: bool = True) -> EvaluationVerdict:
    """An Evaluation Agent verdict. Set ``acceptable=False`` to trigger a replan."""
    return EvaluationVerdict(
        acceptable=acceptable,
        verdict_rationale=(
            "Test AUC 0.712 against a 0.500 dummy floor, with a train/test gap of 0.03."
            if acceptable
            else "Test AUC 0.548 is barely above the dummy floor; the features do not carry the outcome."
        ),
        overall_grade="B" if acceptable else "D",
        bias_variance=BiasVarianceDiagnosis(
            train_score=0.742 if acceptable else 0.951,
            validation_score=0.718 if acceptable else 0.561,
            test_score=0.712 if acceptable else 0.548,
            gap=0.030 if acceptable else 0.403,
            verdict="good_fit" if acceptable else "overfitting",
            detail=(
                "A 0.03 gap on 3,000 rows is within noise for this fold count."
                if acceptable
                else "A 0.40 gap means the model memorised the training split."
            ),
        ),
        calibration=CalibrationDiagnosis(
            applicable=True,
            brier_score=0.164,
            expected_calibration_error=0.041,
            verdict="Usable as a ranking; apply isotonic calibration before quoting probabilities.",
        ),
        generalisation_notes="Stratified folds; no group structure to violate.",
        drift_risk="medium",
        drift_rationale="Contract mix shifts with pricing changes, so the dominant feature is not stationary.",
        fairness_slices=[],
        fairness_notes="region was audited; the largest AUC gap between regions is 0.03.",
        confidence_intervals=[],
        residual_notes="",
        error_analysis=[
            "False negatives concentrate in two-year contracts with high satisfaction — genuinely surprising churners.",
        ],
        learning_curve_notes="The validation curve is still rising at 2,400 rows: more data would help.",
        weaknesses=["Probabilities are uncalibrated.", "No temporal holdout, so drift is untested."],
        recommended_action="accept" if acceptable else "retry_feature_engineering",
        action_rationale=(
            "Good enough to rank a retention list, which is the stated objective."
            if acceptable
            else "Interaction features between contract_type and tenure are untried and cheap."
        ),
        specific_improvements=["Fit an isotonic calibrator on the validation split."],
    )


def insight_report() -> InsightReport:
    """A Business Insight Agent output."""
    return InsightReport(
        executive_summary=(
            "Churn is a commitment problem, not a pricing problem. Contract type and "
            "account age carry 55% of the model's signal; every billing feature combined "
            "carries 14%."
        ),
        insights=[
            BusinessInsight(
                headline="Month-to-month subscribers are the entire churn problem.",
                detail=(
                    "They are 57% of the base and the single largest positive driver of "
                    "predicted risk, second only to short tenure."
                ),
                supporting_evidence="contract_type_month_to_month holds 24% of SHAP attribution.",
                recommended_action="Offer a one-year term at a discount to the top risk decile.",
                expected_value="Roughly 190 accounts in the top decile; a 15% save rate retains ~28 per quarter.",
                confidence="medium",
                audience="executive",
            ),
            BusinessInsight(
                headline="Support ticket volume is a weak early-warning signal, not a cause.",
                detail="It carries 13% of attribution, a third of contract type's weight.",
                supporting_evidence="Permutation importance ranks it fifth of 25 features.",
                recommended_action="Use ticket spikes to time an outreach, not to justify a discount.",
                expected_value="",
                confidence="medium",
                audience="operations",
            ),
        ],
        key_drivers_plain_language=[
            "How long someone has been a subscriber (31% of the signal).",
            "Whether they are on a rolling monthly contract (24%).",
            "Their satisfaction score (18%).",
        ],
        caveats=[
            "AUC 0.71 ranks well but does not predict individuals reliably.",
            "Cross-sectional data cannot establish that contracts cause retention.",
        ],
        suggested_next_experiments=[
            "A holdout test of the discount offer on the top risk decile.",
            "Add month-over-month usage deltas, which this snapshot does not contain.",
        ],
    )


def visualization_plan() -> VisualizationPlan:
    """A Visualization Agent output covering the argument the report makes."""
    return VisualizationPlan(
        charts=[
            ChartSpec(
                kind=ChartKind.CLASS_BALANCE,
                title="Churn class balance",
                columns=[CHURN_TARGET],
                rationale="Shows why accuracy was rejected as the primary metric.",
                parameters=[],
                priority="high",
            ),
            ChartSpec(
                kind=ChartKind.ROC_CURVE,
                title="ROC curve, winning model vs baseline",
                columns=[],
                rationale="The headline metric, shown against the dummy floor.",
                parameters=[],
                priority="high",
            ),
            ChartSpec(
                kind=ChartKind.FEATURE_IMPORTANCE,
                title="Feature attribution",
                columns=[],
                rationale="Backs the claim that churn is a commitment problem.",
                parameters=[],
                priority="high",
            ),
            ChartSpec(
                kind=ChartKind.CONFUSION_MATRIX,
                title="Confusion matrix at the 0.5 threshold",
                columns=[],
                rationale="Makes the false-negative cost concrete for the retention team.",
                parameters=[],
                priority="medium",
            ),
            ChartSpec(
                kind=ChartKind.HISTOGRAM,
                title="Annual income distribution",
                columns=["annual_income"],
                rationale="Justifies median over mean imputation visually.",
                parameters=[],
                priority="low",
            ),
        ],
        dashboard_narrative=(
            "Read left to right: the class balance explains the metric choice, the ROC "
            "curve gives the result, and the attribution chart explains it."
        ),
    )


def final_report() -> FinalReport:
    """A Report Agent output."""
    return FinalReport(
        title="Subscriber churn: model and recommendation",
        subtitle="3,000 subscribers, 26% churn, ROC AUC 0.712",
        executive_summary=(
            "A histogram gradient boosting model ranks subscribers by churn risk at "
            "ROC AUC 0.712 on a held-out test split. One column, cancellation_tickets, "
            "was removed before training because it records an event that happens after "
            "churn; leaving it in would have produced a meaningless AUC of 0.99."
        ),
        sections=[
            ReportSection(
                heading="What the data is",
                order=1,
                body_markdown="3,000 subscribers, 16 columns, one row per subscriber.",
                chart_refs=["class_balance"],
            ),
            ReportSection(
                heading="Decisions and why",
                order=2,
                body_markdown=(
                    "- Dropped `cancellation_tickets` (AUC 0.9975 vs the target; post-outcome).\n"
                    "- Median-imputed `annual_income` (log-normal, 8.6% missing).\n"
                    "- Chose ROC AUC over accuracy (26% positive class)."
                ),
                chart_refs=[],
            ),
            ReportSection(
                heading="Results",
                order=3,
                body_markdown="Test AUC 0.712 against a 0.500 dummy floor.",
                chart_refs=["roc_curve", "confusion_matrix"],
            ),
            ReportSection(
                heading="Limitations",
                order=4,
                body_markdown="Uncalibrated probabilities; no temporal holdout.",
                chart_refs=[],
            ),
        ],
        deployment=DeploymentRecommendation(
            pattern=DeploymentPattern.BATCH_INFERENCE,
            rationale=(
                "Retention offers go out on a weekly cycle, and the model scores 3,000 "
                "rows in 40ms, so nothing about this problem needs a live endpoint."
            ),
            estimated_latency_ms=0.4,
            estimated_throughput_rps=25_000.0,
            model_size_mb=1.8,
            infrastructure_notes="A weekly scheduled job writing scores to the CRM table.",
            monitoring_plan=[
                "Alert if the month-to-month share of the base moves more than 5 points.",
                "Recompute AUC monthly against realised churn.",
            ],
            retraining_cadence="quarterly, or on a monitoring alert",
            rollout_strategy="Score-only for one cycle, compare against the current heuristic list, then switch.",
            risks=["The dominant feature is non-stationary under pricing changes."],
        ),
        appendix_notes=["Full decision log in run_summary.json."],
    )


#: Realistic canned output per agent output model. Anything absent falls through
#: to :func:`synthesise`, so a new agent type never breaks the suite.
CANNED_BUILDERS: dict[type[BaseModel], Callable[[], BaseModel]] = {
    DatasetUnderstanding: churn_understanding,
    ProblemDefinition: churn_problem,
    ExecutionPlan: churn_plan,
    CleaningPlan: churn_cleaning,
    FeaturePlan: churn_features,
    ModelSelection: churn_model_selection,
    TuningDecision: churn_tuning_decision,
    ExplainabilityReport: explainability_narration,
    EvaluationVerdict: evaluation_verdict,
    InsightReport: insight_report,
    VisualizationPlan: visualization_plan,
    FinalReport: final_report,
}


# ===========================================================================
# The fake LLM
# ===========================================================================


class FakeCall:
    """One recorded ``structured`` or ``text`` invocation."""

    def __init__(
        self,
        *,
        output_model: type[BaseModel] | None,
        user: str,
        system: list[PromptBlock],
        effort: str | None,
        max_tokens: int | None,
    ) -> None:
        self.output_model = output_model
        self.user = user
        self.system = system
        self.effort = effort
        self.max_tokens = max_tokens

    @property
    def system_text(self) -> str:
        """All system blocks concatenated, for substring assertions."""
        return "\n\n".join(block.text for block in self.system)

    @property
    def prompt_text(self) -> str:
        """System blocks plus the user turn: everything the model saw."""
        return f"{self.system_text}\n\n{self.user}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        name = self.output_model.__name__ if self.output_model else "text"
        return f"<FakeCall {name} effort={self.effort} user_chars={len(self.user)}>"


class FakeLLMClient:
    """An offline stand-in for :class:`automl_architect.core.llm.LLMClient`.

    Resolution order for a ``structured`` call, first match wins:

    1. an exception queued with :meth:`fail_next`;
    2. a response queued for that exact ``output_model`` via :meth:`register`;
    3. a realistic canned builder from :data:`CANNED_BUILDERS`;
    4. :func:`synthesise`, which autofills a schema-valid instance.

    Usage accounting is deterministic — token counts derive from prompt and
    response length — so a test can assert that a run recorded cost without
    asserting a magic number that a prompt edit would invalidate.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        use_canned: bool = True,
    ) -> None:
        self.settings = settings or config_module.get_settings()
        self.usage = Usage()
        self.calls: list[FakeCall] = []
        self.use_canned = use_canned
        self._queued: dict[type[BaseModel], list[Any]] = {}
        self._sticky: dict[type[BaseModel], Any] = {}
        self._text_queue: list[str] = []
        self._failures: list[Exception] = []
        self._seen_prefixes: set[str] = set()

    # -- registration -----------------------------------------------------

    def register(
        self,
        output_model: type[BaseModel],
        value: Any,
        *,
        sticky: bool = False,
    ) -> FakeLLMClient:
        """Queue a response for one output model type.

        Args:
            output_model: The model class an agent will request.
            value: An instance, a dict, or a callable ``(output_model, user,
                system) -> instance | dict``.
            sticky: If true, reuse this response for every call to that type
                instead of consuming it once.

        Returns:
            ``self``, so registrations can be chained.
        """
        if sticky:
            self._sticky[output_model] = value
        else:
            self._queued.setdefault(output_model, []).append(value)
        return self

    def register_text(self, *responses: str) -> FakeLLMClient:
        """Queue plain-text responses for :meth:`text` calls."""
        self._text_queue.extend(responses)
        return self

    def fail_next(self, error: Exception) -> FakeLLMClient:
        """Make the next ``structured`` call raise ``error``."""
        self._failures.append(error)
        return self

    # -- internals --------------------------------------------------------

    def _resolve(
        self, output_model: type[BaseModel], user: str, system: list[PromptBlock]
    ) -> BaseModel:
        raw: Any
        queue = self._queued.get(output_model)
        if queue:
            raw = queue.pop(0)
        elif output_model in self._sticky:
            raw = self._sticky[output_model]
        elif self.use_canned and output_model in CANNED_BUILDERS:
            raw = CANNED_BUILDERS[output_model]()
        else:
            raw = synthesise(output_model)

        if callable(raw) and not isinstance(raw, BaseModel):
            raw = raw(output_model, user, system)
        if isinstance(raw, output_model):
            return raw
        if isinstance(raw, BaseModel):
            return output_model.model_validate(raw.model_dump())
        return output_model.model_validate(raw)

    def _account(self, system: list[PromptBlock], user: str, rendered: str) -> Usage:
        """Deterministic token/cost accounting, including a cache-hit simulation."""
        cached_text = "".join(b.text for b in system if b.cache)
        fresh_text = "".join(b.text for b in system if not b.cache) + user

        prefix_key = cached_text[:512]
        cached_tokens = len(cached_text) // 4
        if cached_text and prefix_key in self._seen_prefixes:
            cache_read, cache_write = cached_tokens, 0
        elif cached_text:
            cache_read, cache_write = 0, cached_tokens
            self._seen_prefixes.add(prefix_key)
        else:
            cache_read = cache_write = 0

        usage = Usage(
            input_tokens=max(1, len(fresh_text) // 4),
            output_tokens=max(1, len(rendered) // 4),
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            calls=1,
        )
        usage.cost_usd = estimate_cost_usd(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
        )
        self.usage.merge(usage)
        return usage

    @staticmethod
    def _system_blocks(
        system: list[PromptBlock] | None,
        agent_instructions: str | None,
        run_context: str | None,
    ) -> list[PromptBlock]:
        if system is not None:
            return system
        if agent_instructions is None:
            raise ValueError("provide either `system` or `agent_instructions`")
        return llm_module.build_system(
            run_context=run_context, agent_instructions=agent_instructions
        )

    # -- LLMClient surface ------------------------------------------------

    def structured(
        self,
        *,
        output_model: type[BaseModel],
        user: str,
        system: list[PromptBlock] | None = None,
        agent_instructions: str | None = None,
        run_context: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        max_attempts: int = 3,
    ) -> StructuredResult:
        """Return a validated instance of ``output_model`` without any network I/O."""
        blocks = self._system_blocks(system, agent_instructions, run_context)
        self.calls.append(
            FakeCall(
                output_model=output_model,
                user=user,
                system=blocks,
                effort=effort,
                max_tokens=max_tokens,
            )
        )
        if self._failures:
            raise self._failures.pop(0)

        value = self._resolve(output_model, user, blocks)
        rendered = json.dumps(value.model_dump(mode="json"), default=str)
        usage = self._account(blocks, user, rendered)

        return StructuredResult(
            value=value,
            text="",
            thinking=f"[fake thinking for {output_model.__name__}]",
            usage=usage,
            stop_reason="end_turn",
            model="fake-opus",
            seconds=0.0,
            attempts=1,
            fallback_used=False,
        )

    def text(
        self,
        *,
        user: str,
        system: list[PromptBlock] | None = None,
        agent_instructions: str | None = None,
        run_context: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Return queued prose, or a deterministic echo if nothing is queued."""
        blocks = self._system_blocks(system, agent_instructions, run_context)
        self.calls.append(
            FakeCall(
                output_model=None,
                user=user,
                system=blocks,
                effort=effort,
                max_tokens=max_tokens,
            )
        )
        if self._failures:
            raise self._failures.pop(0)

        body = self._text_queue.pop(0) if self._text_queue else f"[fake prose for {user[:60]}]"
        usage = self._account(blocks, user, body)
        return LLMResult(
            text=body,
            thinking="",
            usage=usage,
            stop_reason="end_turn",
            model="fake-opus",
            seconds=0.0,
        )

    def count_tokens(self, *, system: list[PromptBlock], user: str) -> int:
        """Length-based token estimate. Deterministic, never a network call."""
        total = sum(len(b.text) for b in system) + len(user)
        return max(1, total // 4)

    # -- assertions helpers ------------------------------------------------

    def calls_for(self, output_model: type[BaseModel]) -> list[FakeCall]:
        """Every recorded call that requested ``output_model``."""
        return [c for c in self.calls if c.output_model is output_model]

    def last_call_for(self, output_model: type[BaseModel]) -> FakeCall:
        """The most recent call for ``output_model``; fails the test if absent."""
        matches = self.calls_for(output_model)
        assert matches, f"no call recorded for {output_model.__name__}"
        return matches[-1]

    @property
    def n_calls(self) -> int:
        return len(self.calls)


# ===========================================================================
# Session-wide offline guard
# ===========================================================================


@pytest.fixture(autouse=True)
def _offline_env(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate every test from the network, the real workspace, and the SDK cache.

    Points the Anthropic SDK at TCP port 9 (discard) so an accidental live call
    fails with a connection error in milliseconds rather than reaching the API,
    and redirects the workspace to a temp directory so runs never write into the
    developer's checkout.
    """
    workspace = tmp_path_factory.mktemp("automl_ws")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-not-real")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("AUTOML_WORKSPACE", str(workspace))
    monkeypatch.setenv("AUTOML_DATABASE_URL", f"sqlite+pysqlite:///{(workspace / 'test.db').as_posix()}")
    monkeypatch.delenv("AUTOML_MLFLOW_TRACKING_URI", raising=False)
    reset_settings_cache()
    llm_module.reset_llm_client()
    yield
    reset_settings_cache()
    llm_module.reset_llm_client()


@pytest.fixture
def settings() -> Settings:
    """The settings singleton, already redirected at the temp workspace."""
    return config_module.get_settings()


@pytest.fixture
def fake_llm(settings: Settings) -> FakeLLMClient:
    """A fresh fake client, not yet installed as the process-wide singleton."""
    return FakeLLMClient(settings=settings)


@pytest.fixture
def installed_fake_llm(fake_llm: FakeLLMClient, monkeypatch: pytest.MonkeyPatch) -> FakeLLMClient:
    """The fake, installed so ``get_llm_client()`` returns it everywhere.

    Agents and orchestrators that construct their own client via
    ``get_llm_client()`` pick this up without needing dependency injection.
    """
    monkeypatch.setattr(llm_module, "_shared", fake_llm, raising=False)
    return fake_llm


# ===========================================================================
# Data fixtures
# ===========================================================================


def _load_or_generate(path: Path) -> pd.DataFrame:
    """Read an example CSV, generating the example set first if it is absent."""
    if not path.exists():  # pragma: no cover - only on a fresh checkout
        import sys

        sys.path.insert(0, str(EXAMPLES_DIR))
        from generate_datasets import write_all  # type: ignore[import-not-found]

        write_all(EXAMPLES_DIR)
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def churn_df() -> pd.DataFrame:
    """The churn example: imbalanced binary target, mixed types, one real leak."""
    return _load_or_generate(CHURN_CSV)


@pytest.fixture(scope="session")
def house_df() -> pd.DataFrame:
    """The regression example: skewed target, collinear pair, outliers."""
    return _load_or_generate(HOUSE_CSV)


@pytest.fixture(scope="session")
def sales_df() -> pd.DataFrame:
    """The panel time-series example: trend, weekly seasonality, two gaps."""
    return _load_or_generate(SALES_CSV)


@pytest.fixture
def churn_csv() -> Path:
    """Path to churn.csv, generating the example set if needed."""
    _load_or_generate(CHURN_CSV)
    return CHURN_CSV


@pytest.fixture
def small_churn_df(churn_df: pd.DataFrame) -> pd.DataFrame:
    """A 400-row stratified-ish slice, for tests where fitting speed matters."""
    positives = churn_df[churn_df[CHURN_TARGET] == 1].head(104)
    negatives = churn_df[churn_df[CHURN_TARGET] == 0].head(296)
    return pd.concat([positives, negatives]).sample(frac=1.0, random_state=0).reset_index(drop=True)


# ===========================================================================
# Profile / state fixtures
# ===========================================================================


def stub_profile(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    dataset_id: str = "ds_test",
    leakage: list[str] | None = None,
) -> DatasetProfile:
    """Build a :class:`DatasetProfile` from a frame with plain pandas.

    Deliberately independent of ``profiling/profiler.py``: tests for agents,
    context rendering, and the orchestrator need *a* valid profile, and they
    should not fail because the real profiler is mid-rewrite. The statistics are
    real, just not exhaustive.

    Args:
        frame: The dataframe to describe.
        target: Target column name, if any.
        dataset_id: Identifier to stamp on the profile.
        leakage: Columns to record as leakage findings.

    Returns:
        A populated, schema-valid profile.
    """
    columns: list[ColumnProfile] = []
    for name in frame.columns:
        series = frame[name]
        n_unique = int(series.nunique(dropna=True))
        n_missing = int(series.isna().sum())
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        looks_id = n_unique == len(frame) and len(frame) > 1

        if pd.api.types.is_bool_dtype(series):
            kind = ColumnKind.BOOLEAN
        elif looks_id and not is_numeric:
            kind = ColumnKind.IDENTIFIER
        elif is_numeric:
            kind = (
                ColumnKind.NUMERIC_DISCRETE
                if pd.api.types.is_integer_dtype(series) and n_unique <= 25
                else ColumnKind.NUMERIC_CONTINUOUS
            )
        else:
            kind = ColumnKind.CATEGORICAL_NOMINAL

        profile = ColumnProfile(
            name=str(name),
            kind=kind,
            dtype=str(series.dtype),
            n_missing=n_missing,
            missing_fraction=n_missing / max(1, len(frame)),
            n_unique=n_unique,
            cardinality_ratio=n_unique / max(1, len(frame)),
            is_constant=n_unique <= 1,
            memory_bytes=int(series.memory_usage(deep=True)),
            looks_like_id=looks_id,
        )
        if is_numeric:
            numeric = series.dropna().astype("float64")
            if len(numeric):
                profile.mean = float(numeric.mean())
                profile.std = float(numeric.std())
                profile.minimum = float(numeric.min())
                profile.maximum = float(numeric.max())
                profile.skewness = float(numeric.skew())
                profile.variance = float(numeric.var())
                profile.quantiles = Quantiles(
                    p01=float(numeric.quantile(0.01)),
                    p05=float(numeric.quantile(0.05)),
                    p25=float(numeric.quantile(0.25)),
                    p50=float(numeric.quantile(0.50)),
                    p75=float(numeric.quantile(0.75)),
                    p95=float(numeric.quantile(0.95)),
                    p99=float(numeric.quantile(0.99)),
                )
        elif kind is ColumnKind.CATEGORICAL_NOMINAL:
            counts = series.value_counts(dropna=True).head(8)
            profile.top_values = [
                CategoryCount(value=str(v), count=int(c), fraction=float(c) / max(1, len(frame)))
                for v, c in counts.items()
            ]
        columns.append(profile)

    target_summary: TargetSummary | None = None
    if target and target in frame.columns:
        series = frame[target]
        n_unique = int(series.nunique(dropna=True))
        if n_unique <= 10:
            counts = series.value_counts(dropna=True)
            majority, minority = int(counts.iloc[0]), int(counts.iloc[-1])
            ratio = majority / max(1, minority)
            target_summary = TargetSummary(
                name=target,
                kind=ColumnKind.CATEGORICAL_NOMINAL,
                n_classes=n_unique,
                class_counts=[
                    CategoryCount(value=str(v), count=int(c), fraction=float(c) / len(frame))
                    for v, c in counts.items()
                ],
                imbalance_ratio=ratio,
                is_imbalanced=ratio > 3.0,
                n_missing=int(series.isna().sum()),
            )
        else:
            numeric = series.dropna().astype("float64")
            target_summary = TargetSummary(
                name=target,
                kind=ColumnKind.NUMERIC_CONTINUOUS,
                mean=float(numeric.mean()),
                std=float(numeric.std()),
                skewness=float(numeric.skew()),
                n_missing=int(series.isna().sum()),
            )

    findings = [
        LeakageFinding(
            column=col,
            score=0.99,
            method="auc_single_feature",
            severity=Severity.CRITICAL,
            reason=f"{col} separates the target almost perfectly and is recorded post-outcome.",
        )
        for col in (leakage or [])
        if col in frame.columns
    ]

    total_cells = max(1, int(frame.shape[0] * frame.shape[1]))
    missing_cells = int(frame.isna().sum().sum())
    duplicates = int(frame.duplicated().sum())
    return DatasetProfile(
        dataset_id=dataset_id,
        n_rows=int(len(frame)),
        n_columns=int(frame.shape[1]),
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
        n_duplicate_rows=duplicates,
        duplicate_fraction=duplicates / max(1, len(frame)),
        total_missing_cells=missing_cells,
        missing_cell_fraction=missing_cells / total_cells,
        columns=columns,
        target=target_summary,
        leakage_findings=findings,
        identifier_columns=[c.name for c in columns if c.looks_like_id],
        constant_columns=[c.name for c in columns if c.is_constant],
    )


@pytest.fixture
def churn_profile(churn_df: pd.DataFrame) -> DatasetProfile:
    """A hand-computed profile of churn.csv, independent of the real profiler."""
    return stub_profile(churn_df, target=CHURN_TARGET, leakage=[CHURN_LEAK])


def make_run_config(path: Path | None = None, **overrides: Any) -> RunConfig:
    """A :class:`RunConfig` with test-sized budgets.

    Args:
        path: CSV path for the data source; defaults to churn.csv.
        **overrides: Any RunConfig field to override.

    Returns:
        A validated RunConfig.
    """
    payload: dict[str, Any] = {
        "project": "tests",
        "source": DataSource(kind=SourceKind.CSV, uri=str(path or CHURN_CSV)),
        "target_column": CHURN_TARGET,
        "time_budget_seconds": 120,
        "max_experiments": 3,
        "cv_folds": 3,
        "enable_tuning": False,
        "enable_explainability": True,
        "report_formats": ["markdown", "json"],
    }
    payload.update(overrides)
    return RunConfig.model_validate(payload)


@pytest.fixture
def run_config(churn_csv: Path) -> RunConfig:
    """A RunConfig pointed at churn.csv with small budgets."""
    return make_run_config(churn_csv)


@pytest.fixture
def run_state(
    run_config: RunConfig,
    churn_df: pd.DataFrame,
    churn_profile: DatasetProfile,
    settings: Settings,
) -> RunState:
    """A RunState mid-flight: data loaded, profiled, problem defined, context frozen.

    This is the state most agent tests want — far enough along that ``build_prompt``
    has facts to render, without depending on any executor module.
    """
    state = RunState(
        config=run_config,
        settings=settings,
        bus=EventBus(run_id=run_config.run_id),
    )
    state.raw_df = churn_df.copy()
    state.working_df = churn_df.copy()
    state.profile = churn_profile
    state.problem = churn_problem()
    state.understanding = churn_understanding()
    state.freeze_context(build_run_context(churn_profile, run_config))
    state.mark_started()
    return state


@pytest.fixture
def regression_state(
    house_df: pd.DataFrame,
    settings: Settings,
) -> RunState:
    """A RunState for the regression example, for task-type-sensitive executors."""
    config = make_run_config(HOUSE_CSV, target_column=HOUSE_TARGET)
    profile = stub_profile(house_df, target=HOUSE_TARGET)
    state = RunState(config=config, settings=settings, bus=EventBus(run_id=config.run_id))
    state.raw_df = house_df.copy()
    state.working_df = house_df.copy()
    state.profile = profile
    state.problem = ProblemDefinition(
        task_type=TaskType.REGRESSION,
        target_column=HOUSE_TARGET,
        rationale="sale_price is continuous with 1,543 distinct values, so this is regression.",
        confidence="high",
        primary_metric="rmse",
        secondary_metrics=["mae", "r2"],
        metric_rationale="RMSE penalises the luxury-property errors that matter most in currency terms.",
        business_objective="Price listings within a defensible band.",
    )
    state.freeze_context(build_run_context(profile, config))
    state.mark_started()
    return state


# ===========================================================================
# Credentials gate for live tests
# ===========================================================================


def has_live_credentials() -> bool:
    """Whether a real Anthropic credential is present in the environment."""
    return bool(os.environ.get("AUTOML_LIVE_TESTS")) and bool(
        os.environ.get("ANTHROPIC_API_KEY_LIVE")
    )


requires_live = pytest.mark.skipif(
    not has_live_credentials(),
    reason="live API tests need AUTOML_LIVE_TESTS=1 and ANTHROPIC_API_KEY_LIVE",
)
