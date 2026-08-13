"""End-to-end self-test for the reporting layer.

Builds a small synthetic run — real dataframe, real fitted sklearn pipeline, real
splits, a hand-made profile, a few experiment results, and a short authored
report — then renders every chart kind and every export format. Run it with::

    python -m automl_architect.reporting.selftest

It exists because the reporting layer's failure modes are all integration-shaped:
a template variable that does not exist, a reportlab flowable that cannot wrap, a
chart that needs a column the splits do not carry. None of those show up in an
import check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..core.schemas import (
    BiasVarianceDiagnosis,
    BusinessInsight,
    CalibrationDiagnosis,
    CategoryCount,
    ChartKind,
    ChartSpec,
    CleaningAction,
    CleaningDecision,
    CleaningPlan,
    ColumnKind,
    ColumnProfile,
    ConfidenceInterval,
    CorrelationPair,
    DataQualityIssue,
    DataSource,
    DatasetProfile,
    DeploymentPattern,
    DeploymentRecommendation,
    EvaluationVerdict,
    ExperimentLog,
    ExperimentResult,
    ExplainabilityReport,
    FairnessSlice,
    FeatureAttribution,
    FeatureDecision,
    FeatureOp,
    FeaturePlan,
    FinalReport,
    InsightReport,
    MetricValue,
    MissingStrategy,
    ModelCandidate,
    ModelFamily,
    ModelSelection,
    ProblemDefinition,
    Quantiles,
    ReportSection,
    RunConfig,
    RunStatus,
    Severity,
    SourceKind,
    TargetSummary,
    TaskType,
    TuningDecision,
    TuningMethod,
    TuningResult,
    UsageTotals,
    VisualizationPlan,
    dict_to_params,
)
from ..core.state import DataSplits, RunState

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_calls", "data_gb"]
CATEGORICAL_FEATURES = ["contract"]
TEMPORAL_FEATURE = "signup_date"
TARGET = "churned"


def make_dataframe(n_rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic churn table with a genuine signal in it."""
    rng = np.random.default_rng(seed)
    tenure = rng.gamma(shape=2.2, scale=9.0, size=n_rows).round(1)
    charges = (rng.normal(72, 22, n_rows) + tenure * 0.35).round(2)
    calls = rng.poisson(1.6, n_rows)
    data_gb = np.abs(rng.normal(18, 9, n_rows)).round(2)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"], size=n_rows, p=[0.55, 0.28, 0.17]
    )
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    signup = [start + timedelta(days=int(d)) for d in rng.integers(0, 700, n_rows)]

    logit = (
        -0.9
        - 0.06 * tenure
        + 0.018 * charges
        + 0.42 * calls
        - 0.02 * data_gb
        + np.where(contract == "month-to-month", 0.85, -0.45)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    churned = np.where(rng.random(n_rows) < probability, "churned", "active")

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n_rows)],
            "tenure_months": tenure,
            "monthly_charges": charges,
            "support_calls": calls,
            "data_gb": data_gb,
            "contract": contract,
            TEMPORAL_FEATURE: pd.to_datetime(signup).tz_localize(None),
            TARGET: churned,
        }
    )
    # A little missingness, so the missingness chart has something to say.
    missing_index = rng.choice(n_rows, size=max(1, n_rows // 25), replace=False)
    frame.loc[missing_index, "data_gb"] = np.nan
    return frame


def make_profile(frame: pd.DataFrame) -> DatasetProfile:
    """Hand-build a profile for the synthetic frame.

    The real profiler lives in ``profiling/``; this is the minimum shape the
    reporting layer reads.
    """
    columns: list[ColumnProfile] = []
    for name in frame.columns:
        series = frame[name]
        kind = ColumnKind.UNKNOWN
        if name == "customer_id":
            kind = ColumnKind.IDENTIFIER
        elif pd.api.types.is_numeric_dtype(series):
            kind = (
                ColumnKind.NUMERIC_DISCRETE
                if pd.api.types.is_integer_dtype(series)
                else ColumnKind.NUMERIC_CONTINUOUS
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            kind = ColumnKind.DATETIME
        else:
            kind = ColumnKind.CATEGORICAL_NOMINAL
        profile = ColumnProfile(
            name=str(name),
            kind=kind,
            dtype=str(series.dtype),
            n_missing=int(series.isna().sum()),
            missing_fraction=float(series.isna().mean()),
            n_unique=int(series.nunique(dropna=True)),
            cardinality_ratio=float(series.nunique(dropna=True) / max(len(series), 1)),
            memory_bytes=int(series.memory_usage(deep=True)),
            looks_like_id=name == "customer_id",
        )
        if kind in (ColumnKind.NUMERIC_CONTINUOUS, ColumnKind.NUMERIC_DISCRETE):
            numeric = series.dropna().astype(float)
            profile.mean = float(numeric.mean())
            profile.std = float(numeric.std())
            profile.minimum = float(numeric.min())
            profile.maximum = float(numeric.max())
            profile.skewness = float(numeric.skew())
            profile.kurtosis = float(numeric.kurt())
            profile.quantiles = Quantiles(
                p25=float(numeric.quantile(0.25)),
                p50=float(numeric.quantile(0.50)),
                p75=float(numeric.quantile(0.75)),
            )
        if kind == ColumnKind.CATEGORICAL_NOMINAL:
            counts = series.value_counts().head(5)
            profile.top_values = [
                CategoryCount(
                    value=str(value), count=int(count), fraction=float(count / len(series))
                )
                for value, count in counts.items()
            ]
        columns.append(profile)

    target_counts = frame[TARGET].value_counts()
    majority, minority = int(target_counts.max()), int(target_counts.min())
    numeric_frame = frame[NUMERIC_FEATURES]
    correlations = numeric_frame.corr(numeric_only=True)
    pairs: list[CorrelationPair] = []
    for i, left in enumerate(NUMERIC_FEATURES):
        for right in NUMERIC_FEATURES[i + 1 :]:
            pairs.append(
                CorrelationPair(
                    left=left,
                    right=right,
                    coefficient=float(correlations.loc[left, right]),
                )
            )
    pairs.sort(key=lambda p: abs(p.coefficient), reverse=True)

    encoded_target = (frame[TARGET] == "churned").astype(int)
    target_pairs = [
        CorrelationPair(
            left=name,
            right=TARGET,
            coefficient=float(frame[name].corr(encoded_target)),
            method="point_biserial",
        )
        for name in NUMERIC_FEATURES
    ]
    target_pairs.sort(key=lambda p: abs(p.coefficient), reverse=True)

    return DatasetProfile(
        dataset_id="ds_selftest",
        n_rows=len(frame),
        n_columns=len(frame.columns),
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
        n_duplicate_rows=int(frame.duplicated().sum()),
        duplicate_fraction=float(frame.duplicated().mean()),
        total_missing_cells=int(frame.isna().sum().sum()),
        missing_cell_fraction=float(frame.isna().sum().sum() / frame.size),
        columns=columns,
        target=TargetSummary(
            name=TARGET,
            kind=ColumnKind.CATEGORICAL_NOMINAL,
            n_classes=int(target_counts.size),
            class_counts=[
                CategoryCount(
                    value=str(value), count=int(count), fraction=float(count / len(frame))
                )
                for value, count in target_counts.items()
            ],
            imbalance_ratio=majority / max(minority, 1),
            is_imbalanced=majority / max(minority, 1) > 3,
        ),
        top_correlations=pairs[:5],
        target_correlations=target_pairs,
        highly_correlated_pairs=[p for p in pairs if abs(p.coefficient) > 0.8],
        quality_issues=[
            DataQualityIssue(
                code="missing_values",
                severity=Severity.LOW,
                columns=["data_gb"],
                detail="data_gb is missing for roughly 4% of rows.",
            )
        ],
        temporal_columns=[TEMPORAL_FEATURE],
        identifier_columns=["customer_id"],
        profile_seconds=0.31,
    )


def fit_pipeline(frame: pd.DataFrame, seed: int = 42) -> tuple[Any, DataSplits, Any, list[str]]:
    """Fit a real pipeline and materialise real splits."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEMPORAL_FEATURE]
    features = frame[feature_columns].copy()
    encoder = LabelEncoder().fit(frame[TARGET])
    labels = encoder.transform(frame[TARGET])

    X_train, X_hold, y_train, y_hold = train_test_split(
        features, labels, test_size=0.35, random_state=seed, stratify=labels
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_hold, y_hold, test_size=0.55, random_state=seed, stratify=y_hold
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=400, random_state=seed)),
        ]
    )
    # Impute here rather than in the pipeline: the point is a fitted model to
    # render charts from, not a production-grade cleaning step.
    for split in (X_train, X_valid, X_test):
        split["data_gb"] = split["data_gb"].fillna(features["data_gb"].median())
    pipeline.fit(X_train, y_train)

    feature_names = [
        *NUMERIC_FEATURES,
        "contract=month-to-month",
        "contract=one-year",
        "contract=two-year",
    ]
    splits = DataSplits(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        strategy="stratified_random",
        rationale="Rows are independent customers with no temporal ordering in the "
        "label, so a stratified random split preserves the churn rate in each part.",
    )
    return pipeline, splits, encoder, feature_names


def make_report() -> FinalReport:
    """A short but structurally complete authored report."""
    return FinalReport(
        title="Customer churn: model, drivers, and recommended action",
        subtitle="Synthetic telecom dataset · 600 customers · binary classification",
        executive_summary=(
            "A regularised logistic regression predicts churn with a test ROC AUC of "
            "**0.884**, well clear of the 0.500 majority-class baseline.\n\n"
            "Three drivers dominate: short tenure, month-to-month contracts, and "
            "repeated support calls. Together they account for roughly two thirds of "
            "the model's attributed importance.\n\n"
            "The model is recommended for a batch scoring rollout with monthly "
            "retraining. The main caveat is that the support-call count is recorded "
            "with a lag, so scores for very new accounts are less reliable."
        ),
        sections=[
            ReportSection(
                heading="Data and quality",
                order=1,
                body_markdown=(
                    "The table holds **600 customers**, one row each, with seven "
                    "usable features and a binary churn label.\n\n"
                    "### What we found\n\n"
                    "- `data_gb` is missing for about 4% of rows; the median was "
                    "imputed rather than dropping the column, which would have cost "
                    "a measurable predictor.\n"
                    "- `customer_id` is a pure identifier and was excluded from "
                    "modelling.\n"
                    "- No duplicated rows and no constant columns.\n\n"
                    "| Check | Result |\n"
                    "| --- | --- |\n"
                    "| Rows | 600 |\n"
                    "| Missing cells | 24 |\n"
                    "| Duplicate rows | 0 |\n"
                    "| Class balance | 54 / 46 |\n"
                ),
                chart_refs=["missingness", "class_balance", "correlation_heatmap"],
            ),
            ReportSection(
                heading="Modelling and results",
                order=2,
                body_markdown=(
                    "Four families were tried against a stratified 5-fold split. "
                    "Logistic regression won on ROC AUC and is also the cheapest to "
                    "serve.\n\n"
                    "1. Logistic regression — 0.884\n"
                    "2. Histogram gradient boosting — 0.871\n"
                    "3. Majority-class baseline — 0.500\n\n"
                    "> The gap between the top two models is smaller than the "
                    "bootstrap confidence interval, so the simpler model was kept.\n\n"
                    "Scoring one customer costs about `0.4 ms`."
                ),
                chart_refs=["leaderboard", "roc_curve", "confusion_matrix"],
            ),
            ReportSection(
                heading="What drives churn",
                order=3,
                body_markdown=(
                    "Attribution is consistent between SHAP and permutation "
                    "importance, which is the signal that the ranking is real "
                    "rather than an artefact of one method.\n\n"
                    "- **Tenure** is the strongest single driver: risk falls steeply "
                    "over the first year.\n"
                    "- **Month-to-month contracts** carry roughly double the churn "
                    "rate of annual contracts.\n"
                    "- **Support calls** matter, but only above two calls per month.\n"
                ),
                chart_refs=["feature_importance", "partial_dependence"],
            ),
            ReportSection(
                heading="Risks and limitations",
                order=4,
                body_markdown=(
                    "The dataset is synthetic and small, so absolute numbers should "
                    "be treated as illustrative.\n\n"
                    "- Support-call counts are recorded with a reporting lag.\n"
                    "- No pricing or competitor data is present; a price change "
                    "would shift the relationship the model learned.\n"
                ),
                chart_refs=["calibration_curve"],
            ),
        ],
        deployment=DeploymentRecommendation(
            pattern=DeploymentPattern.BATCH_INFERENCE,
            rationale=(
                "Churn scores are consumed by a monthly retention campaign, so "
                "nightly batch scoring meets the need at a fraction of the cost of a "
                "live endpoint. Measured single-row latency of 0.4 ms means even a "
                "full re-score of the base takes under a second."
            ),
            estimated_latency_ms=0.4,
            estimated_throughput_rps=2400.0,
            model_size_mb=0.04,
            infrastructure_notes="A scheduled job writing to the CRM's scores table.",
            monitoring_plan=[
                "Track the population stability index of tenure and contract monthly.",
                "Alert if the predicted churn rate moves more than 5 points month over month.",
                "Re-check calibration on the previous month's realised churn.",
            ],
            retraining_cadence="monthly",
            rollout_strategy="Score 10% of the base in parallel with the current "
            "heuristic for one cycle, then switch over.",
            risks=[
                "Support-call lag degrades scores for accounts younger than 30 days.",
                "A pricing change would invalidate the learned charge relationship.",
            ],
        ),
        appendix_notes=[
            "All figures come from the held-out test split unless stated otherwise.",
            "The dataset is synthetic and generated for this self-test.",
        ],
    )


def build_state(workspace: Path | None = None) -> RunState:
    """Assemble a fully populated :class:`RunState` for the synthetic run."""
    frame = make_dataframe()
    pipeline, splits, encoder, feature_names = fit_pipeline(frame)
    settings = Settings(workspace=workspace) if workspace else Settings()
    config = RunConfig(
        project="reporting-selftest",
        source=DataSource(kind=SourceKind.DATAFRAME, uri="synthetic://churn"),
        target_column=TARGET,
        report_formats=["markdown", "html", "pdf", "pptx", "json"],
        fairness_attributes=["contract"],
    )
    state = RunState(config=config, settings=settings)
    state.mark_started()
    state.raw_df = frame
    state.working_df = frame
    state.splits = splits
    state.feature_names = feature_names
    state.label_encoder = encoder
    state.best_pipeline = pipeline
    state.best_model = pipeline
    state.profile = make_profile(frame)
    state.problem = ProblemDefinition(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column=TARGET,
        positive_class="churned",
        temporal_column=TEMPORAL_FEATURE,
        rationale="The target takes exactly two values and each row is one customer, "
        "so this is binary classification at the customer grain.",
        alternatives_considered=["survival analysis on time-to-churn"],
        confidence="high",
        primary_metric="roc_auc",
        secondary_metrics=["f1", "precision", "recall"],
        metric_rationale="Classes are close to balanced but the campaign ranks "
        "customers rather than thresholding, so a ranking metric is the right target.",
        business_objective="Rank customers by churn risk for a monthly retention offer.",
        constraints=["Scores must be explainable to the retention team."],
    )
    state.cleaning = CleaningPlan(
        decisions=[
            CleaningDecision(
                action=CleaningAction.IMPUTE_MISSING,
                columns=["data_gb"],
                strategy=MissingStrategy.MEDIAN,
                parameters=dict_to_params({"fallback": 0}),
                rationale="4% of rows are missing and the distribution is right-skewed "
                "(skewness 0.6), so the median is robust to the tail.",
                expected_impact="Retains a predictor worth about 0.01 AUC.",
            ),
            CleaningDecision(
                action=CleaningAction.DROP_COLUMN,
                columns=["customer_id"],
                rationale="Unique per row, so it carries no generalisable signal and "
                "would let a tree memorise the training set.",
                destructive=True,
                severity_if_skipped=Severity.HIGH,
            ),
        ],
        summary="One imputation and one identifier drop.",
        columns_to_drop=["customer_id"],
        drop_rationale=["customer_id is an identifier."],
        skipped_considerations=[
            "Outlier clipping on monthly_charges was skipped: the high values are "
            "genuine premium plans, not errors."
        ],
    )
    state.features = FeaturePlan(
        decisions=[
            FeatureDecision(
                op=FeatureOp.ONE_HOT_ENCODE,
                input_columns=["contract"],
                rationale="Three unordered categories, so one-hot encoding avoids "
                "implying an order the data does not have.",
                hypothesis="Month-to-month customers churn differently from "
                "contracted ones.",
                priority="high",
            ),
            FeatureDecision(
                op=FeatureOp.STANDARD_SCALE,
                input_columns=NUMERIC_FEATURES,
                rationale="Logistic regression with L2 regularisation needs "
                "comparable feature scales or the penalty is applied unevenly.",
                priority="medium",
            ),
        ],
        summary="Encode the contract type and standardise the numeric features.",
        expected_feature_count_delta=2,
        selection_strategy="Keep everything: seven features on 600 rows is not a "
        "dimensionality problem.",
    )
    state.model_selection = ModelSelection(
        candidates=[
            ModelCandidate(
                family=ModelFamily.LOGISTIC,
                rank=1,
                suitability="excellent",
                rationale="600 rows and seven features is squarely linear-model "
                "territory; regularisation controls the variance.",
                expected_strengths=["calibrated probabilities", "readable coefficients"],
                is_baseline=False,
                tune_priority="medium",
            ),
            ModelCandidate(
                family=ModelFamily.HIST_GRADIENT_BOOSTING,
                rank=2,
                suitability="good",
                rationale="Can capture the tenure non-linearity, at the cost of "
                "variance on a sample this small.",
                tune_priority="high",
            ),
            ModelCandidate(
                family=ModelFamily.BASELINE_DUMMY,
                rank=3,
                suitability="poor",
                rationale="Included only to anchor the leaderboard: any model that "
                "cannot beat the majority class has learned nothing.",
                is_baseline=True,
                tune_priority="none",
            ),
        ],
        summary="A linear model, one boosted-tree model, and a baseline.",
        reasoning="With 600 rows the bias-variance trade-off favours the simpler "
        "model unless the boosted trees win by more than the confidence interval.",
        excluded_families=["neural_network"],
        exclusion_rationale=["Far too little data to train a network usefully."],
        validation_strategy="stratified 5-fold cross-validation",
        validation_rationale="Stratification keeps the churn rate stable across "
        "folds, which matters at this sample size.",
    )
    state.experiments = ExperimentLog(
        results=[
            ExperimentResult(
                experiment_id="exp_logistic",
                family=ModelFamily.LOGISTIC,
                label="logistic regression",
                params=dict_to_params({"C": 1.0, "max_iter": 400}),
                metrics=[
                    MetricValue(name="roc_auc", value=0.884, std=0.021),
                    MetricValue(name="f1", value=0.781),
                    MetricValue(name="precision", value=0.769),
                    MetricValue(name="recall", value=0.794),
                ],
                cv_scores=[0.871, 0.889, 0.878, 0.895, 0.882],
                primary_metric="roc_auc",
                primary_score=0.884,
                train_seconds=0.042,
                predict_seconds=0.0004,
                n_features_in=7,
            ),
            ExperimentResult(
                experiment_id="exp_hgb",
                family=ModelFamily.HIST_GRADIENT_BOOSTING,
                label="hist gradient boosting",
                params=dict_to_params({"max_iter": 150, "learning_rate": 0.08}),
                metrics=[MetricValue(name="roc_auc", value=0.871, std=0.034)],
                cv_scores=[0.842, 0.881, 0.869, 0.887, 0.876],
                primary_metric="roc_auc",
                primary_score=0.871,
                train_seconds=0.63,
                n_features_in=7,
            ),
            ExperimentResult(
                experiment_id="exp_dummy",
                family=ModelFamily.BASELINE_DUMMY,
                label="majority class",
                metrics=[MetricValue(name="roc_auc", value=0.5)],
                primary_metric="roc_auc",
                primary_score=0.5,
                train_seconds=0.001,
                n_features_in=7,
                is_baseline=True,
            ),
            ExperimentResult(
                experiment_id="exp_svm",
                family=ModelFamily.SVM,
                label="rbf svm",
                primary_metric="roc_auc",
                failed=True,
                error="fit exceeded the per-model time budget",
            ),
        ],
        best_experiment_id="exp_logistic",
        primary_metric="roc_auc",
        higher_is_better=True,
        leaderboard_notes="The top two are within one standard error of each other.",
    )
    state.tuning_decision = TuningDecision(
        worthwhile=True,
        rationale="A 0.013 gap between the top two models is within noise, so a "
        "cheap random search on the leader is better value than adding families.",
        method=TuningMethod.RANDOM_SEARCH,
        method_rationale="The search space is small and low-dimensional.",
        target_family=ModelFamily.LOGISTIC,
        n_trials=20,
        expected_gain="0.005 to 0.015 AUC",
    )
    state.tuning = TuningResult(
        ran=True,
        method=TuningMethod.RANDOM_SEARCH,
        family=ModelFamily.LOGISTIC,
        n_trials_completed=20,
        best_params=dict_to_params({"C": 0.7}),
        best_score=0.891,
        baseline_score=0.884,
        improvement=0.007,
        seconds=4.8,
        trial_scores=[0.869, 0.874, 0.881, 0.884, 0.887, 0.891],
    )
    state.explainability = ExplainabilityReport(
        global_attributions=[
            FeatureAttribution(feature="tenure_months", importance=0.34, direction="decreases"),
            FeatureAttribution(
                feature="contract=month-to-month", importance=0.21, direction="increases"
            ),
            FeatureAttribution(feature="support_calls", importance=0.18, direction="increases"),
            FeatureAttribution(feature="monthly_charges", importance=0.15, direction="increases"),
            FeatureAttribution(feature="data_gb", importance=0.07, direction="decreases"),
            FeatureAttribution(feature="contract=two-year", importance=0.05, direction="decreases"),
        ],
        permutation_importance=[
            FeatureAttribution(
                feature="tenure_months", importance=0.31, method="permutation"
            ),
            FeatureAttribution(
                feature="support_calls", importance=0.22, method="permutation"
            ),
        ],
        shap_available=True,
        plain_language_explanations=[
            "Customer tenure accounts for about 34% of the model's attributed importance.",
            "Being on a month-to-month contract raises predicted churn risk the most "
            "of any single categorical value.",
        ],
        narrative="The model has learned the two relationships a retention analyst "
        "would expect, and it weights them in the order they would guess.",
        method_notes="SHAP TreeExplainer was unavailable for a linear model, so "
        "coefficient-based attribution was used and cross-checked with permutation "
        "importance on the test split.",
    )
    state.evaluation = EvaluationVerdict(
        acceptable=True,
        verdict_rationale="Test AUC of 0.884 with a train-test gap of 0.018 shows the "
        "model generalises; it beats the baseline by a wide margin on the metric the "
        "business cares about.",
        overall_grade="B",
        bias_variance=BiasVarianceDiagnosis(
            train_score=0.902,
            validation_score=0.889,
            test_score=0.884,
            gap=0.018,
            verdict="good_fit",
            detail="A gap under 0.02 on 600 rows is well within sampling noise.",
        ),
        calibration=CalibrationDiagnosis(
            applicable=True,
            brier_score=0.148,
            expected_calibration_error=0.031,
            verdict="Well calibrated; no post-hoc calibration needed.",
        ),
        generalisation_notes="Cross-validation standard deviation is 0.021.",
        drift_risk="medium",
        drift_rationale="Contract mix shifts with promotions, and contract type is "
        "the second-strongest driver.",
        fairness_slices=[
            FairnessSlice(
                attribute="contract",
                slice_value="month-to-month",
                n_rows=118,
                metric_name="roc_auc",
                metric_value=0.861,
                delta_vs_overall=-0.023,
            ),
            FairnessSlice(
                attribute="contract",
                slice_value="two-year",
                n_rows=36,
                metric_name="roc_auc",
                metric_value=0.902,
                delta_vs_overall=0.018,
            ),
        ],
        fairness_notes="No slice deviates by more than 0.03 AUC.",
        confidence_intervals=[
            ConfidenceInterval(
                metric="roc_auc", point_estimate=0.884, lower=0.842, upper=0.921
            )
        ],
        residual_notes="Not applicable to a classifier.",
        error_analysis=[
            "Most false negatives are long-tenure customers who churned after a "
            "price change, which the data does not carry."
        ],
        learning_curve_notes="Validation score is still rising slightly at full "
        "sample size, so more data would help.",
        weaknesses=["Small sample", "No pricing features"],
        recommended_action="accept",
        action_rationale="The model clears the business threshold and the failure "
        "modes are understood.",
        specific_improvements=[
            "Add a pricing-change flag.",
            "Backfill support-call data so new accounts score reliably.",
        ],
    )
    state.insights = InsightReport(
        executive_summary="Churn is concentrated in the first year of month-to-month "
        "contracts, which is exactly where a retention offer is cheapest to run.",
        insights=[
            BusinessInsight(
                headline="Target the first six months of month-to-month contracts",
                detail="Predicted churn risk is roughly double the base rate for "
                "customers under six months on a month-to-month plan.",
                supporting_evidence="Tenure and contract type carry 55% of attributed "
                "importance combined.",
                recommended_action="Offer a one-year contract incentive at month four.",
                expected_value="Roughly 40 retained accounts per quarter.",
                confidence="medium",
                audience="marketing",
            ),
            BusinessInsight(
                headline="Repeat support callers are a leading indicator",
                detail="Risk rises sharply above two support calls in a month.",
                supporting_evidence="Support calls are the third-strongest driver at "
                "18% of attributed importance.",
                recommended_action="Route third-call customers to a retention specialist.",
                confidence="high",
                audience="operations",
            ),
        ],
        key_drivers_plain_language=[
            "New customers leave more often than established ones.",
            "Flexible contracts churn about twice as much as annual ones.",
        ],
        caveats=["The dataset is synthetic, so effect sizes are illustrative."],
        suggested_next_experiments=[
            "Add competitor pricing to the feature set.",
            "Test an uplift model rather than a risk model for the campaign.",
        ],
    )
    state.usage = UsageTotals(
        input_tokens=184_320,
        output_tokens=21_450,
        cache_read_tokens=96_100,
        cache_write_tokens=12_800,
        llm_calls=13,
        cost_usd=1.5834,
    )
    state.add_warning("SHAP was unavailable for the linear model; used coefficients.")
    return state


def make_visualization_plan() -> VisualizationPlan:
    """A plan that requests every chart kind, so nothing goes untested."""
    rationales: dict[ChartKind, tuple[str, str, list[str]]] = {
        ChartKind.MISSINGNESS: (
            "Missing values by column",
            "Shows whether missingness is concentrated in one column or spread across the table.",
            [],
        ),
        ChartKind.CLASS_BALANCE: (
            "Churn class balance",
            "Establishes whether accuracy would be a misleading metric here.",
            [],
        ),
        ChartKind.HISTOGRAM: (
            "Tenure distribution",
            "Tenure drives the prediction, so its shape decides whether a linear term is enough.",
            ["tenure_months"],
        ),
        ChartKind.BOX: (
            "Spread of the numeric features",
            "Compares dispersion and outlier load across the numeric predictors.",
            NUMERIC_FEATURES,
        ),
        ChartKind.BAR: (
            "Contract mix",
            "Shows how much of the base sits on the highest-risk contract type.",
            ["contract"],
        ),
        ChartKind.SCATTER: (
            "Charges against tenure by churn",
            "Tests whether the two strongest predictors separate the classes on their own.",
            ["tenure_months", "monthly_charges"],
        ),
        ChartKind.LINE: (
            "Monthly charges over signup date",
            "Checks for drift in pricing over the signup window.",
            [TEMPORAL_FEATURE, "monthly_charges"],
        ),
        ChartKind.CORRELATION_HEATMAP: (
            "Feature correlations",
            "Identifies redundancy that would destabilise the linear model's coefficients.",
            [],
        ),
        ChartKind.LEADERBOARD: (
            "Model leaderboard",
            "Compares every candidate on ROC AUC against the baseline.",
            [],
        ),
        ChartKind.ROC_CURVE: (
            "ROC curve",
            "Shows the achievable trade-off between catching churners and false alarms.",
            [],
        ),
        ChartKind.PR_CURVE: (
            "Precision-recall curve",
            "The campaign contacts a fixed number of customers, so precision at low recall is what matters.",
            [],
        ),
        ChartKind.CONFUSION_MATRIX: (
            "Confusion matrix",
            "Turns the score into the counts the retention team will actually see.",
            [],
        ),
        ChartKind.CALIBRATION_CURVE: (
            "Calibration curve",
            "Predicted probabilities are used to size the offer, so they must mean what they say.",
            [],
        ),
        ChartKind.PREDICTION_DISTRIBUTION: (
            "Score distribution by actual class",
            "Shows how much of the score range is genuinely discriminative.",
            [],
        ),
        ChartKind.LEARNING_CURVE: (
            "Learning curve",
            "Answers whether more data would help more than more modelling.",
            [],
        ),
        ChartKind.FEATURE_IMPORTANCE: (
            "Feature importance",
            "Ranks the drivers the retention team can actually act on.",
            [],
        ),
        ChartKind.SHAP_SUMMARY: (
            "SHAP attribution",
            "Cross-checks the importance ranking with a per-prediction method.",
            [],
        ),
        ChartKind.PARTIAL_DEPENDENCE: (
            "Partial dependence of the top drivers",
            "Shows the shape of each relationship, not just its strength.",
            ["tenure_months", "monthly_charges"],
        ),
        ChartKind.RESIDUALS: (
            "Residuals",
            "Included to confirm the renderer skips regression-only charts cleanly.",
            [],
        ),
        ChartKind.RESIDUAL_HISTOGRAM: (
            "Residual distribution",
            "Included to confirm the renderer skips regression-only charts cleanly.",
            [],
        ),
        ChartKind.TIME_SERIES_FORECAST: (
            "Actual against predicted over signup date",
            "Checks whether error is concentrated in a particular signup cohort.",
            [TEMPORAL_FEATURE],
        ),
    }
    charts = [
        ChartSpec(
            kind=kind,
            title=title,
            columns=columns,
            rationale=rationale,
            priority="high" if kind in (ChartKind.LEADERBOARD, ChartKind.ROC_CURVE) else "medium",
        )
        for kind, (title, rationale, columns) in rationales.items()
    ]
    return VisualizationPlan(
        charts=charts,
        dashboard_narrative=(
            "The dashboard reads top to bottom as the analysis ran: what the data "
            "looked like, how the candidate models compared, and which features the "
            "winning model leans on.\n\n"
            "Two charts are worth pausing on. The leaderboard shows the baseline "
            "in grey — every real model clears it comfortably. The calibration "
            "curve matters because the retention campaign sizes its offer from the "
            "predicted probability, not just the ranking."
        ),
    )


def run_selftest(workspace: Path | None = None) -> dict[str, Any]:
    """Render every chart and every report format for a synthetic run.

    Args:
        workspace: Optional workspace root. Defaults to the configured one.

    Returns:
        A dict with the chart results, the dashboard path, and the report paths.
    """
    from .charts import render_charts
    from .writer import write_report

    state = build_state(workspace)
    plan = make_visualization_plan()
    state.visualization_plan = plan
    bundle = render_charts(state, plan)
    report = make_report()
    state.report = report
    state.mark_finished(RunStatus.COMPLETED)
    report_bundle = write_report(state, report, state.config.report_formats)
    return {
        "run_dir": str(state.artifact_dir),
        "charts": [
            {
                "title": a.spec.title,
                "kind": a.spec.kind.value,
                "rendered": a.rendered,
                "html": a.html_path,
                "png": a.png_path,
                "error": a.error,
            }
            for a in bundle.artifacts
        ],
        "dashboard": bundle.dashboard_path,
        "report": report_bundle.model_dump(),
        "warnings": list(state.warnings),
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the self-test."""
    parser = argparse.ArgumentParser(description="Render a synthetic AutoML report.")
    parser.add_argument(
        "--workspace", type=Path, default=None, help="Workspace root for artifacts."
    )
    args = parser.parse_args(argv)
    result = run_selftest(args.workspace)

    rendered = [c for c in result["charts"] if c["rendered"]]
    skipped = [c for c in result["charts"] if not c["rendered"]]
    print(f"run directory: {result['run_dir']}")
    print(f"\ncharts rendered: {len(rendered)}/{len(result['charts'])}")
    for chart in rendered:
        image = Path(chart["png"]).name if chart["png"] else "no png"
        print(f"  [ok]   {chart['kind']:<24} {Path(chart['html']).name}  ({image})")
    for chart in skipped:
        print(f"  [skip] {chart['kind']:<24} {chart['error']}")

    print(f"\ndashboard: {result['dashboard']}")
    print("report paths:")
    for key, value in result["report"].items():
        if key == "warnings":
            continue
        print(f"  {key:<16} {value}")
    warnings = result["report"].get("warnings") or []
    if warnings:
        print("\nreport warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if result["warnings"]:
        print("\nrun warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    for key in ("markdown_path", "html_path", "json_path"):
        if not result["report"].get(key):
            print(f"\nFAILED: {key} was not written")
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
