"""Self-test for ``execution/explainer.py`` and ``execution/diagnostics.py``.

Builds a small fitted pipeline for classification and for regression, runs both
executors against a real :class:`RunState`, and prints the rendered
``to_prompt()`` output. Also exercises the degraded path (no fitted model) to
prove neither module raises.

Run it with the project interpreter::

    .venv/Scripts/python.exe scripts/selftest_execution_explain.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Runnable from a checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Redirect artifacts away from the repo before settings are constructed.
os.environ.setdefault(
    "AUTOML_WORKSPACE", str(Path(tempfile.gettempdir()) / "automl_selftest_explain")
)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automl_architect.core.schemas import (
    DataSource,
    ProblemDefinition,
    RunConfig,
    SourceKind,
    TaskType,
)
from automl_architect.core.state import DataSplits, RunState
from automl_architect.execution.diagnostics import compute_diagnostics
from automl_architect.execution.explainer import compute_explanations

RANDOM_STATE = 7


def _make_frame(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    return pd.DataFrame(
        {
            "tenure_months": rng.integers(1, 72, n).astype(float),
            "monthly_charges": rng.normal(70, 25, n).round(2),
            "support_tickets": rng.poisson(1.2, n).astype(float),
            "city": rng.choice(["Paris", "Rome", "Oslo", "Lisbon"], n, p=[0.4, 0.3, 0.2, 0.1]),
            "plan": rng.choice(["basic", "pro"], n),
        }
    )


def _split(frame: pd.DataFrame, y: pd.Series) -> DataSplits:
    n = len(frame)
    n_train, n_valid = int(n * 0.65), int(n * 0.15)
    return DataSplits(
        X_train=frame.iloc[:n_train],
        y_train=y.iloc[:n_train],
        X_valid=frame.iloc[n_train : n_train + n_valid],
        y_valid=y.iloc[n_train : n_train + n_valid],
        X_test=frame.iloc[n_train + n_valid :],
        y_test=y.iloc[n_train + n_valid :],
        strategy="stratified_holdout",
        rationale="self-test",
    )


def _pipeline(numeric: list[str], categorical: list[str], model: object) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def _state(
    frame: pd.DataFrame,
    target: pd.Series,
    features: pd.DataFrame,
    problem: ProblemDefinition,
) -> RunState:
    config = RunConfig(
        project="selftest",
        source=DataSource(kind=SourceKind.DATAFRAME, uri="memory://selftest"),
        target_column=problem.target_column,
        fairness_attributes=["city", "not_a_real_column"],
        random_state=RANDOM_STATE,
        time_budget_seconds=900,
    )
    state = RunState(config=config)
    working = frame.copy()
    working[problem.target_column or "target"] = target
    state.raw_df = working
    state.working_df = working
    state.feature_frame = features
    state.problem = problem
    state.splits = _split(features, target)
    return state


def run_classification() -> None:
    print("=" * 78)
    print("CLASSIFICATION SELF-TEST (RandomForest pipeline, one-hot categoricals)")
    print("=" * 78)
    frame = _make_frame()
    signal = (
        -0.05 * frame["tenure_months"]
        + 0.03 * frame["monthly_charges"]
        + 0.7 * frame["support_tickets"]
        + 1.2 * (frame["city"] == "Paris")
        - 1.0
    )
    rng = np.random.default_rng(RANDOM_STATE + 1)
    target = pd.Series(
        np.where(signal + rng.normal(0, 0.8, len(frame)) > 0, "churn", "stay"), name="churn"
    )
    features = frame
    problem = ProblemDefinition(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class="churn",
        rationale="two-class categorical target",
        confidence="high",
        primary_metric="roc_auc",
        secondary_metrics=["f1", "accuracy"],
        metric_rationale="ranking quality matters more than a fixed threshold",
        business_objective="retain at-risk subscribers",
    )
    state = _state(frame, target, features, problem)
    numeric = ["tenure_months", "monthly_charges", "support_tickets"]
    categorical = ["city", "plan"]
    pipeline = _pipeline(
        numeric,
        categorical,
        RandomForestClassifier(n_estimators=60, max_depth=6, random_state=RANDOM_STATE),
    )
    pipeline.fit(state.splits.X_train, state.splits.y_train)
    state.best_pipeline = pipeline
    state.best_model = pipeline[-1]
    state.preprocessor = pipeline[:-1]

    report = compute_explanations(state)
    _print_explanations(report)

    bundle = compute_diagnostics(state)
    print(bundle.to_prompt())
    print(f"warnings so far: {state.warnings}")
    print()


def run_regression() -> None:
    print("=" * 78)
    print("REGRESSION SELF-TEST (Ridge pipeline, coef_ attributions)")
    print("=" * 78)
    frame = _make_frame(700)
    rng = np.random.default_rng(RANDOM_STATE + 2)
    target = pd.Series(
         12.0
        + 1.8 * frame["monthly_charges"]
        - 0.9 * frame["tenure_months"]
        + 6.0 * (frame["city"] == "Oslo")
        + rng.normal(0, 12, len(frame)),
        name="revenue",
    )
    problem = ProblemDefinition(
        task_type=TaskType.REGRESSION,
        target_column="revenue",
        rationale="continuous numeric target",
        confidence="high",
        primary_metric="rmse",
        secondary_metrics=["mae", "r2"],
        metric_rationale="errors are symmetric and in target units",
        business_objective="forecast account revenue",
    )
    state = _state(frame, target, frame, problem)
    pipeline = _pipeline(
        ["tenure_months", "monthly_charges", "support_tickets"],
        ["city", "plan"],
        Ridge(alpha=1.0, random_state=RANDOM_STATE),
    )
    pipeline.fit(state.splits.X_train, state.splits.y_train)
    state.best_pipeline = pipeline

    report = compute_explanations(state)
    _print_explanations(report)

    bundle = compute_diagnostics(state)
    print(bundle.to_prompt())
    print(f"warnings so far: {state.warnings}")
    print()


def run_degraded() -> None:
    print("=" * 78)
    print("DEGRADED SELF-TEST (no fitted model)")
    print("=" * 78)
    frame = _make_frame(60)
    target = pd.Series(np.arange(60, dtype=float), name="y")
    problem = ProblemDefinition(
        task_type=TaskType.REGRESSION,
        target_column="y",
        rationale="n/a",
        confidence="low",
        primary_metric="rmse",
        metric_rationale="n/a",
        business_objective="n/a",
    )
    state = _state(frame, target, frame, problem)
    report = compute_explanations(state)
    bundle = compute_diagnostics(state)
    assert report.global_attributions == []
    assert bundle.bias_variance.verdict == "inconclusive"
    print(f"explainability degraded cleanly: {report.method_notes}")
    print(f"diagnostics notes: {bundle.notes}")
    print(f"warnings: {state.warnings}")
    print()


def _print_explanations(report: object) -> None:
    print("--- ExplainabilityReport ---")
    print(f"shap_available: {report.shap_available}")
    print(f"shap_summary_path: {report.shap_summary_path}")
    print("global_attributions:")
    for attribution in report.global_attributions:
        print(
            f"  {attribution.feature:<20} {attribution.importance:6.3f} "
            f"{attribution.direction:<10} ({attribution.method})"
        )
    print("permutation_importance:")
    for attribution in report.permutation_importance[:6]:
        print(f"  {attribution.feature:<20} {attribution.importance:6.3f}")
    print(f"total normalised importance: {sum(a.importance for a in report.global_attributions):.4f}")
    print("partial_dependence_paths:")
    for path in report.partial_dependence_paths:
        print(f"  {path}")
    print("counterfactuals:")
    for counterfactual in report.counterfactuals[:4]:
        print(f"  {counterfactual.description}")
    print("plain language:")
    for line in report.plain_language_explanations:
        print(f"  {line}")
    print(f"method_notes: {report.method_notes}")
    print()


def main() -> None:
    run_classification()
    run_regression()
    run_degraded()
    print("self-test complete")


if __name__ == "__main__":
    main()
