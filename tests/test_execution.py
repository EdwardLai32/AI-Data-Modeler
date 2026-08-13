"""The deterministic half of the system: cleaning, features, splits, metrics, models.

Nothing here calls an LLM. These modules take a typed decision and apply it with
real pandas and scikit-learn, so the tests are about *behaviour on data*: did the
leakage column actually leave the frame, does the class ratio survive the split,
is the RMSE the number you get by hand.

Two invariants recur and are worth naming, because they are the project's hard
rules rather than incidental preferences:

*   **An agent-supplied column name is never trusted.** Every executor is handed a
    plan that references a column which does not exist, and must log and continue.
*   **A failure degrades, it does not crash.** An unavailable model family, an
    op on the wrong dtype, a missing optional dependency â€” the run continues with
    reduced capability and a recorded warning.

One thing to know before reading the feature tests. ``feature_ops`` does *not*
return a fully numeric frame: anything learned from the feature distribution â€”
encoding, scaling, binning, power transforms, PCA, selection â€” is deferred into an
unfitted pipeline on ``state.preprocessor`` that the trainer fits on the training
partition alone. Encoding early would leak the test split's category set and scale
into the transform. So "is this model-ready" is a question about the frame *and*
the preprocessor together, and :func:`materialise` answers it the same way the
trainer does.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from automl_architect.core.schemas import (
    CleaningAction,
    CleaningDecision,
    CleaningPlan,
    ExperimentLog,
    ExplainabilityReport,
    FeatureDecision,
    FeatureOp,
    FeaturePlan,
    MissingStrategy,
    ModelFamily,
    Param,
    TaskType,
    TuningDecision,
    TuningMethod,
    TuningResult,
    params_to_dict,
)
from automl_architect.core.state import DataSplits, RunState

from .conftest import (
    CHURN_LEAK,
    CHURN_TARGET,
    HOUSE_TARGET,
    churn_cleaning,
    churn_features,
    churn_model_selection,
    import_or_skip,
)

# ---------------------------------------------------------------------------
# Module handles
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def metrics_mod():  # noqa: ANN201
    return import_or_skip("automl_architect.execution.metrics", feature="execution/metrics.py")


@pytest.fixture(scope="module")
def zoo():  # noqa: ANN201
    return import_or_skip("automl_architect.execution.model_zoo", feature="execution/model_zoo.py")


@pytest.fixture(scope="module")
def splitter():  # noqa: ANN201
    return import_or_skip("automl_architect.execution.splitter", feature="execution/splitter.py")


@pytest.fixture(scope="module")
def cleaning_ops():  # noqa: ANN201
    return import_or_skip("automl_architect.execution.cleaning_ops", feature="execution/cleaning_ops.py")


@pytest.fixture(scope="module")
def feature_ops():  # noqa: ANN201
    return import_or_skip("automl_architect.execution.feature_ops", feature="execution/feature_ops.py")


@pytest.fixture(scope="module")
def trainer():  # noqa: ANN201
    return import_or_skip("automl_architect.execution.trainer", feature="execution/trainer.py")


@pytest.fixture(scope="module")
def tuner():  # noqa: ANN201
    return import_or_skip("automl_architect.execution.tuner", feature="execution/tuner.py")


@pytest.fixture(scope="module")
def explainer():  # noqa: ANN201
    return import_or_skip("automl_architect.execution.explainer", feature="execution/explainer.py")


@pytest.fixture(scope="module")
def diagnostics_mod():  # noqa: ANN201
    return import_or_skip("automl_architect.execution.diagnostics", feature="execution/diagnostics.py")


# ===========================================================================
# metrics
# ===========================================================================


class TestMetrics:
    """``primary_metric_for`` decides what the whole run optimises, so it matters."""

    def test_primary_metric_per_task(self, metrics_mod) -> None:
        binary = metrics_mod.primary_metric_for(TaskType.BINARY_CLASSIFICATION)
        assert binary in {"roc_auc", "average_precision", "f1"}, binary
        regression = metrics_mod.primary_metric_for(TaskType.REGRESSION)
        assert regression in {"rmse", "mae", "r2", "mape"}, regression
        multiclass = metrics_mod.primary_metric_for(TaskType.MULTICLASS_CLASSIFICATION)
        assert isinstance(multiclass, str) and multiclass

    def test_every_supported_task_has_a_metric(self, metrics_mod) -> None:
        for task in TaskType:
            if not task.is_supported:
                continue
            metric = metrics_mod.primary_metric_for(task)
            assert isinstance(metric, str) and metric, task

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("roc_auc", True),
            ("accuracy", True),
            ("f1", True),
            ("r2", True),
            ("average_precision", True),
            ("silhouette", True),
            ("rmse", False),
            ("mae", False),
            ("log_loss", False),
            ("brier_score", False),
        ],
    )
    def test_metric_direction(self, metrics_mod, metric: str, expected: bool) -> None:
        """Getting a direction backwards silently selects the worst model."""
        assert metrics_mod.higher_is_better(metric) is expected, metric

    def test_direction_is_case_insensitive(self, metrics_mod) -> None:
        assert metrics_mod.higher_is_better("ROC_AUC") is True
        assert metrics_mod.higher_is_better("RMSE") is False

    def test_binary_scores_on_perfect_predictions(self, metrics_mod) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        proba = np.where(y_true == 1, 0.9, 0.1)
        scores = metrics_mod.score_predictions(
            TaskType.BINARY_CLASSIFICATION, y_true, y_true, y_proba=proba
        )
        assert scores["accuracy"] == pytest.approx(1.0)
        assert scores.get("roc_auc") == pytest.approx(1.0)
        assert all(isinstance(v, float) and math.isfinite(v) for v in scores.values())

    def test_binary_scores_without_probabilities(self, metrics_mod) -> None:
        """A model with no ``predict_proba`` still has to be scoreable."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        scores = metrics_mod.score_predictions(TaskType.BINARY_CLASSIFICATION, y_true, y_pred)
        assert scores["accuracy"] == pytest.approx(4 / 6)
        assert all(math.isfinite(v) for v in scores.values())

    def test_regression_rmse_matches_hand_arithmetic(self, metrics_mod) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5])
        expected_rmse = math.sqrt(float(np.mean((y_true - y_pred) ** 2)))
        scores = metrics_mod.score_predictions(TaskType.REGRESSION, y_true, y_pred)
        assert scores["rmse"] == pytest.approx(expected_rmse)
        assert scores["mae"] == pytest.approx(0.5)
        assert scores["r2"] <= 1.0

    def test_multiclass_scores(self, metrics_mod) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        scores = metrics_mod.score_predictions(
            TaskType.MULTICLASS_CLASSIFICATION, y_true, y_pred, labels=[0, 1, 2]
        )
        assert scores["accuracy"] == pytest.approx(4 / 6)
        assert all(math.isfinite(v) for v in scores.values())

    def test_single_class_present_degrades(self, metrics_mod) -> None:
        """A fold that contains one class makes AUC undefined; it must not raise."""
        y_true = np.zeros(10, dtype=int)
        scores = metrics_mod.score_predictions(
            TaskType.BINARY_CLASSIFICATION, y_true, y_true, y_proba=np.full(10, 0.3)
        )
        assert scores["accuracy"] == pytest.approx(1.0)
        assert "roc_auc" not in scores or math.isnan(scores["roc_auc"]) or 0.0 <= scores["roc_auc"] <= 1.0

    def test_scorer_names_are_real_sklearn_scorers(self, metrics_mod) -> None:
        """The name is passed straight to ``cross_val_score``, so it must resolve."""
        from sklearn.metrics import get_scorer, get_scorer_names

        known = set(get_scorer_names())
        for task, metric in (
            (TaskType.BINARY_CLASSIFICATION, "roc_auc"),
            (TaskType.REGRESSION, "rmse"),
            (TaskType.REGRESSION, "mae"),
            (TaskType.MULTICLASS_CLASSIFICATION, "accuracy"),
        ):
            name = metrics_mod.sklearn_scorer_name(metric, task)
            assert name in known, f"{metric}/{task.value} -> {name!r} is not an sklearn scorer"
            assert get_scorer(name) is not None

    def test_scorer_name_for_error_metric_is_negated(self, metrics_mod) -> None:
        """sklearn scorers are always maximised, so RMSE has to become neg_*."""
        name = metrics_mod.sklearn_scorer_name("rmse", TaskType.REGRESSION)
        assert name.startswith("neg_")


# ===========================================================================
# model zoo
# ===========================================================================


class TestModelZoo:
    def test_families_offered_for_each_supported_task(self, zoo) -> None:
        for task in TaskType:
            if not task.is_supported:
                continue
            families = zoo.available_families(task)
            assert families, f"no families for {task.value}"
            assert len(set(families)) == len(families), f"duplicates for {task.value}"

    def test_offered_families_are_actually_available(self, zoo) -> None:
        """Offering a family whose package is absent guarantees a failed experiment."""
        for task in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
            for family in zoo.available_families(task):
                assert zoo.is_available(family), f"{family.value} offered but not available"

    def test_task_appropriate_families(self, zoo) -> None:
        """A classifier must not be offered for regression, or an unsupervised model for either.

        Note that a family name is a *regularisation* choice, not an estimator
        class: ``lasso`` for classification is legitimately L1-penalised logistic
        regression, so the test checks the ones with no such reading.
        """
        classification = set(zoo.available_families(TaskType.BINARY_CLASSIFICATION))
        regression = set(zoo.available_families(TaskType.REGRESSION))
        assert ModelFamily.LOGISTIC not in regression
        for unsupervised in (ModelFamily.KMEANS, ModelFamily.DBSCAN, ModelFamily.ISOLATION_FOREST):
            assert unsupervised not in classification
            assert unsupervised not in regression

    def test_classifiers_predict_labels_and_regressors_predict_values(self, zoo) -> None:
        """The real task-appropriateness test: what comes out of ``predict``."""
        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))

        y_class = pd.Series((X["a"] > 0).astype(int))
        for family in zoo.available_families(TaskType.BINARY_CLASSIFICATION):
            estimator = zoo.build_estimator(
                family, TaskType.BINARY_CLASSIFICATION, {}, random_state=42
            ).fit(X, y_class)
            assert set(np.unique(estimator.predict(X))) <= {0, 1}, family

        y_value = pd.Series(3.0 * X["a"] + rng.normal(0, 0.2, 120))
        for family in zoo.available_families(TaskType.REGRESSION):
            estimator = zoo.build_estimator(
                family, TaskType.REGRESSION, {}, random_state=42
            ).fit(X, y_value)
            predictions = np.asarray(estimator.predict(X), dtype="float64")
            assert np.isfinite(predictions).all(), family

    def test_baseline_is_offered_for_supervised_tasks(self, zoo) -> None:
        """Without a floor, no score can be judged as good or bad."""
        for task in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
            assert ModelFamily.BASELINE_DUMMY in zoo.available_families(task)

    def test_uninstalled_package_reports_unavailable(self, zoo) -> None:
        """catboost is not installed in this environment."""
        assert zoo.is_available(ModelFamily.CATBOOST) is False

    def test_installed_boosters_report_available(self, zoo) -> None:
        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        assert zoo.is_available(ModelFamily.XGBOOST) is True
        assert zoo.is_available(ModelFamily.LIGHTGBM) is True

    def test_every_estimator_fits_and_predicts(self, zoo) -> None:
        """Smoke-fit every offered classifier so a broken constructor is caught here."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((X["f0"] + rng.normal(0, 0.4, 120) > 0).astype(int))

        for family in zoo.available_families(TaskType.BINARY_CLASSIFICATION):
            estimator = zoo.build_estimator(
                family, TaskType.BINARY_CLASSIFICATION, {}, random_state=42
            )
            assert hasattr(estimator, "fit") and hasattr(estimator, "predict"), family
            estimator.fit(X, y)
            predictions = estimator.predict(X)
            assert len(predictions) == len(y), family

    def test_every_regressor_fits_and_predicts(self, zoo) -> None:
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"f{i}" for i in range(4)])
        y = pd.Series(2.0 * X["f0"] - X["f1"] + rng.normal(0, 0.3, 120))

        for family in zoo.available_families(TaskType.REGRESSION):
            estimator = zoo.build_estimator(family, TaskType.REGRESSION, {}, random_state=42)
            estimator.fit(X, y)
            assert len(estimator.predict(X)) == len(y), family

    def test_supports_proba_agrees_with_the_fitted_estimator(self, zoo) -> None:
        """A wrong answer here makes the ROC curve silently unavailable."""
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("abc"))
        y = pd.Series((X["a"] > 0).astype(int))

        for family in zoo.available_families(TaskType.BINARY_CLASSIFICATION):
            claimed = zoo.supports_proba(family, TaskType.BINARY_CLASSIFICATION)
            estimator = zoo.build_estimator(
                family, TaskType.BINARY_CLASSIFICATION, {}, random_state=42
            )
            estimator.fit(X, y)
            actual = hasattr(estimator, "predict_proba")
            if claimed:
                assert actual, f"{family.value} claims proba support but has no predict_proba"

    def test_regression_never_claims_proba(self, zoo) -> None:
        for family in zoo.available_families(TaskType.REGRESSION):
            assert zoo.supports_proba(family, TaskType.REGRESSION) is False, family

    def test_random_state_makes_fits_reproducible(self, zoo) -> None:
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
        y = pd.Series((X["a"] + X["b"] > 0).astype(int))

        first = zoo.build_estimator(
            ModelFamily.RANDOM_FOREST, TaskType.BINARY_CLASSIFICATION, {"n_estimators": 12}, random_state=7
        ).fit(X, y)
        second = zoo.build_estimator(
            ModelFamily.RANDOM_FOREST, TaskType.BINARY_CLASSIFICATION, {"n_estimators": 12}, random_state=7
        ).fit(X, y)
        np.testing.assert_array_equal(first.predict(X), second.predict(X))

    def test_agent_supplied_params_reach_the_estimator(self, zoo) -> None:
        estimator = zoo.build_estimator(
            ModelFamily.RANDOM_FOREST,
            TaskType.BINARY_CLASSIFICATION,
            params_to_dict([Param(key="n_estimators", value="17")]),
        )
        params = estimator.get_params()
        assert params.get("n_estimators") == 17 or 17 in params.values()

    def test_nonsense_params_do_not_crash_the_build(self, zoo) -> None:
        """Claude can name a parameter the estimator does not have; that must degrade."""
        estimator = zoo.build_estimator(
            ModelFamily.LOGISTIC,
            TaskType.BINARY_CLASSIFICATION,
            {"a_param_that_does_not_exist": 5, "max_iter": 500},
        )
        assert hasattr(estimator, "fit")

    def test_search_spaces_reference_real_parameters(self, zoo) -> None:
        """A tuning space keyed on a non-existent param fails every trial."""
        for task in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
            for family in zoo.available_families(task):
                space = zoo.default_search_space(family, task)
                assert isinstance(space, dict)
                if not space:
                    continue
                valid = set(zoo.build_estimator(family, task, {}).get_params())
                for key in space:
                    leaf = key.split("__")[-1]
                    assert leaf in valid or key in valid, f"{family.value}: {key} is not a parameter"

    def test_unavailable_family_raises_a_typed_error(self, zoo) -> None:
        from automl_architect.core.errors import AutoMLArchitectError

        with pytest.raises(AutoMLArchitectError):
            zoo.build_estimator(ModelFamily.CATBOOST, TaskType.BINARY_CLASSIFICATION, {})


# ===========================================================================
# cleaning
# ===========================================================================


class TestCleaningOps:
    def test_applies_the_canned_plan(self, cleaning_ops, run_state: RunState) -> None:
        """Drop the leak, drop the id, median-impute the skewed income column."""
        before = len(run_state.working_df)
        result = cleaning_ops.apply_cleaning_plan(run_state, churn_cleaning())

        assert result is not None
        assert run_state.working_df is not None
        assert CHURN_LEAK not in run_state.working_df.columns
        assert "customer_id" not in run_state.working_df.columns
        assert run_state.working_df["annual_income"].isna().sum() == 0
        assert len(run_state.working_df) == before, "no rows should be dropped by this plan"
        assert CHURN_TARGET in run_state.working_df.columns

    def test_returns_the_frame_it_assigned(self, cleaning_ops, run_state: RunState) -> None:
        result = cleaning_ops.apply_cleaning_plan(run_state, churn_cleaning())
        assert result is run_state.working_df or result.equals(run_state.working_df)

    def test_median_imputation_uses_the_observed_median(
        self, cleaning_ops, run_state: RunState
    ) -> None:
        """The whole argument for median-over-mean is wasted if the value is wrong."""
        observed_median = float(run_state.working_df["annual_income"].median())
        missing_mask = run_state.working_df["annual_income"].isna()
        assert missing_mask.sum() > 0

        plan = CleaningPlan(
            decisions=[
                CleaningDecision(
                    action=CleaningAction.IMPUTE_MISSING,
                    columns=["annual_income"],
                    strategy=MissingStrategy.MEDIAN,
                    rationale="skew 2.1",
                )
            ],
            summary="impute only",
        )
        frame = cleaning_ops.apply_cleaning_plan(run_state, plan)
        filled = frame.loc[missing_mask.to_numpy(), "annual_income"]
        assert filled.nunique() == 1
        assert float(filled.iloc[0]) == pytest.approx(observed_median, rel=1e-9)

    def test_mean_imputation_uses_the_observed_mean(
        self, cleaning_ops, run_state: RunState
    ) -> None:
        observed_mean = float(run_state.working_df["annual_income"].mean())
        missing_mask = run_state.working_df["annual_income"].isna()
        plan = CleaningPlan(
            decisions=[
                CleaningDecision(
                    action=CleaningAction.IMPUTE_MISSING,
                    columns=["annual_income"],
                    strategy=MissingStrategy.MEAN,
                    rationale="approximately symmetric",
                )
            ],
            summary="impute only",
        )
        frame = cleaning_ops.apply_cleaning_plan(run_state, plan)
        filled = frame.loc[missing_mask.to_numpy(), "annual_income"]
        assert float(filled.iloc[0]) == pytest.approx(observed_mean, rel=1e-6)

    def test_categorical_missing_becomes_its_own_level(self, cleaning_ops, run_state: RunState) -> None:
        run_state.working_df.loc[:9, "region"] = None
        plan = CleaningPlan(
            decisions=[
                CleaningDecision(
                    action=CleaningAction.IMPUTE_MISSING,
                    columns=["region"],
                    strategy=MissingStrategy.MISSING_CATEGORY,
                    rationale="absence may itself be informative",
                )
            ],
            summary="categorical fill",
        )
        frame = cleaning_ops.apply_cleaning_plan(run_state, plan)
        assert frame["region"].isna().sum() == 0

    def test_duplicate_rows_dropped(self, cleaning_ops, run_state: RunState) -> None:
        run_state.working_df = pd.concat(
            [run_state.working_df, run_state.working_df.head(5)], ignore_index=True
        )
        n_before = len(run_state.working_df)
        plan = CleaningPlan(
            decisions=[
                CleaningDecision(
                    action=CleaningAction.DROP_DUPLICATE_ROWS,
                    rationale="5 exact duplicates inflate any score computed over them",
                    destructive=True,
                )
            ],
            summary="dedupe",
        )
        frame = cleaning_ops.apply_cleaning_plan(run_state, plan)
        assert len(frame) == n_before - 5

    def test_rows_missing_the_target_are_dropped(self, cleaning_ops, run_state: RunState) -> None:
        run_state.working_df.loc[:3, CHURN_TARGET] = np.nan
        plan = CleaningPlan(
            decisions=[
                CleaningDecision(
                    action=CleaningAction.DROP_ROWS_MISSING_TARGET,
                    columns=[CHURN_TARGET],
                    rationale="an unlabelled row cannot contribute to supervised training",
                    destructive=True,
                )
            ],
            summary="drop unlabelled",
        )
        frame = cleaning_ops.apply_cleaning_plan(run_state, plan)
        assert frame[CHURN_TARGET].isna().sum() == 0

    def test_outlier_clipping_keeps_every_row(self, cleaning_ops, run_state: RunState) -> None:
        before = len(run_state.working_df)
        maximum = float(run_state.working_df["total_charges"].max())
        plan = CleaningPlan(
            decisions=[
                CleaningDecision(
                    action=CleaningAction.CLIP_OUTLIERS,
                    columns=["total_charges"],
                    parameters=[Param(key="method", value="iqr")],
                    rationale="the tail is real billing history, so clip rather than delete",
                )
            ],
            summary="clip",
        )
        frame = cleaning_ops.apply_cleaning_plan(run_state, plan)
        assert len(frame) == before
        assert float(frame["total_charges"].max()) <= maximum

    def test_unknown_columns_are_ignored_with_a_warning(
        self, cleaning_ops, run_state: RunState
    ) -> None:
        """A hallucinated column name must not become a KeyError inside pandas."""
        plan = CleaningPlan(
            decisions=[
                CleaningDecision(
                    action=CleaningAction.DROP_COLUMN,
                    columns=["a_column_that_never_existed"],
                    rationale="testing the grounding filter",
                    destructive=True,
                )
            ],
            summary="bad reference",
            columns_to_drop=["another_invented_column"],
        )
        frame = cleaning_ops.apply_cleaning_plan(run_state, plan)
        assert frame is not None
        assert run_state.warnings, "an unknown column reference must be recorded"

    def test_target_survives_a_plan_that_would_drop_it(
        self, cleaning_ops, run_state: RunState
    ) -> None:
        """Dropping the target ends the run; the executor has to refuse."""
        plan = CleaningPlan(
            decisions=[
                CleaningDecision(
                    action=CleaningAction.DROP_COLUMN,
                    columns=[CHURN_TARGET],
                    rationale="a mistake the executor must not honour",
                    destructive=True,
                )
            ],
            summary="drop the target",
            columns_to_drop=[CHURN_TARGET],
        )
        frame = cleaning_ops.apply_cleaning_plan(run_state, plan)
        assert CHURN_TARGET in frame.columns

    def test_empty_plan_is_a_no_op(self, cleaning_ops, run_state: RunState) -> None:
        before = run_state.working_df.shape
        frame = cleaning_ops.apply_cleaning_plan(run_state, CleaningPlan(decisions=[], summary=""))
        assert frame.shape == before

    def test_raw_frame_is_left_intact(self, cleaning_ops, run_state: RunState) -> None:
        """``raw_df`` is the audit copy; cleaning writes to ``working_df``."""
        raw_columns = list(run_state.raw_df.columns)
        cleaning_ops.apply_cleaning_plan(run_state, churn_cleaning())
        assert list(run_state.raw_df.columns) == raw_columns

    def test_applied_actions_are_recorded(self, cleaning_ops, run_state: RunState) -> None:
        """The audit trail is a product feature, not debug output."""
        cleaning_ops.apply_cleaning_plan(run_state, churn_cleaning())
        assert run_state.applied_cleaning or run_state.dropped_columns


# ===========================================================================
# features
# ===========================================================================


@pytest.fixture
def cleaned_state(cleaning_ops, run_state: RunState) -> RunState:
    """A state with the canned cleaning plan already applied."""
    cleaning_ops.apply_cleaning_plan(run_state, churn_cleaning())
    return run_state


def materialise(state: RunState) -> np.ndarray:
    """Turn ``state.feature_frame`` into the numeric matrix an estimator sees.

    ``feature_ops`` deliberately does *not* encode or scale in the frame: anything
    fitted from the feature distribution is deferred into an unfitted sklearn
    pipeline on ``state.preprocessor``, which the trainer fits on the training
    partition only. Encoding early would leak the test set's category set and
    scale into the transform. So "is the output model-ready" is a question about
    the frame *and* the preprocessor together, and this helper answers it the same
    way the trainer does.

    Args:
        state: A state that has been through ``apply_feature_plan``.

    Returns:
        A dense float array of the transformed features.
    """
    frame = state.feature_frame
    columns = state.feature_names or [c for c in frame.columns if c != CHURN_TARGET]
    X = frame[columns]
    if state.preprocessor is not None:
        transformed = state.preprocessor.fit_transform(X)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype="float64")
    return X.to_numpy(dtype="float64")


class TestFeatureOps:
    def test_produces_a_model_ready_matrix(self, feature_ops, cleaned_state: RunState) -> None:
        """Frame plus preprocessor must yield a finite numeric matrix, or nothing fits."""
        feature_ops.apply_feature_plan(cleaned_state, churn_features())
        assert cleaned_state.feature_frame is not None

        matrix = materialise(cleaned_state)
        assert matrix.ndim == 2
        assert matrix.shape[0] == len(cleaned_state.feature_frame)
        assert matrix.shape[1] > 0
        assert np.isfinite(matrix).all(), "the transformed matrix contains NaN or inf"

    def test_encoding_is_deferred_not_skipped(self, feature_ops, cleaned_state: RunState) -> None:
        """Categoricals may stay strings in the frame only if a preprocessor handles them."""
        feature_ops.apply_feature_plan(cleaned_state, churn_features())
        frame = cleaned_state.feature_frame
        columns = cleaned_state.feature_names or list(frame.columns)
        non_numeric = [
            c for c in columns if c in frame and not pd.api.types.is_numeric_dtype(frame[c])
        ]
        if non_numeric:
            assert cleaned_state.preprocessor is not None, (
                f"{non_numeric} are non-numeric and no preprocessor was built"
            )

    def test_row_count_is_preserved(self, feature_ops, cleaned_state: RunState) -> None:
        before = len(cleaned_state.working_df)
        frame = feature_ops.apply_feature_plan(cleaned_state, churn_features())
        assert len(frame) == before

    def test_no_missing_values_reach_an_estimator(self, feature_ops, cleaned_state: RunState) -> None:
        feature_ops.apply_feature_plan(cleaned_state, churn_features())
        matrix = materialise(cleaned_state)
        assert not np.isnan(matrix).any(), "NaNs would break most estimators"

    def test_feature_names_exclude_the_target(self, feature_ops, cleaned_state: RunState) -> None:
        feature_ops.apply_feature_plan(cleaned_state, churn_features())
        assert cleaned_state.feature_names, "feature names drive every later report"
        assert CHURN_TARGET not in cleaned_state.feature_names

    def test_one_hot_expands_the_categorical(self, feature_ops, cleaned_state: RunState) -> None:
        """Three levels must become at least two numeric columns after transformation."""
        plan = FeaturePlan(
            decisions=[
                FeatureDecision(
                    op=FeatureOp.ONE_HOT_ENCODE,
                    input_columns=["contract_type"],
                    rationale="three nominal levels",
                )
            ],
            summary="one hot only",
        )
        feature_ops.apply_feature_plan(cleaned_state, plan)
        assert "contract_type" in cleaned_state.feature_names

        matrix = materialise(cleaned_state)
        assert matrix.shape[1] >= len(cleaned_state.feature_names) + 1, (
            "one-hot encoding did not widen the transformed matrix"
        )
        assert np.isfinite(matrix).all()

    def test_unseen_category_encodes_without_failing(
        self, feature_ops, cleaned_state: RunState
    ) -> None:
        """A level present only in the test split must map to zeros, not raise."""
        feature_ops.apply_feature_plan(
            cleaned_state,
            FeaturePlan(
                decisions=[
                    FeatureDecision(
                        op=FeatureOp.ONE_HOT_ENCODE,
                        input_columns=["payment_method"],
                        rationale="crypto_wallet holds 0.5% of rows",
                    )
                ],
                summary="one hot",
            ),
        )
        if cleaned_state.preprocessor is None:
            pytest.skip("no deferred preprocessor to exercise")

        columns = cleaned_state.feature_names
        frame = cleaned_state.feature_frame
        train = frame[frame["payment_method"] != "crypto_wallet"]
        held_out = frame[frame["payment_method"] == "crypto_wallet"]
        if not len(held_out):
            pytest.skip("the rare level did not survive cleaning")

        cleaned_state.preprocessor.fit(train[columns])
        transformed = cleaned_state.preprocessor.transform(held_out[columns])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        assert np.isfinite(np.asarray(transformed, dtype="float64")).all()

    def test_ratio_creates_a_new_column(self, feature_ops, cleaned_state: RunState) -> None:
        plan = FeaturePlan(
            decisions=[
                FeatureDecision(
                    op=FeatureOp.RATIO,
                    input_columns=["total_charges", "tenure_months"],
                    output_name_hint="avg_charge_per_month",
                    rationale="isolates spend level from account age",
                )
            ],
            summary="ratio only",
        )
        before = set(cleaned_state.working_df.columns)
        frame = feature_ops.apply_feature_plan(cleaned_state, plan)
        assert set(frame.columns) - before, "the ratio op produced no new column"

    def test_log_transform_is_finite(self, feature_ops, cleaned_state: RunState) -> None:
        """log1p rather than log, or a zero becomes -inf and every model fails.

        churn.csv contains ``total_charges`` values at 0.0, which is exactly the
        input that separates the two implementations.
        """
        cleaned_state.working_df.loc[:4, "total_charges"] = 0.0
        plan = FeaturePlan(
            decisions=[
                FeatureDecision(
                    op=FeatureOp.LOG_TRANSFORM,
                    input_columns=["total_charges"],
                    rationale="right-skewed and non-negative",
                )
            ],
            summary="log only",
        )
        feature_ops.apply_feature_plan(cleaned_state, plan)
        assert np.isfinite(materialise(cleaned_state)).all()

    def test_date_decompose_yields_calendar_parts(self, feature_ops, cleaned_state: RunState) -> None:
        plan = FeaturePlan(
            decisions=[
                FeatureDecision(
                    op=FeatureOp.DATE_DECOMPOSE,
                    input_columns=["signup_date"],
                    rationale="cohort effects are calendar-driven",
                )
            ],
            summary="date parts",
        )
        frame = feature_ops.apply_feature_plan(cleaned_state, plan)
        new_columns = [c for c in frame.columns if c.startswith("signup_date")]
        assert new_columns, "no calendar features were derived"

    def test_unknown_columns_are_ignored_with_a_warning(
        self, feature_ops, cleaned_state: RunState
    ) -> None:
        plan = FeaturePlan(
            decisions=[
                FeatureDecision(
                    op=FeatureOp.LOG_TRANSFORM,
                    input_columns=["a_column_the_model_imagined"],
                    rationale="testing the grounding filter",
                )
            ],
            summary="bad reference",
        )
        frame = feature_ops.apply_feature_plan(cleaned_state, plan)
        assert frame is not None
        assert cleaned_state.warnings

    def test_op_on_the_wrong_dtype_degrades(self, feature_ops, cleaned_state: RunState) -> None:
        """Log of a string column is a mistake, not a reason to end the run."""
        plan = FeaturePlan(
            decisions=[
                FeatureDecision(
                    op=FeatureOp.LOG_TRANSFORM,
                    input_columns=["region"],
                    rationale="a dtype mistake the executor has to survive",
                )
            ],
            summary="wrong dtype",
        )
        frame = feature_ops.apply_feature_plan(cleaned_state, plan)
        assert frame is not None

    def test_empty_plan_still_yields_a_usable_frame(self, feature_ops, cleaned_state: RunState) -> None:
        """With no feature decisions, encoding still has to happen or nothing can fit."""
        frame = feature_ops.apply_feature_plan(
            cleaned_state, FeaturePlan(decisions=[], summary="")
        )
        assert frame is not None
        assert len(frame) == len(cleaned_state.working_df)

    def test_applied_features_are_recorded(self, feature_ops, cleaned_state: RunState) -> None:
        feature_ops.apply_feature_plan(cleaned_state, churn_features())
        assert cleaned_state.applied_features


# ===========================================================================
# splitter
# ===========================================================================


@pytest.fixture
def featured_state(feature_ops, cleaned_state: RunState) -> RunState:
    """A state with cleaning and feature engineering applied, ready to split."""
    feature_ops.apply_feature_plan(cleaned_state, churn_features())
    return cleaned_state


class TestSplitter:
    def test_strategy_is_named_and_argued(self, splitter, featured_state: RunState) -> None:
        strategy, rationale = splitter.resolve_split_strategy(featured_state)
        assert isinstance(strategy, str) and strategy.strip()
        assert isinstance(rationale, str) and len(rationale) > 20, "the strategy needs a reason"

    def test_imbalanced_classification_gets_stratified(self, splitter, featured_state: RunState) -> None:
        """At 26% positive, an unstratified fold moves AUC more than the models do."""
        strategy, _ = splitter.resolve_split_strategy(featured_state)
        assert "strat" in strategy.lower(), strategy

    def test_time_series_gets_a_temporal_strategy(self, splitter, featured_state: RunState) -> None:
        featured_state.problem.task_type = TaskType.TIME_SERIES_FORECASTING
        featured_state.problem.temporal_column = "signup_date"
        strategy, _ = splitter.resolve_split_strategy(featured_state)
        assert any(token in strategy.lower() for token in ("time", "temporal", "expanding", "chronolog"))

    def test_group_column_gets_a_grouped_strategy(self, splitter, featured_state: RunState) -> None:
        featured_state.problem.group_column = "region"
        strategy, _ = splitter.resolve_split_strategy(featured_state)
        assert "group" in strategy.lower(), strategy

    def test_splits_partition_the_rows(self, splitter, featured_state: RunState) -> None:
        splits = splitter.make_splits(featured_state)
        assert isinstance(splits, DataSplits)
        sizes = splits.sizes()
        assert sizes["train"] > 0
        assert sizes["train"] + sizes["validation"] + sizes["test"] == len(featured_state.feature_frame)

    def test_no_row_appears_in_two_splits(self, splitter, featured_state: RunState) -> None:
        """An overlapping split turns a test score into a training score."""
        splits = splitter.make_splits(featured_state)
        frames = [f for f in (splits.X_train, splits.X_valid, splits.X_test) if f is not None]
        indices = [set(f.index) for f in frames if hasattr(f, "index")]
        for i, left in enumerate(indices):
            for right in indices[i + 1 :]:
                assert not (left & right), "splits overlap"

    def test_features_and_labels_align(self, splitter, featured_state: RunState) -> None:
        splits = splitter.make_splits(featured_state)
        for X, y in ((splits.X_train, splits.y_train), (splits.X_test, splits.y_test)):
            if X is None:
                continue
            assert len(X) == len(y)

    def test_target_is_not_a_feature(self, splitter, featured_state: RunState) -> None:
        """Leaving the label in X makes every score 1.0 and every insight false."""
        splits = splitter.make_splits(featured_state)
        assert CHURN_TARGET not in list(splits.X_train.columns)

    def test_column_sets_match_across_splits(self, splitter, featured_state: RunState) -> None:
        splits = splitter.make_splits(featured_state)
        expected = list(splits.X_train.columns)
        for frame in (splits.X_valid, splits.X_test):
            if frame is not None and len(frame):
                assert list(frame.columns) == expected

    def test_class_ratio_survives_the_split(self, splitter, featured_state: RunState) -> None:
        splits = splitter.make_splits(featured_state)
        train_rate = float(pd.Series(splits.y_train).mean())
        test_rate = float(pd.Series(splits.y_test).mean())
        assert abs(train_rate - test_rate) < 0.05, f"{train_rate:.3f} vs {test_rate:.3f}"

    def test_test_fraction_honours_the_config(self, splitter, featured_state: RunState) -> None:
        splits = splitter.make_splits(featured_state)
        n = len(featured_state.feature_frame)
        assert splits.sizes()["test"] == pytest.approx(
            featured_state.config.test_size * n, rel=0.15
        )

    def test_splitting_is_reproducible(self, splitter, featured_state: RunState) -> None:
        first = splitter.make_splits(featured_state)
        second = splitter.make_splits(featured_state)
        assert list(first.X_test.index) == list(second.X_test.index)

    def test_strategy_and_rationale_land_on_the_splits(self, splitter, featured_state: RunState) -> None:
        """The Evaluation Agent reads these to judge whether a score is trustworthy."""
        splits = splitter.make_splits(featured_state)
        assert splits.strategy.strip()
        assert splits.rationale.strip()

    def test_temporal_split_does_not_train_on_the_future(
        self, splitter, feature_ops, cleaning_ops, sales_df: pd.DataFrame, settings
    ) -> None:
        from automl_architect.core.events import EventBus
        from automl_architect.core.schemas import ProblemDefinition

        from .conftest import SALES_CSV, SALES_TARGET, make_run_config, stub_profile

        frame = sales_df[sales_df["store_id"] == "S001"].reset_index(drop=True)
        config = make_run_config(SALES_CSV, target_column=SALES_TARGET)
        state = RunState(config=config, settings=settings, bus=EventBus(run_id=config.run_id))
        state.raw_df = frame.copy()
        state.working_df = frame.copy()
        state.profile = stub_profile(frame, target=SALES_TARGET)
        state.problem = ProblemDefinition(
            task_type=TaskType.TIME_SERIES_FORECASTING,
            target_column=SALES_TARGET,
            temporal_column="date",
            rationale="a daily date column with trend and weekly seasonality",
            confidence="high",
            primary_metric="rmse",
            metric_rationale="errors are in units and comparable across days",
            business_objective="order the right stock",
        )
        state.feature_frame = frame.copy()
        state.feature_names = [c for c in frame.columns if c != SALES_TARGET]

        splits = splitter.make_splits(state)
        if splits.X_test is None or not len(splits.X_test):
            pytest.skip("splitter produced no test split for this configuration")
        train_dates = pd.to_datetime(frame.loc[splits.X_train.index, "date"])
        test_dates = pd.to_datetime(frame.loc[splits.X_test.index, "date"])
        assert train_dates.max() <= test_dates.min(), "a temporal split must not straddle time"


# ===========================================================================
# trainer
# ===========================================================================


@pytest.fixture
def ready_state(splitter, featured_state: RunState) -> RunState:
    """A fully prepared state: features built, splits made, candidates chosen."""
    featured_state.splits = splitter.make_splits(featured_state)
    selection = churn_model_selection()
    selection.candidates = [
        c
        for c in selection.candidates
        if c.family in {ModelFamily.BASELINE_DUMMY, ModelFamily.LOGISTIC, ModelFamily.RANDOM_FOREST}
    ]
    featured_state.model_selection = selection
    return featured_state


class TestTrainer:
    def test_trains_every_candidate(self, trainer, ready_state: RunState) -> None:
        log = trainer.run_experiments(ready_state)
        assert isinstance(log, ExperimentLog)
        assert log.results, "no experiments were run"
        trained = {r.family for r in log.results}
        assert ModelFamily.BASELINE_DUMMY in trained, "the baseline floor is mandatory"

    def test_leaderboard_resolves_to_a_winner(self, trainer, ready_state: RunState) -> None:
        log = trainer.run_experiments(ready_state)
        assert log.best_experiment_id
        best = log.best()
        assert best is not None and not best.failed
        assert log.primary_metric

    def test_winner_is_actually_the_best_score(self, trainer, ready_state: RunState) -> None:
        """A wrong direction here recommends the worst model with full confidence."""
        log = trainer.run_experiments(ready_state)
        scored = [r for r in log.results if not r.failed and r.primary_score is not None]
        if len(scored) < 2:
            pytest.skip("need at least two successful experiments to compare")
        chooser = max if log.higher_is_better else min
        assert log.best().primary_score == pytest.approx(
            chooser(r.primary_score for r in scored)
        )

    def test_every_result_is_populated(self, trainer, ready_state: RunState) -> None:
        log = trainer.run_experiments(ready_state)
        for result in log.results:
            if result.failed:
                assert result.error, "a failed experiment must say why"
                continue
            assert result.primary_score is not None
            assert result.primary_metric
            assert result.metrics, f"{result.family.value} recorded no metrics"
            assert result.train_seconds >= 0.0
            assert result.n_features_in > 0

    def test_experiment_cap_is_respected(self, trainer, ready_state: RunState) -> None:
        """``max_experiments`` is the operator's compute budget, not a suggestion."""
        ready_state.config.max_experiments = 2
        log = trainer.run_experiments(ready_state)
        non_baseline = [r for r in log.results if not r.is_baseline]
        assert len(non_baseline) <= 2

    def test_one_broken_family_does_not_end_the_run(self, trainer, ready_state: RunState) -> None:
        """An unavailable family must be recorded as failed or skipped, never fatal."""
        from automl_architect.core.schemas import ModelCandidate

        ready_state.model_selection.candidates.append(
            ModelCandidate(
                family=ModelFamily.CATBOOST,
                rank=99,
                suitability="fair",
                rationale="deliberately unavailable in this environment",
            )
        )
        log = trainer.run_experiments(ready_state)
        assert any(not r.failed for r in log.results), "every experiment failed"
        catboost = [r for r in log.results if r.family is ModelFamily.CATBOOST]
        if catboost:
            assert catboost[0].failed and catboost[0].error

    def test_baseline_is_marked_as_such(self, trainer, ready_state: RunState) -> None:
        log = trainer.run_experiments(ready_state)
        baselines = [r for r in log.results if r.family is ModelFamily.BASELINE_DUMMY]
        assert baselines and baselines[0].is_baseline

    def test_final_fit_returns_a_usable_estimator(self, trainer, ready_state: RunState) -> None:
        trainer.run_experiments(ready_state)
        model = trainer.fit_final_model(ready_state, ModelFamily.RANDOM_FOREST, {"n_estimators": 20})
        assert hasattr(model, "predict")
        predictions = model.predict(ready_state.splits.X_test)
        assert len(predictions) == len(ready_state.splits.X_test)

    def test_regression_training_reports_error_metrics(
        self, trainer, splitter, feature_ops, cleaning_ops, regression_state: RunState
    ) -> None:
        cleaning_ops.apply_cleaning_plan(
            regression_state,
            CleaningPlan(
                decisions=[
                    CleaningDecision(
                        action=CleaningAction.IMPUTE_MISSING,
                        columns=["lot_size_sqft"],
                        strategy=MissingStrategy.MEDIAN,
                        rationale="5.5% unrecorded lot sizes on a skewed column",
                    ),
                    CleaningDecision(
                        action=CleaningAction.DROP_COLUMN,
                        columns=["property_id"],
                        rationale="surrogate key",
                        destructive=True,
                    ),
                ],
                summary="minimal",
                columns_to_drop=["property_id"],
            ),
        )
        feature_ops.apply_feature_plan(
            regression_state,
            FeaturePlan(
                decisions=[
                    FeatureDecision(
                        op=FeatureOp.ONE_HOT_ENCODE,
                        input_columns=["neighborhood"],
                        rationale="six nominal levels",
                    )
                ],
                summary="encode",
            ),
        )
        regression_state.splits = splitter.make_splits(regression_state)
        selection = churn_model_selection()
        selection.candidates = [
            c
            for c in selection.candidates
            if c.family in {ModelFamily.BASELINE_DUMMY, ModelFamily.RANDOM_FOREST}
        ]
        regression_state.model_selection = selection

        log = trainer.run_experiments(regression_state)
        successful = [r for r in log.results if not r.failed]
        assert successful
        assert log.higher_is_better is False or log.primary_metric in {"r2"}
        assert HOUSE_TARGET not in list(regression_state.splits.X_train.columns)


# ===========================================================================
# tuner
# ===========================================================================


class TestTuner:
    def test_declining_to_tune_is_recorded_not_silent(self, tuner, ready_state: RunState) -> None:
        decision = TuningDecision(
            worthwhile=False,
            rationale="the baseline gap is 0.002 AUC, far inside fold noise",
            method=TuningMethod.NONE,
        )
        result = tuner.run_tuning(ready_state, decision)
        assert isinstance(result, TuningResult)
        assert result.ran is False
        assert result.skipped_reason, "a skipped tuning run must explain itself"

    def test_random_search_improves_or_reports_no_gain(
        self, trainer, tuner, ready_state: RunState
    ) -> None:
        trainer.run_experiments(ready_state)
        decision = TuningDecision(
            worthwhile=True,
            rationale="3,000 rows makes a short search cheap",
            method=TuningMethod.RANDOM_SEARCH,
            method_rationale="a four-parameter space does not need Optuna",
            target_family=ModelFamily.RANDOM_FOREST,
            n_trials=4,
            timeout_seconds=60,
        )
        result = tuner.run_tuning(ready_state, decision)
        if not result.ran:
            assert result.skipped_reason or result.error
            return
        assert result.family is ModelFamily.RANDOM_FOREST
        assert result.n_trials_completed >= 1
        assert result.best_params, "a completed search must report its winning params"
        assert result.best_score is not None
        assert result.seconds >= 0.0

    def test_timeout_is_respected(self, trainer, tuner, ready_state: RunState) -> None:
        """A tuning budget the operator set must bound wall-clock, not just trials."""
        trainer.run_experiments(ready_state)
        decision = TuningDecision(
            worthwhile=True,
            rationale="testing the budget guard",
            method=TuningMethod.RANDOM_SEARCH,
            target_family=ModelFamily.RANDOM_FOREST,
            n_trials=500,
            timeout_seconds=5,
        )
        result = tuner.run_tuning(ready_state, decision)
        assert result.seconds < 90.0

    def test_unavailable_target_family_degrades(self, tuner, ready_state: RunState) -> None:
        decision = TuningDecision(
            worthwhile=True,
            rationale="targets a family this environment cannot build",
            method=TuningMethod.RANDOM_SEARCH,
            target_family=ModelFamily.CATBOOST,
            n_trials=2,
            timeout_seconds=10,
        )
        result = tuner.run_tuning(ready_state, decision)
        assert result.ran is False or result.error
        assert result.skipped_reason or result.error


# ===========================================================================
# explainer
# ===========================================================================


@pytest.fixture
def fitted_state(trainer, ready_state: RunState) -> RunState:
    """A state with experiments run and a best model fitted on the training split."""
    ready_state.experiments = trainer.run_experiments(ready_state)
    if ready_state.best_model is None:
        ready_state.best_model = trainer.fit_final_model(
            ready_state, ModelFamily.RANDOM_FOREST, {"n_estimators": 30}
        )
    return ready_state


class TestExplainer:
    def test_returns_normalised_attributions(self, explainer, fitted_state: RunState) -> None:
        """The contract says importances sum to 1.0, and reports quote them as percentages."""
        report = explainer.compute_explanations(fitted_state)
        assert isinstance(report, ExplainabilityReport)
        assert report.global_attributions, "no attributions were produced"
        total = sum(a.importance for a in report.global_attributions)
        assert total == pytest.approx(1.0, abs=0.02), f"attributions sum to {total}"

    def test_attributions_name_real_features(self, explainer, fitted_state: RunState) -> None:
        report = explainer.compute_explanations(fitted_state)
        known = set(map(str, fitted_state.splits.X_train.columns))
        for attribution in report.global_attributions:
            assert attribution.feature in known, f"{attribution.feature} is not a model input"

    def test_shap_flag_matches_reality(self, explainer, fitted_state: RunState) -> None:
        """Claiming SHAP ran when it did not misleads the method notes in the report."""
        report = explainer.compute_explanations(fitted_state)
        if report.shap_available:
            pytest.importorskip("shap")
        assert isinstance(report.shap_available, bool)

    def test_plain_language_explanations_are_produced(self, explainer, fitted_state: RunState) -> None:
        report = explainer.compute_explanations(fitted_state)
        assert report.plain_language_explanations or report.permutation_importance

    def test_missing_model_degrades(self, explainer, ready_state: RunState) -> None:
        """No fitted model means no explanation, not an exception mid-report."""
        ready_state.best_model = None
        ready_state.best_pipeline = None
        report = explainer.compute_explanations(ready_state)
        assert isinstance(report, ExplainabilityReport)


# ===========================================================================
# diagnostics
# ===========================================================================


class TestDiagnostics:
    REQUIRED = (
        "bias_variance",
        "calibration",
        "confidence_intervals",
        "fairness",
        "residual_stats",
        "learning_curve",
        "error_examples",
    )

    def test_bundle_exposes_the_contract(self, diagnostics_mod, fitted_state: RunState) -> None:
        bundle = diagnostics_mod.compute_diagnostics(fitted_state)
        for attribute in self.REQUIRED:
            assert hasattr(bundle, attribute), f"DiagnosticsBundle is missing {attribute}"

    def test_bundle_field_types(self, diagnostics_mod, fitted_state: RunState) -> None:
        from automl_architect.core.schemas import (
            BiasVarianceDiagnosis,
            CalibrationDiagnosis,
            ConfidenceInterval,
            FairnessSlice,
        )

        bundle = diagnostics_mod.compute_diagnostics(fitted_state)
        assert isinstance(bundle.bias_variance, BiasVarianceDiagnosis)
        assert isinstance(bundle.calibration, CalibrationDiagnosis)
        assert all(isinstance(c, ConfidenceInterval) for c in bundle.confidence_intervals)
        assert all(isinstance(f, FairnessSlice) for f in bundle.fairness)
        assert isinstance(bundle.residual_stats, dict)
        assert isinstance(bundle.learning_curve, dict)
        assert all(isinstance(e, str) for e in bundle.error_examples)

    def test_bias_variance_verdict_follows_the_gap(self, diagnostics_mod, fitted_state: RunState) -> None:
        bundle = diagnostics_mod.compute_diagnostics(fitted_state)
        diagnosis = bundle.bias_variance
        assert diagnosis.verdict in {"underfitting", "good_fit", "overfitting", "inconclusive"}
        if diagnosis.train_score is not None and diagnosis.test_score is not None:
            assert diagnosis.gap is not None
            assert diagnosis.gap == pytest.approx(
                abs(diagnosis.train_score - diagnosis.test_score), abs=0.05
            )

    def test_confidence_intervals_bracket_their_estimates(
        self, diagnostics_mod, fitted_state: RunState
    ) -> None:
        bundle = diagnostics_mod.compute_diagnostics(fitted_state)
        for interval in bundle.confidence_intervals:
            assert interval.lower <= interval.point_estimate <= interval.upper
            assert 0.0 < interval.level < 1.0

    def test_learning_curve_series_are_the_same_length(
        self, diagnostics_mod, fitted_state: RunState
    ) -> None:
        bundle = diagnostics_mod.compute_diagnostics(fitted_state)
        lengths = {len(v) for v in bundle.learning_curve.values() if isinstance(v, list)}
        assert len(lengths) <= 1, f"learning-curve series have mismatched lengths: {lengths}"

    def test_fairness_slices_reference_configured_attributes(
        self, diagnostics_mod, fitted_state: RunState
    ) -> None:
        fitted_state.config.fairness_attributes = ["region"]
        bundle = diagnostics_mod.compute_diagnostics(fitted_state)
        for slice_ in bundle.fairness:
            assert slice_.attribute in fitted_state.config.fairness_attributes
            assert slice_.n_rows > 0

    def test_to_prompt_renders_everything_as_text(self, diagnostics_mod, fitted_state: RunState) -> None:
        """This string is the only thing the Evaluation Agent sees, so it must be full."""
        rendered = diagnostics_mod.compute_diagnostics(fitted_state).to_prompt()
        assert isinstance(rendered, str)
        assert len(rendered) > 100, "to_prompt() is too short to carry the diagnosis"
        assert any(char.isdigit() for char in rendered), "no measured numbers in the digest"

    def test_diagnostics_without_a_model_degrades(self, diagnostics_mod, ready_state: RunState) -> None:
        ready_state.best_model = None
        ready_state.best_pipeline = None
        bundle = diagnostics_mod.compute_diagnostics(ready_state)
        assert isinstance(bundle.to_prompt(), str)


# ===========================================================================
# cross-module invariant
# ===========================================================================


def test_pipeline_preserves_the_label_end_to_end(
    cleaning_ops, feature_ops, splitter, run_state: RunState
) -> None:
    """Cleaning, encoding, and splitting must not reorder rows away from their labels.

    A silent misalignment here produces a model that scores near chance for
    reasons no metric can explain, so it is worth an explicit end-to-end check.
    """
    original = run_state.working_df[[CHURN_TARGET]].copy()
    cleaning_ops.apply_cleaning_plan(run_state, churn_cleaning())
    frame = feature_ops.apply_feature_plan(run_state, churn_features())

    if CHURN_TARGET in frame.columns and len(frame) == len(original):
        pd.testing.assert_series_equal(
            frame[CHURN_TARGET].reset_index(drop=True).astype("int64"),
            original[CHURN_TARGET].reset_index(drop=True).astype("int64"),
            check_names=False,
        )

    splits = splitter.make_splits(run_state)
    reassembled: list[Any] = []
    for X, y in (
        (splits.X_train, splits.y_train),
        (splits.X_valid, splits.y_valid),
        (splits.X_test, splits.y_test),
    ):
        if X is None or not len(X):
            continue
        reassembled.append(pd.Series(np.asarray(y), index=X.index))
    combined = pd.concat(reassembled).sort_index()
    expected = frame[CHURN_TARGET].loc[combined.index]
    np.testing.assert_array_equal(combined.to_numpy(), expected.to_numpy())


class TestLiveRunRegressions:
    """Defects a real end-to-end run exposed that the offline suite did not.

    Both were invisible to the fake-LLM tests because both depended on a real
    agent choosing a plausible-but-awkward value: a category-frequency floor
    expressed as a count, and a hyperparameter named the way the literature
    names it rather than the way the installed sklearn spells it.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (20, 20),        # the exact value that failed the live run
            (20.0, 20),      # float-typed count: the actual failure mode
            (1, 1),
            (1.0, 1),
            (0.05, 0.05),    # a genuine proportion must stay a float
            (0.5, 0.5),
            (None, None),
            ("nonsense", None),
            (True, None),    # bool is an int subclass, but not a frequency
            (0, None),
            (-5, None),
            (2.5, 2),        # fractional count, rounded with a warning
        ],
    )
    def test_min_frequency_is_coerced_to_something_sklearn_accepts(
        self, raw: Any, expected: Any
    ) -> None:
        """``OneHotEncoder`` overloads this parameter on type, so type is meaning.

        An ``int`` >= 1 is an absolute row count and a ``float`` in (0, 1) is a
        proportion; ``20.0`` is neither. Because the encoder lives in the shared
        preprocessor, one bad value failed every candidate model in the live run,
        the baseline included, and the run ended with ``NoViableModelError``.
        """
        from sklearn.preprocessing import OneHotEncoder

        from automl_architect.execution.feature_ops import _coerce_min_frequency

        got = _coerce_min_frequency(raw, on_invalid=lambda _msg: None)
        assert got == expected and type(got) is type(expected)

        # The assertion that matters is sklearn's own, not our restatement of it.
        OneHotEncoder(
            min_frequency=got, handle_unknown="infrequent_if_exist"
        )._validate_params()

    @pytest.mark.parametrize(
        ("params", "expect_l1_ratio", "expect_c"),
        [
            ({"penalty": "l2", "C": 0.5}, 0.0, 0.5),
            ({"penalty": "l1"}, 1.0, 1.0),
            ({"penalty": "elasticnet"}, 0.5, 1.0),
            ({"penalty": None}, 0.0, math.inf),
            ({"C": 2.0}, 0.0, 2.0),
        ],
    )
    def test_deprecated_penalty_is_translated_not_dropped(
        self, params: dict, expect_l1_ratio: float, expect_c: float
    ) -> None:
        """Agents name hyperparameters from a vocabulary that lags the library.

        sklearn deprecated ``penalty`` in 1.8 and removes it in 1.10. Passing it
        through warns once per fold today and breaks outright tomorrow; dropping
        it would silently discard the agent's intent. So it is translated, and
        an explicitly supplied ``C`` survives the translation.
        """
        from sklearn.datasets import make_classification

        from automl_architect.core.schemas import ModelFamily, TaskType
        from automl_architect.execution.model_zoo import build_estimator

        estimator = build_estimator(
            ModelFamily.LOGISTIC, TaskType.BINARY_CLASSIFICATION, dict(params), random_state=0
        )
        resolved = estimator.get_params()
        assert resolved.get("l1_ratio") == expect_l1_ratio
        assert resolved.get("C") == expect_c

        X, y = make_classification(n_samples=120, n_features=5, random_state=0)
        estimator.set_params(max_iter=400)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            estimator.fit(X, y)  # a surviving deprecation fails here

    @pytest.mark.parametrize(
        "family",
        [
            ModelFamily.LOGISTIC,
            ModelFamily.SVM,
            ModelFamily.KNN,
            ModelFamily.RIDGE,
        ],
    )
    def test_scale_sensitive_families_always_get_a_scaler(self, family: Any) -> None:
        """Scaling is the model's requirement, not the feature plan's preference.

        A live run trained logistic regression on raw-magnitude inputs and logged
        ``lbfgs failed to converge after 2000 iterations``, because the Feature
        Agent's scaling ops had been dropped. Whether to engineer a scaled feature
        is a judgement call; a penalty term or a distance metric on unscaled
        inputs is dominated by whichever column has the largest units, so the
        trainer guarantees it regardless of what the plan asked for.
        """
        from sklearn.pipeline import Pipeline

        from automl_architect.core.schemas import (
            DataSource,
            RunConfig,
            SourceKind,
            TaskType,
        )
        from automl_architect.core.state import RunState
        from automl_architect.execution.model_zoo import build_estimator
        from automl_architect.execution.trainer import wrap_with_preprocessor

        state = RunState(config=RunConfig(source=DataSource(kind=SourceKind.CSV, uri="x.csv")))
        estimator = build_estimator(family, TaskType.BINARY_CLASSIFICATION, {}, random_state=0)
        pipeline = wrap_with_preprocessor(state, estimator, family)

        assert isinstance(pipeline, Pipeline)
        assert "requires_scale" in dict(pipeline.steps), (
            f"{family.value} is scale-sensitive but received no scaler"
        )

    @pytest.mark.parametrize(
        "family",
        [ModelFamily.RANDOM_FOREST, ModelFamily.HIST_GRADIENT_BOOSTING, ModelFamily.BASELINE_DUMMY],
    )
    def test_tree_families_are_not_scaled(self, family: Any) -> None:
        """Splitting on a threshold is scale-invariant; scaling would only cost time."""
        from sklearn.pipeline import Pipeline

        from automl_architect.core.schemas import (
            DataSource,
            RunConfig,
            SourceKind,
            TaskType,
        )
        from automl_architect.core.state import RunState
        from automl_architect.execution.model_zoo import build_estimator
        from automl_architect.execution.trainer import wrap_with_preprocessor

        state = RunState(config=RunConfig(source=DataSource(kind=SourceKind.CSV, uri="x.csv")))
        estimator = build_estimator(family, TaskType.BINARY_CLASSIFICATION, {}, random_state=0)
        pipeline = wrap_with_preprocessor(state, estimator, family)

        steps = dict(pipeline.steps) if isinstance(pipeline, Pipeline) else {}
        assert "requires_scale" not in steps

    def test_badly_scaled_input_now_converges(self) -> None:
        """The behavioural half: the warning the live run emitted must be gone."""
        from sklearn.datasets import make_classification

        from automl_architect.core.schemas import (
            DataSource,
            RunConfig,
            SourceKind,
            TaskType,
        )
        from automl_architect.core.state import RunState
        from automl_architect.execution.model_zoo import build_estimator
        from automl_architect.execution.trainer import wrap_with_preprocessor

        X, y = make_classification(n_samples=400, n_features=6, random_state=0)
        X[:, 0] *= 250_000.0
        X[:, 1] *= 0.00004

        state = RunState(config=RunConfig(source=DataSource(kind=SourceKind.CSV, uri="x.csv")))
        estimator = build_estimator(
            ModelFamily.LOGISTIC, TaskType.BINARY_CLASSIFICATION, {}, random_state=0
        )
        pipeline = wrap_with_preprocessor(state, estimator, ModelFamily.LOGISTIC)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X, y)

        assert not [w for w in caught if "converge" in str(w.message).lower()]
