"""Fold-local, exportable preparation for raw tabular inputs.

All learned values are fitted on the input to fit/fit_transform. transform
replays frozen mappings and uses the fitted sklearn transformer. It never fits
on prediction rows. Row-removing and history-dependent actions are excluded
from this estimator because sklearn pipelines must preserve row alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..config import Settings
from ..core.schemas import (
    CleaningAction,
    CleaningPlan,
    FeatureOp,
    FeaturePlan,
    MissingStrategy,
    RunConfig,
)
from ..core.state import RunState
from .cleaning_ops import apply_cleaning_plan
from .feature_ops import apply_feature_plan

ROW_ACTIONS = {
    CleaningAction.DROP_DUPLICATE_ROWS,
    CleaningAction.DROP_ROWS_MISSING_TARGET,
    CleaningAction.REMOVE_OUTLIER_ROWS,
}
HISTORY_OPS = {FeatureOp.LAG, FeatureOp.ROLLING, FeatureOp.DIFF, FeatureOp.EXPANDING}
SEQUENTIAL_FILL = {
    MissingStrategy.FORWARD_FILL,
    MissingStrategy.BACKWARD_FILL,
    MissingStrategy.INTERPOLATE,
    MissingStrategy.DROP_ROWS,
}


class PlanPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, config, problem=None, cleaning=None, features=None):
        self.config = config
        self.problem = problem
        self.cleaning = cleaning
        self.features = features

    def _state(self, X, y=None):
        config = RunConfig.model_validate(self.config)
        state = RunState(config=config, settings=Settings(_env_file=None, offline=True))
        state.problem = self.problem
        state.raw_df = X.copy()
        if y is not None and state.target:
            state.raw_df[state.target] = np.asarray(y)
        state.working_df = state.raw_df.copy()
        return state

    def _fit_frame(self, X, y):
        from ..profiling.profiler import profile_dataframe

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Preparation requires a dataframe with named columns.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        state = self._state(X, y)
        # Never reuse the outer run's profile inside a validation fold.
        state.profile = profile_dataframe(
            state.raw_df, target=state.target, settings=state.settings
        )
        self.profile_ = state.profile
        cleaning = self.cleaning or CleaningPlan(decisions=[], summary="Default preparation")
        decisions = []
        self.warnings_ = []
        for decision in cleaning.decisions:
            if decision.action in ROW_ACTIONS:
                self.warnings_.append(
                    f"{decision.action.value}: row removal is excluded from fitted preprocessing."
                )
                continue
            if decision.strategy in SEQUENTIAL_FILL:
                self.warnings_.append(
                    f"{decision.strategy.value}: replaced with training-fitted median/mode imputation."
                )
                decision = decision.model_copy(update={"strategy": None})
            decisions.append(decision)
        self.cleaning_ = cleaning.model_copy(update={"decisions": decisions})
        apply_cleaning_plan(state, self.cleaning_)
        self.cleaning_metadata_ = {
            k: v for k, v in state.extras.items() if k.startswith("cleaning_")
        }
        features = self.features or FeaturePlan(decisions=[], summary="Default features")
        allowed = []
        for decision in features.decisions:
            if decision.op in HISTORY_OPS:
                self.warnings_.append(
                    f"{decision.op.value}: requires an explicit historical-data contract; excluded from exportable preparation."
                )
                continue
            if decision.op == FeatureOp.TARGET_ENCODE and (
                getattr(self.problem, "temporal_column", None)
                or getattr(self.problem, "group_column", None)
            ):
                self.warnings_.append(
                    "target_encode: excluded for temporal/grouped data; using ordinary categorical encoding."
                )
                continue
            allowed.append(decision)
        self.features_ = features.model_copy(update={"decisions": allowed})
        frame = apply_feature_plan(state, self.features_)
        self.columns_ = list(state.feature_names)
        self.encodings_ = state.extras.get("feature_encodings", {})
        self.transformer_ = state.preprocessor
        self.applied_cleaning_ = list(state.applied_cleaning)
        self.applied_features_ = list(state.applied_features)
        self.warnings_.extend(state.warnings)
        if not self.columns_:
            raise ValueError("No usable features remain after validated preparation.")
        return frame.reindex(columns=self.columns_)

    def fit(self, X, y=None):
        frame = self._fit_frame(X, y)
        self.transformer_.fit(frame, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        frame = self._fit_frame(X, y)
        # This frame contains cross-fitted target encodings; calling transform
        # here would incorrectly replace them with in-sample encodings.
        return self.transformer_.fit_transform(frame, y)

    def transform(self, X):
        check_is_fitted(self, "transformer_")
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f"Prediction data is missing required columns: {sorted(missing)}")
        state = self._state(X.loc[:, self.feature_names_in_])
        state.profile = self.profile_
        # Reuse only row-local cleaning. Statistical operations are replayed
        # explicitly from fit-time metadata below.
        stable = [
            d
            for d in self.cleaning_.decisions
            if d.action
            not in {
                CleaningAction.IMPUTE_MISSING,
                CleaningAction.CLIP_OUTLIERS,
                CleaningAction.DROP_CONSTANT_COLUMN,
            }
        ]
        apply_cleaning_plan(state, self.cleaning_.model_copy(update={"decisions": stable}))
        frame = state.working_df
        for saved in self.cleaning_metadata_.get("cleaning_imputers", []):
            columns = saved["columns"]
            frame.loc[:, columns] = saved["imputer"].transform(frame[columns])
        for column, value in self.cleaning_metadata_.get("cleaning_fill_values", {}).items():
            if column in frame and value is not None:
                frame[column] = frame[column].fillna(value)
        for column, bounds in self.cleaning_metadata_.get("cleaning_clip_bounds", {}).items():
            if column in frame:
                frame[column] = frame[column].clip(lower=bounds[0], upper=bounds[1])
        state.extras["frozen_feature_encodings"] = self.encodings_
        frame = apply_feature_plan(state, self.features_)
        # The feature selection and schema are fixed by fit, never by this batch.
        return self.transformer_.transform(frame.reindex(columns=self.columns_))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "transformer_")
        return self.transformer_.get_feature_names_out()


def prepare_run_partitions(state):
    """Seal holdout rows before any learned preparation or model planning."""
    from ..profiling.profiler import profile_dataframe
    from .splitter import make_splits

    if "raw_splits" in state.extras:
        return
    frame = state.raw_df.copy().reset_index(drop=True)
    # Exact duplicates must not straddle train and holdout. This operation is
    # independent of learned values and happens once, before partitioning.
    frame = frame.drop_duplicates()
    state.working_df = frame
    state.feature_frame = None
    state.feature_names = [c for c in frame.columns if c != state.target]
    splits = make_splits(state)
    state.extras["sealed_test"] = (splits.X_test, splits.y_test)
    splits.X_test = splits.y_test = None
    state.extras["raw_splits"] = splits
    train = splits.X_train.copy()
    if state.target and splits.y_train is not None:
        # Keep original labels for agent interpretation; fitting uses encoded y.
        train[state.target] = frame.loc[train.index, state.target]
    state.raw_df = train
    state.working_df = train.copy()
    state.profile = profile_dataframe(train, target=state.target, settings=state.settings)
    state._frozen_context = None
    state.add_warning(
        "Planning profiles use training rows only. The test partition is sealed until model selection and replanning finish."
    )


def configure_preparation(state):
    prepare_run_partitions(state)
    from ..ingestion.base import redact_source

    state.splits = state.extras["raw_splits"]
    state.preprocessor = PlanPreprocessor(
        config=state.config.model_copy(
            update={"source": redact_source(state.config.source)}
        ).model_dump(mode="json"),
        problem=state.problem,
        cleaning=state.cleaning,
        features=state.features,
    )
    state.feature_names = list(state.splits.X_train.columns)


def score_final_holdout(state):
    from ..core.schemas import MetricValue
    from .metrics import score_predictions

    if "final_test_metrics" in state.extras or state.best_pipeline is None:
        return
    X, y = state.extras.get("sealed_test", (None, None))
    if X is None or y is None or not state.task_type.is_supervised:
        return
    model = state.best_pipeline
    predicted = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    metrics = score_predictions(state.task_type, y, predicted, y_proba=probabilities)
    state.extras["final_test_metrics"] = [
        MetricValue(name=k, value=v) for k, v in metrics.items() if np.isfinite(v)
    ]
    state.bus.log("Final test scored once after model selection; it did not drive replanning.")


def record_fitted_preparation(state):
    """Report actual fitted actions, including any decisions excluded for inference."""
    for step in getattr(state.best_pipeline, "named_steps", {}).values():
        if isinstance(step, PlanPreprocessor):
            state.applied_cleaning = list(step.applied_cleaning_)
            state.applied_features = list(step.applied_features_)
            for warning in step.warnings_:
                if warning not in state.warnings:
                    state.add_warning(warning)
