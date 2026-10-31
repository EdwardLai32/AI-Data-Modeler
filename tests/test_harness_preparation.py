import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from automl_architect.core.schemas import (
    CleaningAction,
    CleaningDecision,
    CleaningPlan,
    DataSource,
    FeatureDecision,
    FeatureOp,
    FeaturePlan,
    MissingStrategy,
    RunConfig,
    SourceKind,
    TaskType,
)
from automl_architect.execution.preparation import PlanPreprocessor


def preparation():
    cfg = RunConfig(
        source=DataSource(kind=SourceKind.CSV, uri="unused.csv"),
        target_column="label",
        task_type_override=TaskType.BINARY_CLASSIFICATION,
    )
    return PlanPreprocessor(
        config=cfg.model_dump(mode="json"),
        cleaning=CleaningPlan(
            summary="test",
            decisions=[
                CleaningDecision(
                    action=CleaningAction.IMPUTE_MISSING,
                    columns=["amount"],
                    strategy=MissingStrategy.MEAN,
                    rationale="training mean",
                )
            ],
        ),
        features=FeaturePlan(
            summary="test",
            decisions=[
                FeatureDecision(
                    op=FeatureOp.TARGET_ENCODE,
                    input_columns=["category"],
                    rationale="test cross fitting",
                )
            ],
        ),
    )


def data():
    rng = np.random.default_rng(19)
    X = pd.DataFrame({"amount": rng.normal(size=90), "category": ["a", "b", "c"] * 30})
    X.loc[0, "amount"] = np.nan
    y = np.asarray([0, 1, 0, 1, 1, 0] * 15)
    return X, y


def test_transform_uses_frozen_training_statistics():
    X, y = data()
    prep = preparation().fit(X.iloc[:60], y[:60])
    mean = prep.cleaning_metadata_["cleaning_fill_values"]["amount"]
    assert np.isclose(mean, X.iloc[:60].amount.mean())
    heldout = X.iloc[60:].copy()
    before = prep.transform(heldout.iloc[:1])
    heldout.loc[heldout.index[1:], "amount"] = 1e12
    heldout.loc[heldout.index[1:], "category"] = "unseen"
    after = prep.transform(heldout)[:1]
    np.testing.assert_allclose(before, after)
    assert prep.cleaning_metadata_["cleaning_fill_values"]["amount"] == mean


def test_fold_clone_refits_and_export_accepts_raw_rows(tmp_path):
    X, y = data()
    a = preparation().fit(X.iloc[:45], y[:45])
    b = clone(a).fit(X.iloc[45:], y[45:])
    assert a.cleaning_metadata_ != b.cleaning_metadata_
    pipeline = Pipeline([("prepare", preparation()), ("model", LogisticRegression())]).fit(X, y)
    dest = tmp_path / "model.joblib"
    joblib.dump(pipeline, dest)
    restored = joblib.load(dest)
    np.testing.assert_array_equal(pipeline.predict(X.iloc[:10]), restored.predict(X.iloc[:10]))


def test_prediction_labels_cannot_change_features():
    X, y = data()
    prep = preparation().fit(X, y)
    with_labels = X.iloc[:8].copy()
    with_labels["label"] = 999
    np.testing.assert_allclose(prep.transform(X.iloc[:8]), prep.transform(with_labels))


def test_calendar_features_preserve_training_schema_for_midnight_rows():
    X, y = data()
    X["date"] = pd.date_range("2025-01-01", periods=len(X), freq="6h")
    prep = preparation()
    prep.features = FeaturePlan(
        summary="calendar",
        decisions=[
            FeatureDecision(
                op=FeatureOp.DATE_DECOMPOSE, input_columns=["date"], rationale="date parts"
            )
        ],
    )
    prep.fit(X, y)
    assert "hour" in prep.encodings_["calendar"]["date"]
    np.testing.assert_allclose(prep.transform(X.iloc[:1]), prep.transform(X.iloc[:2])[:1])


def test_categorical_interaction_is_invariant_to_prediction_batch_cardinality():
    X, y = data()
    X["region"] = ["east", "west"] * 45
    prep = preparation()
    prep.features = FeaturePlan(
        summary="cross",
        decisions=[
            FeatureDecision(
                op=FeatureOp.INTERACTION, input_columns=["category", "region"], rationale="cross"
            )
        ],
    )
    prep.fit(X, y)
    batch = X.copy()
    batch.loc[1:, "region"] = [f"new_{i}" for i in range(len(batch) - 1)]
    np.testing.assert_allclose(prep.transform(X.iloc[:1]), prep.transform(batch)[:1])
