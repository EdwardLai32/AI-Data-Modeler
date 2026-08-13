"""Confirm scale-sensitive families get a scaler, and tree models do not.

The live run trained logistic regression on raw-magnitude inputs and reported
``lbfgs failed to converge after 2000 iterations``. Whether to engineer a scaled
feature is the Feature Agent's judgement, but a penalised linear model or a
distance metric on unscaled inputs is simply wrong, so the trainer now enforces
it. This checks both halves: the scaler appears where it is required, and does
not appear where it would only cost time.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

from automl_architect.core.schemas import (
    DataSource,
    ModelFamily,
    RunConfig,
    SourceKind,
    TaskType,
)
from automl_architect.core.state import RunState
from automl_architect.execution.model_zoo import build_estimator
from automl_architect.execution.trainer import wrap_with_preprocessor

NEEDS_SCALING = [ModelFamily.LOGISTIC, ModelFamily.SVM, ModelFamily.KNN, ModelFamily.RIDGE]
NO_SCALING = [ModelFamily.RANDOM_FOREST, ModelFamily.HIST_GRADIENT_BOOSTING, ModelFamily.BASELINE_DUMMY]


def has_scaler(pipeline: object) -> bool:
    return isinstance(pipeline, Pipeline) and "requires_scale" in dict(pipeline.steps)


def main() -> int:
    failures: list[str] = []
    state = RunState(config=RunConfig(source=DataSource(kind=SourceKind.CSV, uri="x.csv")))
    task = TaskType.BINARY_CLASSIFICATION

    for family in NEEDS_SCALING:
        estimator = build_estimator(family, task, {}, random_state=0)
        pipeline = wrap_with_preprocessor(state, estimator, family)
        if not has_scaler(pipeline):
            failures.append(f"{family.value} did not receive a scaler")
            print(f"  FAIL  {family.value:<26} no scaler")
        else:
            print(f"  ok    {family.value:<26} scaler inserted")

    for family in NO_SCALING:
        estimator = build_estimator(family, task, {}, random_state=0)
        pipeline = wrap_with_preprocessor(state, estimator, family)
        if has_scaler(pipeline):
            failures.append(f"{family.value} was scaled unnecessarily")
            print(f"  FAIL  {family.value:<26} scaled but does not need it")
        else:
            print(f"  ok    {family.value:<26} correctly unscaled")

    # The behavioural check: badly-scaled features must now converge.
    X, y = make_classification(n_samples=400, n_features=6, random_state=0)
    X[:, 0] *= 250_000.0  # one column in wildly different units
    X[:, 1] *= 0.00004
    estimator = build_estimator(ModelFamily.LOGISTIC, task, {}, random_state=0)
    pipeline = wrap_with_preprocessor(state, estimator, ModelFamily.LOGISTIC)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X, y)
    convergence = [w for w in caught if "converge" in str(w.message).lower()]
    if convergence:
        failures.append("logistic still failed to converge on badly-scaled input")
        print(f"  FAIL  convergence: {convergence[0].message}")
    else:
        score = pipeline.score(X, y)
        print(f"  ok    logistic converged on badly-scaled input (accuracy {score:.3f})")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("scaling policy OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
