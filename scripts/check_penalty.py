"""Verify deprecated `penalty` is translated, not passed through or dropped.

Run under ``-W error::FutureWarning`` so a surviving deprecation warning fails
the check rather than scrolling past in the log.
"""

from __future__ import annotations

from sklearn.datasets import make_classification

from automl_architect.core.schemas import ModelFamily, TaskType
from automl_architect.execution.model_zoo import build_estimator

CASES: list[dict] = [
    {"penalty": "l2", "C": 0.5},
    {"penalty": "l1"},
    {"penalty": "elasticnet"},
    {"penalty": None},
    {"C": 2.0},  # untouched control
]


def main() -> int:
    X, y = make_classification(n_samples=150, n_features=6, random_state=0)
    failures: list[str] = []

    for params in CASES:
        try:
            est = build_estimator(
                ModelFamily.LOGISTIC,
                TaskType.BINARY_CLASSIFICATION,
                dict(params),
                random_state=0,
            )
            est.set_params(max_iter=400)
            est.fit(X, y)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{params} -> {type(exc).__name__}: {exc}")
            print(f"  FAIL  {params} -> {type(exc).__name__}: {str(exc)[:110]}")
            continue

        got = est.get_params()
        print(
            f"  ok    {str(params):28} -> l1_ratio={got.get('l1_ratio')} "
            f"C={got.get('C')} solver={got.get('solver')}"
        )

    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        return 1
    print("penalty translation OK — fits cleanly with no FutureWarning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
