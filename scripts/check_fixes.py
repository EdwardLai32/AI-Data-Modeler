"""Regression checks for the two defects the first live run exposed.

1. ``min_frequency`` was coerced to ``float`` unconditionally, so an agent asking
   to pool categories seen fewer than 20 times produced ``20.0`` — neither an int
   count nor a proportion in (0, 1). Because the one-hot encoder sits in the
   shared preprocessor, that single bad parameter failed *every* candidate model,
   the baseline included, and the run ended with NoViableModelError.

2. The Feature Agent's postprocess filtered input columns against the source
   schema, which silently stripped references to columns that earlier ops in the
   same plan create.

Both are checked against the real sklearn validator rather than a reimplementation
of its rules, because the point is what sklearn accepts, not what we think it does.
"""

from __future__ import annotations

from sklearn.preprocessing import OneHotEncoder

from automl_architect.agents.features import FeatureAgent
from automl_architect.core.schemas import FeatureDecision, FeatureOp, FeaturePlan
from automl_architect.execution.feature_ops import _coerce_min_frequency


def check_min_frequency() -> list[str]:
    """Every coerced value must be one sklearn will actually accept."""
    failures: list[str] = []
    warnings: list[str] = []
    note = warnings.append

    cases: list[tuple[object, object]] = [
        (20, 20),  # the value that broke the live run
        (20.0, 20),  # float-typed count, the actual failure mode
        (1, 1),
        (1.0, 1),
        (0.05, 0.05),  # a genuine proportion must stay a float
        (0.5, 0.5),
        (None, None),
        ("nonsense", None),
        (True, None),  # bool is an int subclass; not a frequency
        (0, None),
        (-5, None),
        (2.5, 2),  # fractional count, rounded with a warning
    ]

    for raw, expected in cases:
        got = _coerce_min_frequency(raw, on_invalid=note)
        if got != expected or type(got) is not type(expected):
            failures.append(
                f"_coerce_min_frequency({raw!r}) -> {got!r} ({type(got).__name__}), "
                f"expected {expected!r} ({type(expected).__name__})"
            )
            continue
        # The real test: does sklearn accept it?
        try:
            OneHotEncoder(min_frequency=got, handle_unknown="infrequent_if_exist")._validate_params()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"sklearn rejected min_frequency={got!r} from {raw!r}: {exc}")

    print(f"  min_frequency: {len(cases)} cases, {len(failures)} failure(s)")
    for message in warnings:
        print(f"    warned: {message}")
    return failures


def check_sequential_feature_columns() -> list[str]:
    """A later op consuming an earlier op's output must survive postprocess."""
    failures: list[str] = []

    plan = FeaturePlan(
        summary="log-transform income, then scale the result",
        decisions=[
            FeatureDecision(
                op=FeatureOp.LOG_TRANSFORM,
                input_columns=["annual_income"],
                output_name_hint="annual_income_log1p",
                rationale="Right-skewed positive values compress usefully under log1p.",
            ),
            FeatureDecision(
                op=FeatureOp.STANDARD_SCALE,
                input_columns=["annual_income_log1p", "tenure_months"],
                rationale="Scale the engineered feature alongside the raw one.",
            ),
        ],
    )

    class _Probe(FeatureAgent):
        """Only postprocess is under test, so the LLM plumbing is bypassed."""

        def __init__(self) -> None:  # noqa: D107 - deliberately skips BaseAgent.__init__
            pass

    from automl_architect.core.schemas import DataSource, RunConfig, SourceKind
    from automl_architect.core.state import RunState

    import pandas as pd

    state = RunState(
        config=RunConfig(source=DataSource(kind=SourceKind.CSV, uri="x.csv")),
    )
    state.raw_df = pd.DataFrame(
        {"annual_income": [1.0, 2.0, 3.0], "tenure_months": [1, 2, 3], "churned": [0, 1, 0]}
    )

    result = _Probe().postprocess(plan, state)
    scale = next((d for d in result.decisions if d.op is FeatureOp.STANDARD_SCALE), None)

    if scale is None:
        failures.append("the standard_scale decision was dropped entirely")
    elif "annual_income_log1p" not in scale.input_columns:
        failures.append(
            "postprocess stripped 'annual_income_log1p' from standard_scale — the "
            f"surviving inputs were {scale.input_columns}"
        )

    print(f"  sequential columns: {len(failures)} failure(s)")
    if scale is not None:
        print(f"    standard_scale inputs kept: {scale.input_columns}")
    return failures


def main() -> int:
    print("Regression checks for the live-run defects")
    print("=" * 60)
    failures = check_min_frequency() + check_sequential_feature_columns()
    print("=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("All regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
