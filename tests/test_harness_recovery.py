"""A fresh orchestrator must reuse completed work, including fitted models."""

import numpy as np
import pandas as pd
import pytest

from automl_architect.config import Settings
from automl_architect.core.schemas import EventKind, RunStatus
from automl_architect.orchestrator.engine import Orchestrator
from automl_architect.runner import build_run_config


@pytest.mark.slow
def test_recovery_reuses_model_and_scores_the_sealed_test_once(tmp_path):
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"amount": rng.normal(size=150), "category": ["a", "b", "c"] * 50})
    frame["target"] = (frame.amount > 0).astype(int)
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    settings = Settings(workspace=tmp_path, offline=True, isolate_training=False)
    config = build_run_config(
        str(source),
        target="target",
        max_experiments=2,
        cv_folds=2,
        time_budget_seconds=180,
        enable_tuning=False,
        enable_explainability=False,
        enable_visualizations=False,
        enable_self_improvement=False,
        report_formats=["json"],
    )
    first = Orchestrator(config, settings=settings)

    def interrupt_after_training(event):
        if event.kind is EventKind.STEP_COMPLETED and first.state.best_pipeline is not None:
            first.cancel()

    first.bus.subscribe(interrupt_after_training)
    interrupted = first.run()
    assert interrupted.status is RunStatus.CANCELLED
    assert not interrupted.final_test_metrics
    expected = first.state.best_pipeline.predict(frame.drop(columns="target").iloc[:5])
    finished_steps = {s.step_id for s in first.state.steps if s.status.value == "completed"}
    fresh = Orchestrator(config, settings=settings)
    fresh.restore()
    summary = fresh.run()
    assert summary.status is RunStatus.COMPLETED, summary.error
    assert summary.final_test_metrics
    assert not any(
        e.kind is EventKind.STEP_STARTED and e.step_id in finished_steps for e in fresh.bus
    )
    np.testing.assert_array_equal(
        expected, fresh.state.best_pipeline.predict(frame.drop(columns="target").iloc[:5])
    )
    assert fresh.bus.events[0].sequence > first.bus.events[-1].sequence
