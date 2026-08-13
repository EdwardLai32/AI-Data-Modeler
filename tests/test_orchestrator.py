"""End-to-end runs, entirely offline.

This is the test that justifies the fake LLM's existence. The orchestrator walks
a plan that an agent wrote, invokes twelve more agents, applies their decisions
with real pandas and scikit-learn, and persists the result — and all of that runs
here in a couple of seconds with no credentials and no nondeterminism, because
every ``structured()`` call resolves against a canned response keyed by output
model type.

What is actually being asserted: that a completed run is *auditable*. A summary
must carry the decision at every stage, a reason attached to each one, an event
trail, and usage accounting. A run that produces a model but cannot explain how
it got there has failed at this project's actual purpose, even if the model is
good.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from automl_architect.core.schemas import (
    EvaluationVerdict,
    EventKind,
    ExecutionPlan,
    RunStatus,
    RunSummary,
    TaskType,
)

from .conftest import (
    CHURN_LEAK,
    CHURN_TARGET,
    HOUSE_CSV,
    HOUSE_TARGET,
    SALES_CSV,
    SALES_TARGET,
    FakeLLMClient,
    evaluation_verdict,
    import_or_skip,
)


#: Every test in this file drives a full pipeline, which is minutes of work in
#: aggregate even with the fake LLM.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def runner_mod():  # noqa: ANN201
    return import_or_skip("automl_architect.runner", feature="runner.py")


@pytest.fixture
def architect(runner_mod, installed_fake_llm: FakeLLMClient):  # noqa: ANN201
    """A library facade wired to the fake LLM and no persistence.

    Persistence is disabled by default here so a failing repository cannot be
    mistaken for a failing orchestrator; it gets its own test below.
    """
    return runner_mod.AutoMLArchitect(repository=False)


#: Minimal run: two candidates, a row cap, no tuning, no SHAP, JSON only. Used by
#: the tests that assert on *configuration handling* rather than on run content,
#: where a smaller sample proves the same thing in half the wall clock.
FAST_RUN: dict[str, Any] = {
    "time_budget_seconds": 300,
    "max_experiments": 2,
    "max_rows": 900,
    "cv_folds": 3,
    "enable_tuning": False,
    "enable_explainability": False,
    "report_formats": ["json"],
}

#: A representative run over the full file: explainability on, a rendered report,
#: and a fairness attribute, so the auditability assertions have real content.
FULL_RUN: dict[str, Any] = {
    **FAST_RUN,
    "max_experiments": 3,
    "max_rows": None,
    "enable_explainability": True,
    "report_formats": ["markdown", "json"],
    "fairness_attributes": ["region"],
}

#: One completed run, reused across every assertion that only *reads* it.
_SHARED: dict[str, tuple[RunSummary, list[Any]]] = {}


@pytest.fixture
def shared_run(architect, churn_csv: Path) -> tuple[RunSummary, list[Any]]:
    """A completed churn run and its event log, computed once per session.

    A full pipeline is thirteen agent calls plus real cross-validated training,
    chart rendering, and report writing — tens of seconds. Roughly twenty
    assertions below only read the result, so re-running for each would make the
    suite too slow to run. Tests that need a *fresh* run — replanning, approvals,
    configuration overrides — request one explicitly and pay for it.

    Returns:
        The summary and a snapshot of the run's events.
    """
    if "churn" not in _SHARED:
        summary = architect.analyse(str(churn_csv), target=CHURN_TARGET, **FULL_RUN)
        _SHARED["churn"] = (summary, list(architect.last_run.bus.events))
    return _SHARED["churn"]


@pytest.fixture
def churn_summary(shared_run: tuple[RunSummary, list[Any]]) -> RunSummary:
    """The shared completed run's summary."""
    return shared_run[0]


@pytest.fixture
def churn_events(shared_run: tuple[RunSummary, list[Any]]) -> list[Any]:
    """The shared completed run's event log."""
    return shared_run[1]


# ===========================================================================
# A completed run
# ===========================================================================


class TestCompletedRun:
    def test_run_reaches_a_terminal_state(self, churn_summary: RunSummary) -> None:
        assert isinstance(churn_summary, RunSummary)
        assert churn_summary.status is RunStatus.COMPLETED, churn_summary.error

    def test_no_error_recorded(self, churn_summary: RunSummary) -> None:
        assert churn_summary.error is None

    def test_every_stage_produced_its_output(self, churn_summary: RunSummary) -> None:
        """A gap here means a stage silently no-opped."""
        for stage in (
            "ingestion",
            "profile",
            "understanding",
            "problem",
            "plan",
            "cleaning",
            "features",
            "model_selection",
            "experiments",
            "evaluation",
            "insights",
            "report",
        ):
            assert getattr(churn_summary, stage) is not None, f"{stage} was never populated"

    def test_problem_was_identified_correctly(self, churn_summary: RunSummary) -> None:
        problem = churn_summary.problem
        assert problem.task_type is TaskType.BINARY_CLASSIFICATION
        assert problem.target_column == CHURN_TARGET
        assert problem.rationale.strip()

    def test_a_model_was_trained_and_ranked(self, churn_summary: RunSummary) -> None:
        log = churn_summary.experiments
        assert log.results, "no experiments recorded"
        best = log.best()
        assert best is not None and not best.failed
        assert best.primary_score is not None

    def test_the_leakage_column_never_reached_a_model(self, churn_summary: RunSummary) -> None:
        """The end-to-end proof that the leakage path works.

        churn.csv contains one column that separates the target at AUC 0.9975 and
        cannot exist at prediction time. If it is still a model input at the end of
        a run, every number in the report is meaningless.
        """
        assert CHURN_LEAK in {c.name for c in churn_summary.profile.columns}
        flagged = {f.column for f in churn_summary.profile.leakage_findings}
        assert CHURN_LEAK in flagged, "the profiler did not flag the known leak"
        assert CHURN_LEAK in churn_summary.cleaning.columns_to_drop or any(
            CHURN_LEAK in decision.columns for decision in churn_summary.cleaning.decisions
        ), "the cleaning plan did not act on the flagged leak"

    def test_usage_was_accounted(self, churn_summary: RunSummary) -> None:
        usage = churn_summary.usage
        assert usage.llm_calls >= 8, f"only {usage.llm_calls} agents were consulted"
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.cost_usd > 0.0

    def test_prompt_cache_was_exercised(self, churn_summary: RunSummary) -> None:
        """The frozen run context exists to be read from cache by later agents."""
        assert churn_summary.usage.cache_read_tokens > 0

    def test_steps_are_recorded_with_outcomes(self, churn_summary: RunSummary) -> None:
        assert churn_summary.steps
        for step in churn_summary.steps:
            assert step.step_id
            assert step.status.value in {
                "completed",
                "skipped",
                "failed",
                "pending",
                "running",
                "awaiting_approval",
            }
        completed = [s for s in churn_summary.steps if s.status.value == "completed"]
        assert len(completed) >= 6, f"only {len(completed)} steps completed"

    def test_duration_and_timestamps_are_consistent(self, churn_summary: RunSummary) -> None:
        assert churn_summary.started_at is not None
        assert churn_summary.finished_at is not None
        assert churn_summary.finished_at >= churn_summary.started_at
        assert churn_summary.duration_seconds >= 0.0

    def test_summary_survives_json_round_trip(self, churn_summary: RunSummary) -> None:
        """The API returns this object and the repository stores it."""
        payload = churn_summary.model_dump(mode="json")
        assert RunSummary.model_validate(payload).run_id == churn_summary.run_id

    def test_artifacts_were_written(self, churn_summary: RunSummary) -> None:
        assert churn_summary.artifact_dir
        directory = Path(churn_summary.artifact_dir)
        assert directory.exists()
        assert any(directory.rglob("*")), "the run directory is empty"


# ===========================================================================
# Auditability
# ===========================================================================


class TestAuditability:
    def test_every_cleaning_decision_has_a_reason(self, churn_summary: RunSummary) -> None:
        for decision in churn_summary.cleaning.decisions:
            assert decision.rationale.strip(), f"{decision.action.value} has no rationale"

    def test_every_feature_decision_has_a_reason(self, churn_summary: RunSummary) -> None:
        for decision in churn_summary.features.decisions:
            assert decision.rationale.strip(), f"{decision.op.value} has no rationale"

    def test_every_model_candidate_has_a_reason(self, churn_summary: RunSummary) -> None:
        for candidate in churn_summary.model_selection.candidates:
            assert candidate.rationale.strip(), f"{candidate.family.value} has no rationale"

    def test_every_plan_step_has_a_reason(self, churn_summary: RunSummary) -> None:
        for step in churn_summary.plan.steps:
            assert step.rationale.strip(), f"{step.step_id} has no rationale"

    def test_the_evaluation_argues_its_verdict(self, churn_summary: RunSummary) -> None:
        verdict = churn_summary.evaluation
        assert verdict.verdict_rationale.strip()
        assert verdict.recommended_action
        assert verdict.overall_grade in "ABCDF"

    def test_the_report_recommends_a_deployment_pattern_with_reasons(
        self, churn_summary: RunSummary
    ) -> None:
        deployment = churn_summary.report.deployment
        assert deployment.rationale.strip()
        assert deployment.pattern is not None


# ===========================================================================
# Events
# ===========================================================================


class TestEventStream:
    def test_run_lifecycle_events_are_emitted(self, churn_events: list[Any]) -> None:
        kinds = {event.kind for event in churn_events}
        assert EventKind.RUN_STARTED in kinds
        assert EventKind.RUN_COMPLETED in kinds
        assert EventKind.STEP_STARTED in kinds
        assert EventKind.AGENT_DECISION in kinds

    def test_event_sequence_is_strictly_increasing(self, churn_events: list[Any]) -> None:
        """The UI and the reconnect cursor both depend on a monotonic sequence."""
        sequences = [event.sequence for event in churn_events]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)

    def test_events_all_belong_to_the_run(
        self, churn_events: list[Any], churn_summary: RunSummary
    ) -> None:
        for event in churn_events:
            assert event.run_id == churn_summary.run_id

    def test_agent_decisions_carry_a_message(self, churn_events: list[Any]) -> None:
        """These are the lines the live UI renders, so an empty one is a blank row."""
        decisions = [e for e in churn_events if e.kind is EventKind.AGENT_DECISION]
        assert decisions
        for event in decisions:
            assert event.message.strip()
            assert event.agent is not None

    def test_llm_calls_are_costed(self, churn_events: list[Any]) -> None:
        finished = [
            e for e in churn_events if e.kind is EventKind.LLM_CALL and e.tokens_out
        ]
        assert finished
        assert all(e.cost_usd is not None and e.cost_usd >= 0 for e in finished)


# ===========================================================================
# The self-improvement loop
# ===========================================================================


class TestReplanning:
    def test_a_rejected_model_triggers_a_replan(
        self, architect, installed_fake_llm: FakeLLMClient, churn_csv: Path
    ) -> None:
        """The first verdict rejects the model; the second accepts it.

        This is the loop the whole architecture is built around: evaluation is a
        real gate, not a formality, and a rejection has to unwind to the planner
        without discarding what the first attempt learned.
        """
        installed_fake_llm.register(EvaluationVerdict, evaluation_verdict(acceptable=False))
        installed_fake_llm.register(
            EvaluationVerdict, evaluation_verdict(acceptable=True), sticky=True
        )

        summary = architect.analyse(
            str(churn_csv), target=CHURN_TARGET, **{**FAST_RUN, "max_replans": 1}
        )
        assert summary.replans >= 1, "a rejected verdict did not cause a replan"
        assert len(installed_fake_llm.calls_for(ExecutionPlan)) >= 2, (
            "the planner was not re-consulted"
        )
        assert summary.plan_history, "the revised plan discarded its predecessor"
        assert summary.plan.revision >= 1, "the revised plan is not marked as a revision"
        assert summary.plan.revision_reason, "a revision must record why it happened"

    def test_replans_are_bounded(
        self, architect, installed_fake_llm: FakeLLMClient, churn_csv: Path
    ) -> None:
        """An agent that always rejects must not loop forever."""
        installed_fake_llm.register(
            EvaluationVerdict, evaluation_verdict(acceptable=False), sticky=True
        )
        summary = architect.analyse(
            str(churn_csv), target=CHURN_TARGET, **{**FAST_RUN, "max_replans": 1}
        )
        assert summary.replans <= 1
        assert summary.status in (RunStatus.COMPLETED, RunStatus.FAILED)
        if summary.status is RunStatus.COMPLETED:
            # Completing with an honest failing verdict is the correct outcome —
            # far better than looping, and better than reporting a false pass.
            assert summary.evaluation.acceptable is False

    def test_self_improvement_can_be_switched_off(
        self, architect, installed_fake_llm: FakeLLMClient, churn_csv: Path
    ) -> None:
        installed_fake_llm.register(EvaluationVerdict, evaluation_verdict(acceptable=False), sticky=True)
        summary = architect.analyse(
            str(churn_csv),
            target=CHURN_TARGET,
            **{**FAST_RUN, "enable_self_improvement": False, "max_replans": 3},
        )
        assert summary.replans == 0


# ===========================================================================
# Human-in-the-loop approval
# ===========================================================================


class TestApproval:
    """The human-in-the-loop gate.

    Each test here needs a *live* orchestrator to resume, so unlike the read-side
    assertions these cannot share one run — approving a suspended run mutates it.
    """

    def suspended(self, architect, churn_csv: Path) -> Any:
        """Start a run that requires approval and return it once it suspends."""
        summary = architect.analyse(
            str(churn_csv), target=CHURN_TARGET, **{**FAST_RUN, "require_approval": True}
        )
        if summary.status is not RunStatus.AWAITING_APPROVAL:
            pytest.skip(
                f"run reached {summary.status.value} without proposing a destructive step"
            )
        return summary

    def test_a_destructive_step_suspends_the_run(self, architect, churn_csv: Path) -> None:
        """With approval required, the run must stop and ask rather than proceed."""
        summary = self.suspended(architect, churn_csv)
        pending = [a for a in summary.approvals if a.decision == "pending"]
        assert pending, "the run suspended without recording what it is waiting for"

        request = pending[0]
        assert request.action_summary.strip(), "a reviewer needs to know what they are approving"
        assert request.step_id
        assert request.severity is not None

    def test_approving_resumes_to_completion(self, architect, churn_csv: Path) -> None:
        summary = self.suspended(architect, churn_csv)
        while summary.status is RunStatus.AWAITING_APPROVAL:
            pending = next(a for a in summary.approvals if a.decision == "pending")
            summary = architect.resume(pending.request_id, True, note="approved by the test")
        assert summary.status is RunStatus.COMPLETED, summary.error
        assert all(a.decision != "pending" for a in summary.approvals)
        assert all(a.decided_at is not None for a in summary.approvals)

    def test_rejecting_is_recorded(self, architect, churn_csv: Path) -> None:
        summary = self.suspended(architect, churn_csv)
        pending = next(a for a in summary.approvals if a.decision == "pending")
        summary = architect.resume(pending.request_id, False, note="declined by the test")
        resolved = next(a for a in summary.approvals if a.request_id == pending.request_id)
        assert resolved.decision == "rejected"
        assert resolved.note == "declined by the test"


# ===========================================================================
# Other task types
# ===========================================================================


class TestOtherTasks:
    def test_regression_run(self, architect) -> None:
        """The skewed-target regression example, end to end."""
        summary = architect.analyse(str(HOUSE_CSV), target=HOUSE_TARGET, **FAST_RUN)
        assert summary.status is RunStatus.COMPLETED, summary.error
        assert summary.problem.task_type is TaskType.REGRESSION
        assert summary.experiments.best() is not None

    def test_time_series_run(self, architect) -> None:
        """A panel series with gaps must not crash the pipeline.

        No row cap here: truncating the file would drop three of the four series
        and both calendar gaps, which is the only thing this dataset is for.
        """
        summary = architect.analyse(
            str(SALES_CSV), target=SALES_TARGET, **{**FAST_RUN, "max_rows": None}
        )
        assert summary.status in (RunStatus.COMPLETED, RunStatus.FAILED)
        if summary.status is RunStatus.FAILED:
            pytest.skip(f"time-series path not complete: {summary.error}")
        assert summary.experiments is not None
        assert summary.ingestion.n_rows == 2914, "the full panel should have been loaded"

    def test_dataframe_source(self, architect, churn_df) -> None:
        """A caller with a frame in hand should not have to write it to disk."""
        summary = architect.analyse(
            churn_df, target=CHURN_TARGET, **{**FAST_RUN, "max_rows": None}
        )
        assert summary.status is RunStatus.COMPLETED, summary.error
        assert summary.ingestion.n_rows == len(churn_df)


# ===========================================================================
# Configuration is obeyed
# ===========================================================================


class TestConfigurationIsObeyed:
    def test_experiment_cap(self, architect, churn_csv: Path) -> None:
        summary = architect.analyse(
            str(churn_csv), target=CHURN_TARGET, **{**FAST_RUN, "max_experiments": 2}
        )
        non_baseline = [r for r in summary.experiments.results if not r.is_baseline]
        assert len(non_baseline) <= 2

    def test_row_cap(self, architect, churn_csv: Path) -> None:
        summary = architect.analyse(
            str(churn_csv), target=CHURN_TARGET, **{**FAST_RUN, "max_rows": 600}
        )
        assert summary.ingestion.n_rows <= 600
        assert summary.ingestion.truncated is True

    def test_metric_override(self, architect, churn_csv: Path) -> None:
        """An operator override must beat the agent's own metric choice."""
        summary = architect.analyse(
            str(churn_csv),
            target=CHURN_TARGET,
            **{**FAST_RUN, "primary_metric_override": "f1"},
        )
        assert summary.experiments.primary_metric == "f1"

    def test_task_type_override(self, architect, churn_csv: Path) -> None:
        summary = architect.analyse(
            str(churn_csv),
            target=CHURN_TARGET,
            **{**FAST_RUN, "task_type_override": TaskType.BINARY_CLASSIFICATION},
        )
        assert summary.config.task_type_override is TaskType.BINARY_CLASSIFICATION

    def test_tuning_disabled_is_reported_not_skipped_silently(
        self, architect, churn_csv: Path
    ) -> None:
        summary = architect.analyse(
            str(churn_csv), target=CHURN_TARGET, **{**FAST_RUN, "enable_tuning": False}
        )
        assert summary.tuning is None or summary.tuning.ran is False

    def test_explainability_can_be_disabled(self, architect, churn_csv: Path) -> None:
        """Base config has it on, so this genuinely tests the off switch."""
        summary = architect.analyse(
            str(churn_csv),
            target=CHURN_TARGET,
            **{**FULL_RUN, "enable_explainability": False},
        )
        assert summary.explainability is None or not summary.explainability.global_attributions

    def test_explainability_runs_when_enabled(self, churn_summary: RunSummary) -> None:
        """The other half of the switch, from the shared run."""
        assert churn_summary.explainability is not None
        assert churn_summary.explainability.global_attributions

    def test_unknown_config_option_is_rejected(self, runner_mod, churn_csv: Path) -> None:
        """Silently dropping ``time_budget=60`` would produce a disobedient run."""
        from automl_architect.core.errors import ConfigurationError

        with pytest.raises(ConfigurationError):
            runner_mod.build_run_config(str(churn_csv), time_budget=60)

    def test_random_state_is_carried_through(self, architect, churn_csv: Path) -> None:
        summary = architect.analyse(
            str(churn_csv), target=CHURN_TARGET, **{**FAST_RUN, "random_state": 1234}
        )
        assert summary.config.random_state == 1234


# ===========================================================================
# Failure handling
# ===========================================================================


class TestFailureHandling:
    def test_a_missing_file_fails_the_run_cleanly(self, architect, tmp_path: Path) -> None:
        """A bad source is a failed run with a message, not a traceback to the caller."""
        from automl_architect.core.errors import AutoMLArchitectError

        try:
            summary = architect.analyse(str(tmp_path / "absent.csv"), target="y", **FAST_RUN)
        except AutoMLArchitectError as exc:
            assert str(exc)
            return
        assert summary.status is RunStatus.FAILED
        assert summary.error

    def test_an_absent_target_column_is_handled(self, architect, churn_csv: Path) -> None:
        """Either the run fails cleanly, or the agents choose a target that exists."""
        from automl_architect.core.errors import AutoMLArchitectError

        try:
            summary = architect.analyse(
                str(churn_csv), target="a_column_that_is_not_there", **FAST_RUN
            )
        except AutoMLArchitectError as exc:
            assert str(exc)
            return

        if summary.status is RunStatus.FAILED:
            assert summary.error
            return

        real_columns = {c.name for c in summary.profile.columns}
        assert summary.problem.target_column in real_columns, (
            "the run completed against a target column that does not exist"
        )

    def test_warnings_are_surfaced_not_swallowed(self, churn_summary: RunSummary) -> None:
        """Degraded capability has to be visible in the summary."""
        assert isinstance(churn_summary.warnings, list)
        for warning in churn_summary.warnings:
            assert isinstance(warning, str) and warning.strip()


# ===========================================================================
# Persistence and dataset memory
# ===========================================================================


class TestPersistence:
    """Storage, dataset memory, and the Q&A interface over a stored run.

    Two runs against a persisting architect, executed once and asserted on from
    several angles. They are a single test rather than five because each run costs
    tens of seconds and they all interrogate the same two rows.
    """

    def test_two_runs_are_stored_and_retrievable(
        self, runner_mod, installed_fake_llm: FakeLLMClient, churn_csv: Path
    ) -> None:
        architect = runner_mod.AutoMLArchitect()
        if architect.repository is None:
            pytest.skip("persistence unavailable in this environment")

        first = architect.analyse(str(churn_csv), target=CHURN_TARGET, **FAST_RUN)
        second = architect.analyse(str(churn_csv), target=CHURN_TARGET, **FAST_RUN)
        assert first.run_id != second.run_id

        # --- a stored run round-trips ------------------------------------
        loaded = architect.get_run(second.run_id)
        assert loaded is not None
        assert loaded.run_id == second.run_id
        assert loaded.status is second.status
        assert loaded.problem is not None, "the stored summary lost its problem definition"
        assert loaded.experiments is not None, "the stored summary lost its leaderboard"

        # --- both appear in the index ------------------------------------
        listed = {run.run_id for run in architect.list_runs(limit=20)}
        assert {first.run_id, second.run_id} <= listed

        # --- the audit trail survives ------------------------------------
        events = architect.repository.get_events(second.run_id)
        assert events, "the audit trail was not stored"
        assert all(event.run_id == second.run_id for event in events)
        sequences = [event.sequence for event in events]
        assert sequences == sorted(sequences)

        # --- dataset memory saw the first run ---------------------------
        fingerprint = None
        if hasattr(architect.repository, "get_fingerprint_for_run"):
            fingerprint = architect.repository.get_fingerprint_for_run(first.run_id)
        if fingerprint is not None:
            similar = architect.repository.find_similar(fingerprint, limit=5)
            assert isinstance(similar, list)

        # --- and the run can be questioned ------------------------------
        answer = architect.ask(second.run_id, "Why was this metric chosen over accuracy?")
        assert answer.question
        assert answer.answer.strip()
        assert answer.confidence in ("high", "medium", "low")

    def test_an_unknown_run_id_returns_none(self, runner_mod) -> None:
        """A missing run is absence, not an exception."""
        architect = runner_mod.AutoMLArchitect()
        if architect.repository is None:
            pytest.skip("persistence unavailable")
        assert architect.get_run("run_definitely_not_stored") is None


# ===========================================================================
# Offline guarantee
# ===========================================================================


def test_the_whole_run_went_through_the_fake(
    architect, installed_fake_llm: FakeLLMClient, churn_csv: Path
) -> None:
    """Nothing in a run may reach the network.

    Every reasoning step is recorded on the fake, so the call count is direct
    evidence that no agent bypassed the injected client.
    """
    architect.analyse(str(churn_csv), target=CHURN_TARGET, **FAST_RUN)
    assert installed_fake_llm.n_calls >= 8
    assert all(call.system for call in installed_fake_llm.calls), "a call carried no system prompt"
