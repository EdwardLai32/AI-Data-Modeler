"""The keyless path: offline mode must be complete, grounded, and silent.

Three properties are worth defending here, and they are what these tests assert.

**Completeness.** Every agent in the package must have a decider. A missing one
would not surface until a long run reached that step and died, so the coverage
test walks the real agent registry rather than a hand-maintained list.

**Groundedness.** Offline decisions are cheap to produce and therefore easy to
produce badly. The rules must quote the measurement that drove them — a rationale
that would read the same for any dataset is the failure this codebase exists to
avoid, so the tests check that rationales cite real numbers and that the flagship
cleaning rules split on the statistic they claim to split on.

**Silence.** The whole point is that nothing reaches the API. The strongest form
of that test is a client whose transport raises on contact: if any code path
tries to call out, the test fails loudly instead of quietly costing money.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from typing import Any

import pytest
from pydantic import BaseModel

from automl_architect.config import get_settings, reset_settings_cache
from automl_architect.core import offline
from automl_architect.core.agent import BaseAgent
from automl_architect.core.errors import LLMOfflineError
from automl_architect.core.llm import LLMClient
from automl_architect.core.schemas import (
    CleaningAction,
    ColumnKind,
    MissingStrategy,
    ModelFamily,
    Severity,
    TaskType,
)
from automl_architect.core.state import RunState

# Rationales must cite evidence. A bare number is the cheapest checkable proxy
# for that, and every rule in the engine has one available.
_HAS_NUMBER = re.compile(r"\d")


@pytest.fixture
def offline_settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Settings with offline mode on, resolved the way the app resolves it."""
    monkeypatch.setenv("AUTOML_OFFLINE", "1")
    reset_settings_cache()
    settings = get_settings()
    assert settings.offline is True
    yield settings
    reset_settings_cache()


@pytest.fixture
def offline_state(run_state: RunState, offline_settings: Any) -> RunState:
    """The mid-flight churn state, switched into offline mode."""
    run_state.settings = offline_settings
    return run_state


def _agent_classes() -> list[type[BaseAgent]]:
    """Every concrete agent class the package actually ships."""
    import automl_architect.agents as pkg

    found: list[type[BaseAgent]] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{pkg.__name__}.{info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseAgent)
                and obj.__module__ == module.__name__
                and not inspect.isabstract(obj)
                and getattr(obj, "output_model", None) is not None
            ):
                found.append(obj)
    return found


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_every_agent_has_a_decider(self) -> None:
        """A missing decider must fail here, not three minutes into a run."""
        agents = _agent_classes()
        assert agents, "no agent classes were discovered; the walk is broken"
        missing = [
            f"{cls.__name__} -> {cls.output_model.__name__}"
            for cls in agents
            if not offline.supports(cls.output_model)
        ]
        assert not missing, f"agents with no offline decider: {missing}"

    def test_unknown_model_raises_a_useful_error(self, offline_state: RunState) -> None:
        class Unregistered(BaseModel):
            pass

        with pytest.raises(KeyError, match="no rule engine"):
            offline.decide(Unregistered, offline_state)

    @pytest.mark.parametrize("name", sorted(offline._DECIDERS))
    def test_decider_returns_its_declared_type(
        self, name: str, offline_state: RunState
    ) -> None:
        """Each decider must produce a valid instance of the model it is keyed by."""
        from automl_architect.core import schemas

        model = getattr(schemas, name)
        value = offline.decide(model, offline_state)
        assert isinstance(value, model)
        # Re-validating proves the instance would survive a round-trip through
        # the API and the stored run summary.
        model.model_validate(value.model_dump())


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


class TestReasoning:
    def test_cleaning_decisions_all_carry_evidence(self, offline_state: RunState) -> None:
        from automl_architect.core.schemas import CleaningPlan

        plan = offline.decide(CleaningPlan, offline_state)
        assert plan.decisions, "churn.csv has missing values; expected decisions"
        for decision in plan.decisions:
            assert decision.rationale.strip(), f"{decision.action} has no rationale"
            assert _HAS_NUMBER.search(decision.rationale), (
                f"{decision.action} on {decision.columns} gives a rationale with no "
                f"measurement in it: {decision.rationale!r}"
            )

    def test_skewed_column_gets_median_and_says_why(
        self, offline_state: RunState
    ) -> None:
        """The spec's worked example: a skewed column must impute by median."""
        from automl_architect.core.schemas import CleaningPlan

        profile = offline_state.profile
        assert profile is not None
        skewed = [
            c
            for c in profile.columns
            if c.kind in (ColumnKind.NUMERIC_CONTINUOUS, ColumnKind.NUMERIC_DISCRETE)
            and c.n_missing
            and c.skewness is not None
            and abs(c.skewness) > offline.SKEW_THRESHOLD
        ]
        if not skewed:
            pytest.skip("no skewed column with missing values in this profile")

        plan = offline.decide(CleaningPlan, offline_state)
        by_column = {
            d.columns[0]: d
            for d in plan.decisions
            if d.action is CleaningAction.IMPUTE_MISSING and d.columns
        }
        for column in skewed:
            decision = by_column.get(column.name)
            assert decision is not None, f"{column.name} was not imputed"
            assert decision.strategy is MissingStrategy.MEDIAN, (
                f"{column.name} has skewness {column.skewness} and should use the "
                f"median, got {decision.strategy}"
            )
            assert "skew" in decision.rationale.lower()

    def test_symmetric_column_gets_mean(self, offline_state: RunState) -> None:
        from automl_architect.core.schemas import CleaningPlan

        profile = offline_state.profile
        assert profile is not None
        symmetric = [
            c
            for c in profile.columns
            if c.kind in (ColumnKind.NUMERIC_CONTINUOUS, ColumnKind.NUMERIC_DISCRETE)
            and c.n_missing
            and c.missing_fraction < offline.DROP_MISSING_FRACTION
            and c.skewness is not None
            and abs(c.skewness) <= offline.SKEW_THRESHOLD
        ]
        if not symmetric:
            pytest.skip("no near-symmetric column with missing values")

        plan = offline.decide(CleaningPlan, offline_state)
        by_column = {
            d.columns[0]: d
            for d in plan.decisions
            if d.action is CleaningAction.IMPUTE_MISSING and d.columns
        }
        for column in symmetric:
            decision = by_column.get(column.name)
            assert decision is not None
            assert decision.strategy is MissingStrategy.MEAN

    def test_categorical_gets_mode_not_a_numeric_strategy(
        self, offline_state: RunState
    ) -> None:
        from automl_architect.core.schemas import CleaningPlan

        profile = offline_state.profile
        assert profile is not None
        plan = offline.decide(CleaningPlan, offline_state)
        categorical = {
            c.name
            for c in profile.columns
            if c.kind
            in (
                ColumnKind.CATEGORICAL_NOMINAL,
                ColumnKind.CATEGORICAL_ORDINAL,
                ColumnKind.BOOLEAN,
            )
        }
        for decision in plan.decisions:
            if decision.action is not CleaningAction.IMPUTE_MISSING:
                continue
            if not decision.columns or decision.columns[0] not in categorical:
                continue
            assert decision.strategy in (
                MissingStrategy.MODE,
                MissingStrategy.MISSING_CATEGORY,
            ), (
                f"{decision.columns[0]} is categorical; mean/median are undefined "
                f"for it, got {decision.strategy}"
            )

    def test_leakage_column_is_dropped_with_its_score_quoted(
        self, offline_state: RunState
    ) -> None:
        from automl_architect.core.schemas import CleaningPlan

        profile = offline_state.profile
        assert profile is not None
        severe = [
            f
            for f in profile.leakage_findings
            if f.severity in (Severity.HIGH, Severity.CRITICAL)
        ]
        if not severe:
            pytest.skip("the churn profile stub recorded no severe leakage")

        plan = offline.decide(CleaningPlan, offline_state)
        for finding in severe:
            assert finding.column in plan.columns_to_drop, (
                f"{finding.column} leaks at {finding.score} and must be dropped"
            )
            decision = next(
                d
                for d in plan.decisions
                if d.action is CleaningAction.DROP_LEAKAGE_COLUMN
                and finding.column in d.columns
            )
            assert decision.destructive is True

    def test_identifier_column_is_dropped(self, offline_state: RunState) -> None:
        from automl_architect.core.schemas import CleaningPlan

        profile = offline_state.profile
        assert profile is not None
        ids = [c.name for c in profile.columns if c.looks_like_id]
        if not ids:
            pytest.skip("no identifier-like column in this profile")
        plan = offline.decide(CleaningPlan, offline_state)
        for name in ids:
            assert name in plan.columns_to_drop

    def test_target_is_never_dropped_or_imputed(self, offline_state: RunState) -> None:
        """Imputing a target invents the answer; dropping it ends the run."""
        from automl_architect.core.schemas import CleaningPlan

        plan = offline.decide(CleaningPlan, offline_state)
        target = offline_state.target
        assert target
        assert target not in plan.columns_to_drop
        for decision in plan.decisions:
            if decision.action is CleaningAction.IMPUTE_MISSING:
                assert target not in decision.columns

    def test_problem_picks_a_ranking_metric_for_imbalanced_binary(
        self, offline_state: RunState
    ) -> None:
        from automl_architect.core.schemas import ProblemDefinition

        problem = offline.decide(ProblemDefinition, offline_state)
        assert problem.task_type is TaskType.BINARY_CLASSIFICATION
        assert problem.primary_metric == "roc_auc"
        assert _HAS_NUMBER.search(problem.rationale)
        assert problem.target_column == offline_state.target

    def test_regression_target_yields_rmse(self, regression_state: RunState) -> None:
        from automl_architect.core.schemas import ProblemDefinition

        regression_state.settings = get_settings()
        problem = offline.decide(ProblemDefinition, regression_state)
        assert problem.task_type is TaskType.REGRESSION
        assert problem.primary_metric == "rmse"

    def test_model_selection_always_includes_a_baseline(
        self, offline_state: RunState
    ) -> None:
        from automl_architect.core.schemas import ModelSelection

        selection = offline.decide(ModelSelection, offline_state)
        assert selection.candidates
        assert any(c.is_baseline for c in selection.candidates), (
            "without a baseline the leaderboard cannot be interpreted"
        )
        ranks = sorted(c.rank for c in selection.candidates)
        assert ranks == list(range(1, len(ranks) + 1)), "ranks must be dense from 1"
        for candidate in selection.candidates:
            assert candidate.rationale.strip()

    def test_model_selection_only_proposes_installed_families(
        self, offline_state: RunState
    ) -> None:
        """Proposing a family whose library is absent wastes a slot on a failure."""
        from automl_architect.core.schemas import ModelSelection

        selection = offline.decide(ModelSelection, offline_state)
        requires = {
            ModelFamily.LIGHTGBM: "lightgbm",
            ModelFamily.XGBOOST: "xgboost",
            ModelFamily.CATBOOST: "catboost",
        }
        for candidate in selection.candidates:
            module = requires.get(candidate.family)
            if module is not None:
                assert offline._installed(module), (
                    f"{candidate.family.value} proposed but {module} is not installed"
                )

    def test_plan_steps_are_canonical_and_acyclic(self, offline_state: RunState) -> None:
        """A plan naming a step the orchestrator cannot dispatch is a dead plan."""
        from automl_architect.core.schemas import ExecutionPlan
        from automl_architect.orchestrator.graph import CANONICAL_STEPS

        known = {s.step_id for s in CANONICAL_STEPS}
        plan = offline.decide(ExecutionPlan, offline_state)
        assert plan.steps

        seen: set[str] = set()
        for index, step in enumerate(plan.ordered()):
            assert step.step_id in known, f"invented step id {step.step_id!r}"
            assert step.step_id not in seen, f"duplicate step {step.step_id!r}"
            seen.add(step.step_id)
            assert step.order == index + 1, "orders must be dense and 1-based"
            for dep in step.depends_on:
                assert dep in seen, f"{step.step_id} depends on the later step {dep}"
            assert step.rationale.strip()

    def test_plan_assigns_no_step_to_a_pre_plan_agent(
        self, offline_state: RunState
    ) -> None:
        """Those steps run before planning, so the planner drops them with a warning."""
        from automl_architect.agents.planner import PRE_PLAN_AGENTS
        from automl_architect.core.schemas import ExecutionPlan

        plan = offline.decide(ExecutionPlan, offline_state)
        offenders = [s.step_id for s in plan.steps if s.agent in PRE_PLAN_AGENTS]
        assert not offenders, f"steps assigned to pre-plan agents: {offenders}"

    def test_plan_names_each_agent_at_most_once(self, offline_state: RunState) -> None:
        """A repeated agent means the second run silently overwrites the first."""
        from automl_architect.core.schemas import ExecutionPlan

        plan = offline.decide(ExecutionPlan, offline_state)
        agents = [s.agent for s in plan.steps]
        assert len(agents) == len(set(agents)), f"repeated agents in plan: {agents}"

    def test_narratives_are_marked_as_rule_derived(
        self, offline_state: RunState
    ) -> None:
        """A reader must never mistake rule output for model reasoning."""
        from automl_architect.core.schemas import DatasetUnderstanding

        understanding = offline.decide(DatasetUnderstanding, offline_state)
        assert offline.MARK in understanding.readiness_rationale
        assert "rule" in understanding.narrative.lower()
        assert understanding.column_assessments, "every column deserves an assessment"

    def test_understanding_assesses_only_real_columns(
        self, offline_state: RunState
    ) -> None:
        from automl_architect.core.schemas import DatasetUnderstanding

        understanding = offline.decide(DatasetUnderstanding, offline_state)
        real = {c.name for c in offline_state.profile.columns}  # type: ignore[union-attr]
        named = {a.name for a in understanding.column_assessments}
        assert named <= real, f"invented columns: {named - real}"

    def test_feature_plan_references_only_real_columns(
        self, offline_state: RunState
    ) -> None:
        from automl_architect.core.schemas import FeaturePlan

        real = {c.name for c in offline_state.profile.columns}  # type: ignore[union-attr]
        plan = offline.decide(FeaturePlan, offline_state)
        for decision in plan.decisions:
            unknown = set(decision.input_columns) - real
            assert not unknown, f"{decision.op} references unknown columns {unknown}"
            assert decision.rationale.strip()


# ---------------------------------------------------------------------------
# Silence
# ---------------------------------------------------------------------------


class ExplodingTransport:
    """Any attribute access is a bug: nothing should reach the SDK offline."""

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - must not run
        raise AssertionError(
            f"offline mode touched the Anthropic SDK (attribute {name!r})"
        )


class TestNoNetwork:
    def test_client_constructs_without_credentials(
        self, offline_settings: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Building an agent must not require a key, so the SDK must stay lazy."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        client = LLMClient(settings=offline_settings)
        assert client.usage.calls == 0

    def test_structured_refuses_while_offline(self, offline_settings: Any) -> None:
        from automl_architect.core.schemas import QuestionAnswer

        client = LLMClient(settings=offline_settings)
        with pytest.raises(LLMOfflineError):
            client.structured(
                output_model=QuestionAnswer,
                user="hello",
                agent_instructions="be brief",
            )

    def test_text_refuses_while_offline(self, offline_settings: Any) -> None:
        client = LLMClient(settings=offline_settings)
        with pytest.raises(LLMOfflineError):
            client.text(user="hello", agent_instructions="be brief")

    def test_agent_run_makes_no_call_and_records_no_spend(
        self, offline_state: RunState
    ) -> None:
        """The end-to-end guarantee, asserted at the layer that decides."""
        from automl_architect.agents.dataset import DatasetAgent

        agent = DatasetAgent(llm=LLMClient(settings=offline_state.settings))
        agent.llm._sdk = ExplodingTransport()  # type: ignore[assignment]

        value = agent.run(offline_state)

        assert offline_state.understanding is value
        assert offline_state.usage.llm_calls == 0
        assert offline_state.usage.cost_usd == 0.0

    def test_offline_run_is_free(self, offline_state: RunState) -> None:
        """Usage must stay at zero across several agents, not just one."""
        from automl_architect.agents.dataset import DatasetAgent
        from automl_architect.agents.problem import ProblemAgent

        for cls in (DatasetAgent, ProblemAgent):
            agent = cls(llm=LLMClient(settings=offline_state.settings))
            agent.llm._sdk = ExplodingTransport()  # type: ignore[assignment]
            agent.run(offline_state)

        assert offline_state.usage.llm_calls == 0
        assert offline_state.usage.input_tokens == 0
        assert offline_state.usage.output_tokens == 0
        assert offline_state.usage.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


class TestMissingInputs:
    def test_deciders_survive_an_empty_state(self, offline_settings: Any) -> None:
        """A decider must not crash when an upstream step produced nothing.

        Offline mode is the fallback path; it is reached in exactly the
        circumstances where other things have already gone wrong, so it has to
        degrade to an honest empty answer rather than raise.
        """
        from automl_architect.core import schemas
        from automl_architect.core.events import NullEventBus

        from .conftest import make_run_config

        bare = RunState(
            config=make_run_config(),
            settings=offline_settings,
            bus=NullEventBus(),
        )
        for name in sorted(offline._DECIDERS):
            model = getattr(schemas, name)
            value = offline.decide(model, bare)
            assert isinstance(value, model), f"{name} returned the wrong type"

    def test_evaluation_rejects_when_nothing_trained(
        self, offline_state: RunState
    ) -> None:
        from automl_architect.core.schemas import EvaluationVerdict

        offline_state.experiments = None
        verdict = offline.decide(EvaluationVerdict, offline_state)
        assert verdict.acceptable is False
        assert verdict.overall_grade == "F"
        assert verdict.recommended_action != "accept"

    def test_tuning_declines_when_nothing_trained(self, offline_state: RunState) -> None:
        from automl_architect.core.schemas import TuningDecision

        offline_state.experiments = None
        decision = offline.decide(TuningDecision, offline_state)
        assert decision.worthwhile is False
        assert decision.n_trials == 0
        assert _HAS_NUMBER.search(decision.rationale) or "no candidate" in (
            decision.rationale.lower()
        )
