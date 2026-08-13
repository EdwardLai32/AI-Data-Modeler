"""Agent behaviour, exercised entirely offline against the fake LLM.

Two layers of testing here.

The first exercises ``core/agent.py`` itself with a purpose-built stub subclass.
That code is written and can be tested today: event emission, usage accounting,
the grounding filter, and the prompt-cache block ordering are all cross-cutting
guarantees that every agent inherits, so a bug there is a bug thirteen times over.

The second discovers whatever agent classes exist in ``automl_architect.agents``
and applies the same invariants to each without knowing what any of them do. That
is deliberate: an agent test that asserts on prompt wording breaks every time the
prompt is tuned, while an agent test that asserts "you never pass a hallucinated
column name to pandas" stays true forever.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest
from pydantic import BaseModel

from automl_architect.core.agent import BaseAgent, HybridAgent
from automl_architect.core.errors import AgentError, LLMTransientError
from automl_architect.core.events import EventKind
from automl_architect.core.llm import PLATFORM_PREAMBLE
from automl_architect.core.schemas import AgentName, QuestionAnswer
from automl_architect.core.state import RunState

from .conftest import CANNED_BUILDERS, FakeLLMClient, requires_live, synthesise, try_import

BOGUS_COLUMN = "__a_column_that_does_not_exist__"

#: List fields holding references to real dataframe columns. Anything an agent
#: puts here has to survive the grounding filter before it reaches pandas.
COLUMN_LIST_FIELDS = frozenset(
    {
        "columns",
        "input_columns",
        "columns_to_drop",
        "affected_columns",
        "suggested_target_columns",
        "temporal_columns",
        "geo_columns",
        "text_columns",
        "identifier_columns",
    }
)

#: Single-column references. Worse than the list case when wrong: a target column
#: that does not exist ends the run instead of degrading one decision.
COLUMN_SCALAR_FIELDS = frozenset(
    {"target_column", "temporal_column", "group_column"}
)

#: Fields holding a list of *submodels* whose named inner field is a column
#: reference. These need naming explicitly: a bare "any field called ``name``"
#: rule would fire on the many unrelated ``name`` fields in these schemas
#: (``TargetSummary.name``, ``ColumnProfile.name``, plan step names), so the
#: mapping is parent field -> inner field. ``DatasetUnderstanding``'s
#: per-column assessments are the case that matters: ``DatasetAgent.postprocess``
#: drops assessments naming an unknown column, and without this the suite never
#: exercised that filter.
COLUMN_SUBMODEL_FIELDS: dict[str, str] = {"column_assessments": "name"}

COLUMN_REFERENCE_FIELDS = COLUMN_LIST_FIELDS | COLUMN_SCALAR_FIELDS


# ===========================================================================
# The base class
# ===========================================================================


class StubAgent(BaseAgent[QuestionAnswer]):
    """A minimal agent, used to test what every agent inherits."""

    name = AgentName.INSIGHT
    output_model = QuestionAnswer
    effort = "medium"
    max_tokens = 2_000

    def __init__(self, llm: Any = None) -> None:
        super().__init__(llm=llm)
        self.applied: list[QuestionAnswer] = []
        self.postprocessed = 0

    def instructions(self, state: RunState) -> str:
        return "You answer one question about the run. Cite the evidence you used."

    def build_prompt(self, state: RunState) -> str:
        return f"Dataset has {state.profile.n_rows} rows. Why was this metric chosen?"

    def postprocess(self, value: QuestionAnswer, state: RunState) -> QuestionAnswer:
        self.postprocessed += 1
        return value

    def apply(self, state: RunState, value: QuestionAnswer) -> None:
        self.applied.append(value)


class StubHybridAgent(HybridAgent[QuestionAnswer]):
    """A hybrid agent, to prove the executor runs before the reasoning."""

    name = AgentName.EXPERIMENT
    output_model = QuestionAnswer

    def __init__(self, llm: Any = None) -> None:
        super().__init__(llm=llm)
        self.order: list[str] = []

    def compute(self, state: RunState) -> None:
        self.order.append("compute")

    def instructions(self, state: RunState) -> str:
        return "Interpret the measured results."

    def build_prompt(self, state: RunState) -> str:
        self.order.append("build_prompt")
        return "What do these results mean?"

    def apply(self, state: RunState, value: QuestionAnswer) -> None:
        self.order.append("apply")


class TestBaseAgent:
    def test_run_reasons_grounds_and_applies(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        agent = StubAgent(llm=fake_llm)
        value = agent.run(run_state)
        assert isinstance(value, QuestionAnswer)
        assert agent.postprocessed == 1
        assert agent.applied == [value]

    def test_title_defaults_from_the_name(self, fake_llm: FakeLLMClient) -> None:
        assert StubAgent(llm=fake_llm).title == "Insight Agent"

    def test_usage_accumulates_onto_the_state(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        """Cost reporting is a product surface; a dropped call is a wrong invoice."""
        agent = StubAgent(llm=fake_llm)
        agent.run(run_state)
        agent.run(run_state)
        assert run_state.usage.llm_calls == 2
        assert run_state.usage.input_tokens > 0
        assert run_state.usage.output_tokens > 0
        assert run_state.usage.cost_usd > 0.0

    def test_second_call_reads_the_prompt_cache(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        """The whole point of the frozen run context is a cache hit on call two."""
        agent = StubAgent(llm=fake_llm)
        agent.run(run_state)
        first_reads = run_state.usage.cache_read_tokens
        agent.run(run_state)
        assert run_state.usage.cache_read_tokens > first_reads

    def test_emits_the_event_trail(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        StubAgent(llm=fake_llm).run(run_state)
        kinds = [event.kind for event in run_state.bus.events]
        assert kinds.count(EventKind.LLM_CALL) == 2, "expected a start and a finish event"
        assert EventKind.AGENT_THINKING in kinds
        assert EventKind.AGENT_DECISION in kinds

    def test_decision_event_carries_a_summary(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        StubAgent(llm=fake_llm).run(run_state)
        decisions = run_state.bus.of_kind(EventKind.AGENT_DECISION)
        assert decisions and decisions[-1].message.strip()
        assert decisions[-1].agent is AgentName.INSIGHT

    def test_llm_call_event_records_tokens_and_cost(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        StubAgent(llm=fake_llm).run(run_state)
        finished = [e for e in run_state.bus.of_kind(EventKind.LLM_CALL) if e.tokens_out]
        assert finished
        assert finished[-1].cost_usd is not None and finished[-1].cost_usd > 0

    def test_transport_failure_becomes_an_agent_error(
        self, fake_llm: FakeLLMClient, run_state: RunState
    ) -> None:
        """The orchestrator branches on AgentError, so the wrapping has to happen."""
        fake_llm.fail_next(LLMTransientError("overloaded"))
        with pytest.raises(AgentError) as info:
            StubAgent(llm=fake_llm).run(run_state)
        assert "insight" in str(info.value)

    def test_agent_error_is_not_double_wrapped(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        fake_llm.fail_next(AgentError("insight", "already contextualised"))
        with pytest.raises(AgentError) as info:
            StubAgent(llm=fake_llm).run(run_state)
        assert str(info.value).count("[insight]") == 1

    def test_effort_and_budget_reach_the_client(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        StubAgent(llm=fake_llm).run(run_state)
        call = fake_llm.last_call_for(QuestionAnswer)
        assert call.effort == "medium"
        assert call.max_tokens == 2_000

    def test_apply_is_mandatory(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        """An agent that forgets to write its result back must fail loudly."""

        class Forgetful(StubAgent):
            def apply(self, state: RunState, value: QuestionAnswer) -> None:
                BaseAgent.apply(self, state, value)

        with pytest.raises(NotImplementedError):
            Forgetful(llm=fake_llm).run(run_state)


class TestGrounding:
    def test_unknown_columns_are_dropped(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        agent = StubAgent(llm=fake_llm)
        kept = agent.keep_known_columns(["tenure_months", BOGUS_COLUMN], run_state)
        assert kept == ["tenure_months"]

    def test_dropping_a_column_records_a_warning(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        """Silent filtering hides a misbehaving prompt; the warning is the signal."""
        agent = StubAgent(llm=fake_llm)
        agent.keep_known_columns([BOGUS_COLUMN], run_state, context="cleaning plan")
        assert run_state.warnings
        assert BOGUS_COLUMN in run_state.warnings[-1]
        assert "cleaning plan" in run_state.warnings[-1]

    def test_all_known_columns_produce_no_warning(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        agent = StubAgent(llm=fake_llm)
        agent.keep_known_columns(["tenure_months", "region"], run_state)
        assert not run_state.warnings

    def test_known_columns_fall_back_to_the_profile(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        """Before the frame is loaded the profile is the only schema available."""
        run_state.raw_df = None
        run_state.working_df = None
        run_state.feature_frame = None
        agent = StubAgent(llm=fake_llm)
        assert "tenure_months" in agent.known_columns(run_state)

    def test_no_schema_means_no_filtering(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        """With nothing to validate against, passing everything through beats dropping all."""
        run_state.raw_df = None
        run_state.working_df = None
        run_state.feature_frame = None
        run_state.profile = None
        agent = StubAgent(llm=fake_llm)
        assert agent.keep_known_columns([BOGUS_COLUMN], run_state) == [BOGUS_COLUMN]


class TestPromptAssembly:
    def test_blocks_are_ordered_stable_first(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        blocks = StubAgent(llm=fake_llm).system_blocks(run_state)
        assert len(blocks) == 3
        assert blocks[0].text == PLATFORM_PREAMBLE
        assert blocks[1].text == run_state.run_context
        assert "answer one question" in blocks[2].text

    def test_cache_breakpoints_sit_on_the_stable_blocks(
        self, fake_llm: FakeLLMClient, run_state: RunState
    ) -> None:
        blocks = StubAgent(llm=fake_llm).system_blocks(run_state)
        assert blocks[0].cache is True
        assert blocks[1].cache is True
        assert blocks[2].cache is False, "agent instructions vary and must not hold a breakpoint"

    def test_stable_blocks_are_byte_identical_across_calls(
        self, fake_llm: FakeLLMClient, run_state: RunState
    ) -> None:
        """Any churn in blocks 0-1 costs every agent in the run a cache miss."""
        agent = StubAgent(llm=fake_llm)
        first = agent.system_blocks(run_state)
        second = agent.system_blocks(run_state)
        assert first[0].text == second[0].text
        assert first[1].text == second[1].text

    def test_run_context_carries_measured_facts(self, run_state: RunState) -> None:
        context = run_state.run_context
        assert context is not None
        assert "MEASURED DATASET FACTS" in context
        assert "tenure_months" in context
        assert "3,000" in context, "the real row count must be in the cached prefix"

    def test_frozen_context_is_immutable(self, run_state: RunState) -> None:
        original = run_state.run_context
        run_state.freeze_context("a different digest entirely")
        assert run_state.run_context == original


class TestHybridAgent:
    def test_computation_precedes_reasoning(self, fake_llm: FakeLLMClient, run_state: RunState) -> None:
        """The point of a hybrid agent is that it interprets measured results."""
        agent = StubHybridAgent(llm=fake_llm)
        agent.run(run_state)
        assert agent.order == ["compute", "build_prompt", "apply"]


# ===========================================================================
# Whatever agents exist
# ===========================================================================


def discover_agents() -> list[type[BaseAgent]]:
    """Find every concrete ``BaseAgent`` subclass in ``automl_architect.agents``."""
    package = try_import("automl_architect.agents")
    if package is None:
        return []

    found: dict[str, type[BaseAgent]] = {}
    module_names = [
        info.name
        for info in pkgutil.iter_modules(getattr(package, "__path__", []))
        if not info.name.startswith("_")
    ]
    for module_name in module_names:
        try:
            module = importlib.import_module(f"automl_architect.agents.{module_name}")
        except ImportError:  # a sibling module it depends on may not exist yet
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseAgent)
                and obj not in (BaseAgent, HybridAgent)
                and not inspect.isabstract(obj)
                and getattr(obj, "output_model", None) is not None
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return list(found.values())


AGENTS = discover_agents()
agent_params = pytest.mark.parametrize("agent_cls", AGENTS, ids=lambda c: c.__name__)
needs_agents = pytest.mark.skipif(not AGENTS, reason="automl_architect.agents not written yet")


@needs_agents
def test_every_agent_name_resolves_through_the_registry() -> None:
    """The registry is the orchestrator's only way to find an agent.

    Tested against ``agents.AGENTS`` rather than against class discovery because
    the registry is the real contract surface: it resolves lazily, so a broken
    entry surfaces as a failed step mid-run rather than at import.
    """
    package = try_import("automl_architect.agents")
    registry = getattr(package, "AGENTS", None)
    if registry is None:
        pytest.skip("agents package exposes no AGENTS registry")

    for name in AgentName:
        agent_cls = registry[name]
        assert isinstance(agent_cls, type), f"{name.value} did not resolve to a class"
        assert issubclass(agent_cls, BaseAgent), f"{name.value} -> {agent_cls!r}"
        assert agent_cls.name is name, (
            f"registry maps {name.value} to {agent_cls.__name__}, which declares "
            f"name={agent_cls.name.value}"
        )


@needs_agents
def test_registry_rejects_an_unknown_agent_name() -> None:
    from automl_architect.core.errors import ConfigurationError

    package = try_import("automl_architect.agents")
    registry = getattr(package, "AGENTS", None)
    if registry is None:
        pytest.skip("agents package exposes no AGENTS registry")
    with pytest.raises((ConfigurationError, KeyError)):
        registry["not_an_agent"]


@needs_agents
@agent_params
def test_agent_declares_its_contract(agent_cls: type[BaseAgent]) -> None:
    assert isinstance(agent_cls.name, AgentName)
    assert issubclass(agent_cls.output_model, BaseModel)
    assert agent_cls.effort in ("low", "medium", "high", "xhigh", "max")
    assert agent_cls.max_tokens > 0


@needs_agents
@agent_params
def test_instructions_are_substantial(agent_cls: type[BaseAgent], fake_llm: FakeLLMClient, run_state: RunState) -> None:
    """The instruction block is the agent's whole definition; a stub is a bug."""
    instructions = agent_cls(llm=fake_llm).instructions(run_state)
    assert isinstance(instructions, str)
    assert len(instructions) > 200, f"{agent_cls.__name__} instructions are only {len(instructions)} chars"


@needs_agents
@agent_params
def test_prompt_is_built_from_state(agent_cls: type[BaseAgent], fake_llm: FakeLLMClient, run_state: RunState) -> None:
    prompt = agent_cls(llm=fake_llm).build_prompt(run_state)
    assert isinstance(prompt, str)
    assert prompt.strip(), f"{agent_cls.__name__} built an empty prompt"


def _skip_if_degraded(llm: FakeLLMClient, state: RunState, agent_cls: type[BaseAgent]) -> None:
    """Skip when an agent legitimately declined to reason.

    Some agents interpret executor output that this fixture does not provide — the
    Explainability Agent with no fitted model, for instance. Rule 3 says that must
    degrade rather than crash, so returning a valid empty result *with a recorded
    warning* is correct behaviour and the prompt assertions do not apply. Skipping
    silently would hide a real bug, so the warning is required.
    """
    if llm.calls:
        return
    assert state.warnings, (
        f"{agent_cls.__name__} produced a result without reasoning and without "
        "recording why it degraded"
    )
    pytest.skip(f"{agent_cls.__name__} degraded (no executor input in this fixture)")


@needs_agents
@agent_params
def test_agent_runs_and_writes_to_the_state(
    agent_cls: type[BaseAgent], installed_fake_llm: FakeLLMClient, run_state: RunState
) -> None:
    """A full offline cycle for every agent: reason, ground, apply."""
    value = agent_cls(llm=installed_fake_llm).run(run_state)
    assert isinstance(value, agent_cls.output_model)


@needs_agents
@agent_params
def test_agent_asks_for_its_declared_output_model(
    agent_cls: type[BaseAgent], installed_fake_llm: FakeLLMClient, run_state: RunState
) -> None:
    agent_cls(llm=installed_fake_llm).run(run_state)
    _skip_if_degraded(installed_fake_llm, run_state, agent_cls)
    assert installed_fake_llm.calls[-1].output_model is agent_cls.output_model


@needs_agents
@agent_params
def test_agent_prompt_includes_the_cached_context(
    agent_cls: type[BaseAgent], installed_fake_llm: FakeLLMClient, run_state: RunState
) -> None:
    """Reasoning without the measured facts is exactly what this system forbids."""
    agent_cls(llm=installed_fake_llm).run(run_state)
    _skip_if_degraded(installed_fake_llm, run_state, agent_cls)
    call = installed_fake_llm.calls[-1]
    assert PLATFORM_PREAMBLE in call.system_text
    assert run_state.run_context in call.system_text


def _inject_bogus_columns(value: BaseModel) -> BaseModel:
    """Return a copy of ``value`` with a nonexistent column in every reference field.

    Both shapes are covered: list fields get the bogus name appended, and scalar
    fields are overwritten with it. The scalar case is the more dangerous of the
    two — ``ProblemDefinition.target_column`` pointing at a column that does not
    exist ends the run rather than degrading a single decision.
    """
    payload = value.model_dump()

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key, sub in list(node.items()):
                if key in COLUMN_LIST_FIELDS and isinstance(sub, list):
                    node[key] = [*sub, BOGUS_COLUMN]
                elif key in COLUMN_SCALAR_FIELDS and isinstance(sub, str | None):
                    node[key] = BOGUS_COLUMN
                elif key in COLUMN_SUBMODEL_FIELDS and isinstance(sub, list) and sub:
                    # Clone an existing entry so the appended one stays schema-valid
                    # whatever required fields the submodel grows.
                    inner = COLUMN_SUBMODEL_FIELDS[key]
                    node[key] = [*sub, {**sub[0], inner: BOGUS_COLUMN}]
                else:
                    node[key] = walk(sub)
            return node
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return type(value).model_validate(walk(payload))


def _collect_column_references(value: BaseModel) -> list[str]:
    """Every string in a column-reference field, anywhere in the object tree."""
    references: list[str] = []

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, dict):
            for sub_key, sub in node.items():
                if sub_key in COLUMN_SCALAR_FIELDS and isinstance(sub, str):
                    references.append(sub)
                else:
                    walk(sub, sub_key)
        elif isinstance(node, list):
            if key in COLUMN_LIST_FIELDS:
                references.extend(str(item) for item in node if isinstance(item, str))
            elif key in COLUMN_SUBMODEL_FIELDS:
                inner = COLUMN_SUBMODEL_FIELDS[key]
                references.extend(
                    str(item[inner])
                    for item in node
                    if isinstance(item, dict) and isinstance(item.get(inner), str)
                )
            else:
                for item in node:
                    walk(item, key)

    walk(value.model_dump())
    return references


@needs_agents
@agent_params
def test_hallucinated_columns_never_survive_postprocess(
    agent_cls: type[BaseAgent], installed_fake_llm: FakeLLMClient, run_state: RunState
) -> None:
    """The most damaging failure mode in the system, tested for every agent.

    An agent naming a column that does not exist must have that reference filtered
    out before an executor hands it to pandas. Agents whose output holds no column
    references are trivially fine and skip.

    ``FeatureAgent`` is the one exemption, and it is a real architectural
    distinction rather than a carve-out for convenience. Feature engineering is
    sequential: ``standard_scale`` legitimately consumes a column that an earlier
    ``log_transform`` in the same plan creates, and at postprocess time that
    column does not exist yet. Filtering here deleted exactly those references
    and broke multi-stage plans. The guarantee therefore lives in
    ``execution.feature_ops._resolve_columns``, which checks against the real
    frame after the creating ops have run — see
    :func:`test_executor_drops_hallucinated_feature_columns`, which holds the
    same invariant at that layer.
    """
    if agent_cls.__name__ == "FeatureAgent":
        pytest.skip("guaranteed by execution.feature_ops._resolve_columns instead")

    model = agent_cls.output_model
    baseline = CANNED_BUILDERS[model]() if model in CANNED_BUILDERS else synthesise(model)
    if BOGUS_COLUMN in _collect_column_references(_inject_bogus_columns(baseline)):
        pass
    else:
        pytest.skip(f"{model.__name__} carries no column references")

    installed_fake_llm.register(model, _inject_bogus_columns(baseline), sticky=True)
    value = agent_cls(llm=installed_fake_llm).run(run_state)

    surviving = _collect_column_references(value)
    assert BOGUS_COLUMN not in surviving, (
        f"{agent_cls.__name__} passed {BOGUS_COLUMN} through to the executor layer"
    )


def test_executor_drops_hallucinated_feature_columns(run_state: RunState) -> None:
    """The FeatureAgent half of the no-hallucinated-columns invariant.

    ``FeatureAgent.postprocess`` deliberately stops filtering column references,
    so this proves the guarantee still holds one layer down: the executor resolves
    against the real frame, drops what does not exist, warns, and does not raise.
    """
    pd = pytest.importorskip("pandas")
    feature_ops = try_import("automl_architect.execution.feature_ops")
    if feature_ops is None:
        pytest.skip("execution.feature_ops not available")

    from automl_architect.core.schemas import FeatureDecision, FeatureOp, FeaturePlan

    run_state.working_df = pd.DataFrame(
        {"tenure_months": [1, 2, 3, 4], "annual_income": [10.0, 20.0, 30.0, 40.0]}
    )
    plan = FeaturePlan(
        summary="scale a real column and a fabricated one",
        decisions=[
            FeatureDecision(
                op=FeatureOp.STANDARD_SCALE,
                input_columns=["tenure_months", BOGUS_COLUMN],
                rationale="Scale the numeric inputs to a common range.",
            )
        ],
    )

    frame = feature_ops.apply_feature_plan(run_state, plan)

    assert BOGUS_COLUMN not in set(map(str, frame.columns))
    assert BOGUS_COLUMN not in set(run_state.feature_names or [])
    assert any(BOGUS_COLUMN in warning for warning in run_state.warnings), (
        "the executor dropped the column silently; the warning is the operator's "
        "only signal that an agent is hallucinating"
    )


def test_sequential_feature_columns_survive_postprocess(
    fake_llm: FakeLLMClient, run_state: RunState
) -> None:
    """A later op may consume a column an earlier op in the same plan creates.

    Regression test for a defect a live run exposed: postprocess validated inputs
    against the source schema, so it stripped ``annual_income_log1p`` from a
    ``standard_scale`` that a preceding ``log_transform`` was about to produce.
    """
    agents_mod = try_import("automl_architect.agents")
    if agents_mod is None:
        pytest.skip("agents package not available")
    agent_cls = agents_mod.get_agent(AgentName.FEATURES)

    from automl_architect.core.schemas import FeatureDecision, FeatureOp, FeaturePlan

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

    result = agent_cls(llm=fake_llm).postprocess(plan, run_state)
    scale = next((d for d in result.decisions if d.op is FeatureOp.STANDARD_SCALE), None)

    assert scale is not None, "the standard_scale decision was dropped entirely"
    assert "annual_income_log1p" in scale.input_columns, (
        "postprocess stripped a column the plan's own log_transform creates; "
        f"surviving inputs were {scale.input_columns}"
    )


@needs_agents
def test_planner_and_evaluator_reason_hardest() -> None:
    """Effort is a cost lever; the two agents that shape the run get the budget.

    ``core/agent.py`` says as much in its docstring: "Planning and evaluation
    warrant more than narration."
    """
    by_name = {agent_cls.name: agent_cls for agent_cls in AGENTS}
    for name in (AgentName.PLANNER, AgentName.EVALUATION):
        agent_cls = by_name.get(name)
        if agent_cls is None:
            pytest.skip(f"{name.value} agent not implemented yet")
        assert agent_cls.effort in ("high", "xhigh", "max"), (
            f"{agent_cls.__name__} reasons at effort={agent_cls.effort}"
        )


@needs_agents
@agent_params
def test_agent_summary_is_one_line(
    agent_cls: type[BaseAgent], fake_llm: FakeLLMClient, run_state: RunState
) -> None:
    """``decision_summary`` goes into the live event stream and the UI."""
    model = agent_cls.output_model
    value = CANNED_BUILDERS[model]() if model in CANNED_BUILDERS else synthesise(model)
    summary = agent_cls(llm=fake_llm).decision_summary(value)
    assert isinstance(summary, str) and summary.strip()
    assert "\n" not in summary.strip(), "a summary with newlines breaks the event stream layout"


@needs_agents
@agent_params
def test_agent_reasons_through_the_injected_client(
    agent_cls: type[BaseAgent], installed_fake_llm: FakeLLMClient, run_state: RunState
) -> None:
    """Every LLM interaction goes through the injected client, or is explained.

    The offline guarantee rests on this: an agent that fabricated a result some
    other way would either be reaching the network or inventing content, and both
    are failures. Declining to reason is allowed only with a recorded warning.
    """
    agent_cls(llm=installed_fake_llm).run(run_state)
    if not installed_fake_llm.calls:
        assert run_state.warnings, (
            f"{agent_cls.__name__} produced a result without calling the client "
            "and without recording why"
        )


# ===========================================================================
# The one test that is allowed to touch the network
# ===========================================================================


@pytest.mark.live
@requires_live
def test_live_structured_call_returns_a_validated_model(run_state: RunState) -> None:
    """A real API round trip, as a credential and contract smoke check.

    Skipped unless ``AUTOML_LIVE_TESTS=1`` and ``ANTHROPIC_API_KEY_LIVE`` are both
    set, because it costs money and needs network access — the rest of this suite
    is offline by construction and must stay that way.

    What it verifies is not agent quality but the two things a fake cannot: that
    the real ``ProblemDefinition`` schema is accepted by the structured-output
    compiler, and that a live response validates into it. A schema that the
    compiler rejects would fail every run while passing every offline test, which
    is exactly the gap this closes.
    """
    import os

    from automl_architect.config import Settings
    from automl_architect.core.llm import LLMClient
    from automl_architect.core.schemas import ProblemDefinition

    # The offline fixture points the SDK at a dead port; undo that for this test
    # only, and use the separate live key so a stray offline test cannot spend.
    os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY_LIVE"]
    os.environ.pop("ANTHROPIC_BASE_URL", None)

    client = LLMClient(settings=Settings(default_effort="low", max_output_tokens=4_000))
    result = client.structured(
        output_model=ProblemDefinition,
        user=(
            "A table of 3,000 subscribers has a `churned` column taking values 0 "
            "(2,221 rows) and 1 (779 rows). Identify the task and the metric."
        ),
        agent_instructions="You identify the machine-learning task a dataset poses.",
        effort="low",
        max_tokens=4_000,
    )

    value = result.value
    assert isinstance(value, ProblemDefinition)
    assert value.rationale.strip(), "the live response carried no reasoning"
    assert value.primary_metric.strip()
    assert result.usage.calls == 1
    assert result.usage.cost_usd > 0.0


class TestSchemaCompilationRetry:
    """A live run lost the Evaluation Agent to `Grammar compilation timed out`.

    Structured output compiles the JSON schema into a sampling grammar
    server-side, and a large schema can exceed that budget under load. The API
    reports it as ``invalid_request_error``, which looks permanent and is not:
    compilation is cached once it succeeds. Classifying it as fatal cost the run
    its quality gate, so the classification itself is worth a test.
    """

    @staticmethod
    def _client(monkeypatch: pytest.MonkeyPatch, errors: list[Exception], result: Any):
        """An LLMClient whose transport raises `errors` then returns `result`."""
        import anthropic

        from automl_architect.core import llm as llm_module

        calls = {"n": 0}

        def fake_parse(**_kwargs: Any) -> Any:
            index = calls["n"]
            calls["n"] += 1
            if index < len(errors):
                raise errors[index]
            return result

        monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
        client = llm_module.LLMClient.__new__(llm_module.LLMClient)
        client.settings = llm_module.get_settings()
        client.usage = llm_module.Usage()

        class _Messages:
            parse = staticmethod(fake_parse)

        class _Beta:
            messages = _Messages()

        class _Transport:
            beta = _Beta()

        client._client = _Transport()  # type: ignore[attr-defined]
        return client, calls

    @staticmethod
    def _bad_request(message: str) -> Exception:
        import anthropic
        import httpx

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(400, request=request, json={"error": {"message": message}})
        return anthropic.BadRequestError(message, response=response, body=None)

    @staticmethod
    def _ok_message() -> Any:
        class _Usage:
            input_tokens = 10
            output_tokens = 5
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0
            iterations: list[Any] = []

        class _Message:
            usage = _Usage()
            stop_reason = "end_turn"
            model = "claude-opus-5"
            content: list[Any] = []
            parsed_output = QuestionAnswer(
                question="q", answer="a", evidence=[], confidence="high", caveats=[]
            )

        return _Message()

    def test_grammar_timeout_is_retried_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, calls = self._client(
            monkeypatch,
            [self._bad_request("Grammar compilation timed out")],
            self._ok_message(),
        )
        result = client.structured(
            output_model=QuestionAnswer,
            user="why?",
            agent_instructions="you answer questions",
        )
        assert result.value.answer == "a"
        assert calls["n"] == 2, "the compile timeout should have been retried once"

    def test_a_genuinely_malformed_request_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrying a real client error just burns budget on the same failure."""
        from automl_architect.core.errors import LLMOutputError

        client, calls = self._client(
            monkeypatch,
            [self._bad_request("max_tokens: must be greater than 0")],
            self._ok_message(),
        )
        with pytest.raises(LLMOutputError):
            client.structured(
                output_model=QuestionAnswer,
                user="why?",
                agent_instructions="you answer questions",
            )
        assert calls["n"] == 1, "a malformed request must fail on the first attempt"
