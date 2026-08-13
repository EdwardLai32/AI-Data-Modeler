"""Contract invariants for ``core/schemas.py``.

These tests guard the two properties the whole architecture rests on, in a way
that a schema edit cannot silently break:

*   **Every decision carries its reasoning.** ``rationale`` is required on each
    decision model. If someone gives it a default, :func:`test_rationale_is_required`
    fails — which is the point, because an agent could then emit an unexplained
    transformation and nothing else in the system would notice.
*   **Agent-facing models stay inside Claude's structured-output schema subset.**
    No free-form objects, no recursion. Tests here walk the generated JSON schema
    rather than reading the source, so a new model is covered automatically.
"""

from __future__ import annotations

import inspect
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from automl_architect.core import context as context_module
from automl_architect.core import schemas as S

from .conftest import synthesise

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def all_models() -> list[type[BaseModel]]:
    """Every Pydantic model defined in ``core.schemas``."""
    return [
        obj
        for _, obj in inspect.getmembers(S, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == S.__name__ and obj is not S.Base
    ]


#: Models an agent returns via structured output. These carry the tightest
#: constraints because the Claude schema compiler has to accept them.
AGENT_OUTPUT_MODELS: list[type[BaseModel]] = [
    S.DatasetUnderstanding,
    S.ProblemDefinition,
    S.ExecutionPlan,
    S.CleaningPlan,
    S.FeaturePlan,
    S.ModelSelection,
    S.TuningDecision,
    S.ExplainabilityReport,
    S.EvaluationVerdict,
    S.InsightReport,
    S.VisualizationPlan,
    S.FinalReport,
    S.MemorySuggestion,
    S.QuestionAnswer,
]

#: Models that represent one explained choice. The rationale rule applies here.
DECISION_MODELS: list[type[BaseModel]] = [
    S.ProblemDefinition,
    S.PlanStep,
    S.CleaningDecision,
    S.FeatureDecision,
    S.ModelCandidate,
    S.TuningDecision,
    S.ChartSpec,
    S.DeploymentRecommendation,
]


def test_all_models_discovered() -> None:
    """Sanity check on the discovery helper itself."""
    models = all_models()
    assert len(models) > 40, f"expected the full contract surface, found {len(models)}"
    assert S.DatasetProfile in models
    assert S.RunSummary in models


# ---------------------------------------------------------------------------
# The reasoning requirement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", DECISION_MODELS, ids=lambda m: m.__name__)
def test_rationale_is_required(model: type[BaseModel]) -> None:
    """A decision model cannot be constructed without its reasoning."""
    field = model.model_fields.get("rationale")
    assert field is not None, f"{model.__name__} has no rationale field"
    assert field.is_required(), (
        f"{model.__name__}.rationale has a default, so an agent could emit an "
        "unexplained decision"
    )


@pytest.mark.parametrize("model", DECISION_MODELS, ids=lambda m: m.__name__)
def test_decision_rejects_missing_rationale(model: type[BaseModel]) -> None:
    """Omitting the rationale is a validation error, not a silent empty string."""
    payload = synthesise(model).model_dump()
    payload.pop("rationale")
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_evaluation_verdict_requires_its_argument() -> None:
    """The quality gate must justify accept/reject and name a next action."""
    for name in ("verdict_rationale", "acceptable", "overall_grade", "recommended_action"):
        assert S.EvaluationVerdict.model_fields[name].is_required(), name


# ---------------------------------------------------------------------------
# Structured-output schema subset
# ---------------------------------------------------------------------------


def _walk(node: Any) -> list[dict[str, Any]]:
    """Every dict node in a JSON schema, depth-first."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        found.append(node)
        for value in node.values():
            found.extend(_walk(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk(item))
    return found


@pytest.mark.parametrize("model", AGENT_OUTPUT_MODELS, ids=lambda m: m.__name__)
def test_no_free_form_objects(model: type[BaseModel]) -> None:
    """No untyped object anywhere: that is the one thing the subset cannot express.

    ``dict[str, Any]`` compiles to ``{"type": "object"}`` with no ``properties``,
    which Claude's structured-output schema compiler rejects. The project rule is
    to use ``list[Param]`` instead, so this test is the enforcement point.
    """
    schema = model.model_json_schema()
    offenders = [
        node
        for node in _walk(schema)
        if node.get("type") == "object" and "properties" not in node and "$ref" not in node
    ]
    assert not offenders, (
        f"{model.__name__} contains {len(offenders)} free-form object node(s); "
        "use list[Param] instead of dict[str, Any]"
    )


@pytest.mark.parametrize("model", all_models(), ids=lambda m: m.__name__)
def test_no_dict_annotations(model: type[BaseModel]) -> None:
    """No model field is annotated as a mapping, agent-facing or not."""
    for name, field in model.model_fields.items():
        annotation = field.annotation
        candidates = [annotation, *get_args(annotation)]
        for candidate in candidates:
            assert get_origin(candidate) is not dict, (
                f"{model.__name__}.{name} is a dict; the contract uses list[Param]"
            )


@pytest.mark.parametrize("model", AGENT_OUTPUT_MODELS, ids=lambda m: m.__name__)
def test_schema_is_not_recursive(model: type[BaseModel]) -> None:
    """No ``$defs`` entry may reference itself, directly or transitively."""
    schema = model.model_json_schema()
    defs: dict[str, Any] = schema.get("$defs", {})

    def refs_of(node: Any) -> set[str]:
        return {
            n["$ref"].rsplit("/", 1)[-1]
            for n in _walk(node)
            if isinstance(n.get("$ref"), str)
        }

    edges = {name: refs_of(body) for name, body in defs.items()}
    for start in edges:
        seen: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            for nxt in edges.get(current, set()):
                assert nxt != start, f"{model.__name__}: {start} is recursive via {current}"
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)


@pytest.mark.parametrize("model", all_models(), ids=lambda m: m.__name__)
def test_extra_fields_forbidden(model: type[BaseModel]) -> None:
    """Every model forbids extras, so a hallucinated field is a loud error."""
    assert model.model_config.get("extra") == "forbid", model.__name__
    payload = synthesise(model).model_dump()
    payload["a_field_the_model_invented"] = 1
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize("model", all_models(), ids=lambda m: m.__name__)
def test_json_round_trip(model: type[BaseModel]) -> None:
    """Every model survives ``model_dump(mode="json")`` -> ``model_validate``.

    The API and the persisted run summary both rely on this; a field whose JSON
    form will not re-validate breaks run replay without breaking anything local.
    """
    original = synthesise(model)
    revalidated = model.model_validate(original.model_dump(mode="json"))
    assert revalidated.model_dump(mode="json") == original.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Param <-> dict conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", 42),
        ("-7", -7),
        ("3.5", 3.5),
        ("1e-3", 0.001),
        ("true", True),
        ("false", False),
        ("True", True),
        ("null", None),
        ("none", None),
        ("", None),
        ("   ", None),
        ("[1, 2, 3]", [1, 2, 3]),
        ('{"a": 1}', {"a": 1}),
        ("balanced", "balanced"),
        ("auto", "auto"),
        ("not json {", "not json {"),
    ],
)
def test_param_value_coercion(raw: str, expected: Any) -> None:
    """String-on-the-wire values arrive at executors as real Python scalars."""
    assert S.params_to_dict([S.Param(key="k", value=raw)]) == {"k": expected}


def test_params_round_trip_preserves_types() -> None:
    """``dict -> params -> dict`` is the identity for JSON-representable values."""
    original = {
        "n_estimators": 300,
        "learning_rate": 0.08,
        "class_weight": "balanced",
        "early_stopping": True,
        "subsample": None,
        "hidden_layers": [64, 32],
        "nested": {"a": [1, 2]},
    }
    assert S.params_to_dict(S.dict_to_params(original)) == original


def test_params_to_dict_handles_empty() -> None:
    """Both ``None`` and ``[]`` mean "no parameters", not a crash."""
    assert S.params_to_dict(None) == {}
    assert S.params_to_dict([]) == {}
    assert S.dict_to_params(None) == []
    assert S.dict_to_params({}) == []


def test_dict_to_params_stringifies_unjsonable_values() -> None:
    """An unserialisable value degrades to ``str()`` rather than raising."""

    class Opaque:
        def __str__(self) -> str:
            return "opaque-value"

    params = S.dict_to_params({"estimator": Opaque()})
    assert params[0].value == "opaque-value"


def test_param_keeps_strings_verbatim() -> None:
    """A string value is not re-quoted on the way out."""
    assert S.dict_to_params({"strategy": "median"})[0].value == "median"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


ENUMS = [
    obj
    for _, obj in inspect.getmembers(S, inspect.isclass)
    if issubclass(obj, S.Enum) and obj.__module__ == S.__name__
]


@pytest.mark.parametrize("enum_cls", ENUMS, ids=lambda e: e.__name__)
def test_enum_values_are_unique_snake_case(enum_cls: type[S.Enum]) -> None:
    """Enum wire values are unique lower_snake_case strings.

    They appear verbatim in prompts and in persisted JSON, so casing drift would
    both confuse the model and break stored-run deserialisation.
    """
    values = [member.value for member in enum_cls]
    assert len(values) == len(set(values)), f"{enum_cls.__name__} has duplicate values"
    for value in values:
        assert isinstance(value, str)
        assert value == value.lower(), f"{enum_cls.__name__}: {value!r} is not lowercase"
        assert " " not in value and "-" not in value, f"{enum_cls.__name__}: {value!r}"


def test_task_type_classification_flags_agree() -> None:
    """``is_classification`` covers exactly the three classification members."""
    classification = {t for t in S.TaskType if t.is_classification}
    assert classification == {
        S.TaskType.BINARY_CLASSIFICATION,
        S.TaskType.MULTICLASS_CLASSIFICATION,
        S.TaskType.MULTILABEL_CLASSIFICATION,
    }


def test_supported_tasks_are_supervised_or_explicitly_unsupervised() -> None:
    """Every supported task is either supervised or a known unsupervised family."""
    unsupervised_ok = {S.TaskType.CLUSTERING, S.TaskType.ANOMALY_DETECTION}
    for task in S.TaskType:
        if task.is_supported and not task.is_supervised:
            assert task in unsupervised_ok, task


def test_every_severity_is_renderable() -> None:
    """``context.render_profile`` sorts issues by a hand-written severity map.

    That map is a dict lookup with no default, so adding a Severity member
    without updating it turns a data-quality report into a KeyError. This test is
    the tripwire.
    """
    profile = synthesise(S.DatasetProfile)
    profile.quality_issues = [
        S.DataQualityIssue(code=f"c_{sev.value}", severity=sev, columns=[], detail="x")
        for sev in S.Severity
    ]
    rendered = context_module.render_profile(profile)
    for sev in S.Severity:
        assert f"c_{sev.value}" in rendered


# ---------------------------------------------------------------------------
# Accessor helpers
# ---------------------------------------------------------------------------


def test_execution_plan_ordered_sorts_by_order() -> None:
    """``ordered()`` is the orchestrator's execution sequence, not list order."""
    plan = S.ExecutionPlan(
        summary="s",
        steps=[
            S.PlanStep(step_id="c", order=3, title="c", agent=S.AgentName.REPORT, objective="o", rationale="r"),
            S.PlanStep(step_id="a", order=1, title="a", agent=S.AgentName.CLEANING, objective="o", rationale="r"),
            S.PlanStep(step_id="b", order=2, title="b", agent=S.AgentName.FEATURES, objective="o", rationale="r"),
        ],
        fallback_strategy="f",
    )
    assert [s.step_id for s in plan.ordered()] == ["a", "b", "c"]


def test_final_report_ordered_sections() -> None:
    report = synthesise(S.FinalReport)
    report.sections = [
        S.ReportSection(heading="second", order=2, body_markdown="b"),
        S.ReportSection(heading="first", order=1, body_markdown="a"),
    ]
    assert [s.heading for s in report.ordered_sections()] == ["first", "second"]


def test_experiment_log_best_and_metric_lookup() -> None:
    """``best()`` resolves by id, and a missing metric returns None not KeyError."""
    winner = S.ExperimentResult(
        family=S.ModelFamily.RANDOM_FOREST,
        metrics=[S.MetricValue(name="roc_auc", value=0.71)],
        primary_metric="roc_auc",
        primary_score=0.71,
    )
    loser = S.ExperimentResult(family=S.ModelFamily.BASELINE_DUMMY, primary_score=0.5)
    log = S.ExperimentLog(
        results=[loser, winner],
        best_experiment_id=winner.experiment_id,
        primary_metric="roc_auc",
    )
    assert log.best() is winner
    assert winner.metric("roc_auc") == pytest.approx(0.71)
    assert winner.metric("f1") is None


def test_experiment_log_best_is_none_when_unset() -> None:
    """An empty or unresolved leaderboard returns None rather than raising."""
    assert S.ExperimentLog().best() is None
    assert S.ExperimentLog(best_experiment_id="exp_missing").best() is None


def test_dataset_profile_column_lookup() -> None:
    profile = synthesise(S.DatasetProfile)
    profile.columns = [
        S.ColumnProfile(name="a", kind=S.ColumnKind.NUMERIC_CONTINUOUS, dtype="float64"),
    ]
    assert profile.column("a") is not None
    assert profile.column("missing") is None


# ---------------------------------------------------------------------------
# Identity and defaults
# ---------------------------------------------------------------------------


def test_generated_ids_are_prefixed_and_unique() -> None:
    """Ids are readable and collision-free; both matter in a shared database."""
    runs = [S.RunConfig(source=S.DataSource(kind=S.SourceKind.CSV)) for _ in range(50)]
    ids = [r.run_id for r in runs]
    assert len(set(ids)) == 50
    assert all(i.startswith("run_") for i in ids)

    experiments = [S.ExperimentResult(family=S.ModelFamily.RIDGE) for _ in range(20)]
    assert all(e.experiment_id.startswith("exp_") for e in experiments)
    assert len({e.experiment_id for e in experiments}) == 20


def test_run_config_defaults_are_conservative() -> None:
    """Defaults must not surprise: no destructive-step bypass, bounded replans."""
    config = S.RunConfig(source=S.DataSource(kind=S.SourceKind.CSV, uri="x.csv"))
    assert config.random_state == 42
    assert config.require_approval is False
    assert 0.0 < config.test_size < 0.5
    assert config.max_replans >= 0
    assert "json" in config.report_formats


def test_data_source_never_stores_secret_values() -> None:
    """Credentials are referenced by env-var name, never carried on the source."""
    fields = set(S.DataSource.model_fields)
    assert "secret_env" in fields
    for leaky in ("password", "secret", "token", "api_key", "credentials"):
        assert leaky not in fields, f"DataSource exposes a {leaky} field"


def test_timestamps_are_timezone_aware() -> None:
    """Naive datetimes compare wrongly across a UTC boundary; all defaults are aware."""
    event = S.RunEvent(run_id="run_x", kind=S.EventKind.LOG)
    assert event.at.tzinfo is not None
    assert S.RunConfig(source=S.DataSource(kind=S.SourceKind.CSV)).created_at.tzinfo is not None


def test_run_summary_holds_every_stage_slot() -> None:
    """A summary can carry the output of every stage, so a run is fully replayable."""
    expected = {
        "ingestion",
        "profile",
        "understanding",
        "problem",
        "plan",
        "cleaning",
        "features",
        "model_selection",
        "experiments",
        "tuning_decision",
        "tuning",
        "explainability",
        "evaluation",
        "insights",
        "visualization_plan",
        "visualizations",
        "report",
        "report_bundle",
    }
    assert expected <= set(S.RunSummary.model_fields)


def test_leakage_finding_score_is_documented_range() -> None:
    """A leakage score is an association strength in [0, 1] by contract."""
    finding = S.LeakageFinding(
        column="c", score=0.99, method="auc", severity=S.Severity.CRITICAL, reason="r"
    )
    assert 0.0 <= finding.score <= 1.0
