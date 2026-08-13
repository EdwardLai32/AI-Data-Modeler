"""Deterministic decision engine — the keyless path.

When ``settings.offline`` is set, no agent calls Claude. Each one instead asks
this module for its typed output, and the rules here derive that output from the
measured :class:`~automl_architect.core.schemas.DatasetProfile` and whatever the
execution layer has already computed.

What this is, and what it is not:

*   **It is not a mock.** Nothing is stubbed, canned, or randomised. Every
    branch reads a real measurement — a skewness, a cardinality ratio, a class
    imbalance, a train/test gap — and every ``rationale`` it writes quotes the
    number that drove it. A cleaning decision produced here is as executable,
    and as auditable, as one Claude produced.
*   **It is not as good as the model.** Rules cannot read a column *name* and
    infer that ``last_login_days`` is a churn precursor, cannot notice that two
    features encode the same thing under different units, and cannot write the
    business narrative the Insight Agent exists to write. Offline output is
    correct and defensible; online output is insightful. The rationales here say
    so, and every decision is tagged so a reader always knows which produced it.

Only the *reasoning* changes offline. Profiling, splitting, training, tuning,
SHAP, charts, and report rendering were always deterministic Python and run
identically either way — an offline run still trains real models on real data
and scores them honestly.

Deciders are keyed by output-model name rather than by ``AgentName`` because the
Report agent owns two schemas (``FinalReport`` and ``QuestionAnswer``); the model
class is the unambiguous key.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Callable

from pydantic import BaseModel

from .schemas import (
    AgentName,
    BiasVarianceDiagnosis,
    BusinessInsight,
    ChartKind,
    ChartSpec,
    CleaningAction,
    CleaningDecision,
    CleaningPlan,
    ColumnAssessment,
    ColumnKind,
    ColumnProfile,
    ColumnRole,
    DatasetProfile,
    DatasetUnderstanding,
    DeploymentPattern,
    DeploymentRecommendation,
    EvaluationVerdict,
    ExecutionPlan,
    ExperimentLog,
    ExplainabilityReport,
    FeatureDecision,
    FeatureOp,
    FeaturePlan,
    FinalReport,
    InsightReport,
    MissingStrategy,
    ModelCandidate,
    ModelFamily,
    ModelSelection,
    Param,
    PlanStep,
    ProblemDefinition,
    QuestionAnswer,
    SearchSpaceEntry,
    Severity,
    TaskType,
    TuningDecision,
    TuningMethod,
    VisualizationPlan,
)
from .state import RunState

logger = logging.getLogger(__name__)

#: Prefix stamped on every rationale this module writes, so a reader of a report
#: or an audit log can tell rule-derived reasoning from model reasoning at a
#: glance. Deliberately short — it appears on hundreds of lines in a long run.
MARK = "[rule]"

#: Above this absolute skewness the mean sits materially off the bulk of the
#: distribution, so the median imputes more representatively. 1.0 is the
#: conventional "moderately skewed" boundary and is what the rationales cite.
SKEW_THRESHOLD = 1.0

#: Missingness above which a column is dropped rather than imputed: past this
#: point imputation invents more of the column than it preserves.
DROP_MISSING_FRACTION = 0.6

#: One-hot is affordable below this many distinct values; above it the encoding
#: is wider than the signal justifies and frequency/target encoding wins.
ONEHOT_MAX_CARDINALITY = 15

#: |r| above which two features are treated as redundant.
CORRELATION_REDUNDANT = 0.95


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _num(value: Any, digits: int = 3) -> str:
    """Format a measurement for a rationale, or ``n/a`` when it is absent."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    if number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


# ---------------------------------------------------------------------------
# Profile accessors
# ---------------------------------------------------------------------------


_NUMERIC_KINDS = {ColumnKind.NUMERIC_CONTINUOUS, ColumnKind.NUMERIC_DISCRETE}
_CATEGORICAL_KINDS = {
    ColumnKind.CATEGORICAL_NOMINAL,
    ColumnKind.CATEGORICAL_ORDINAL,
    ColumnKind.BOOLEAN,
}


def _profile(state: RunState) -> DatasetProfile | None:
    return state.profile


def _feature_columns(state: RunState) -> list[ColumnProfile]:
    """Columns eligible to become features: not the target, not dead weight."""
    profile = _profile(state)
    if profile is None:
        return []
    target = state.target
    return [
        c
        for c in profile.columns
        if c.name != target and not c.is_constant and not c.looks_like_id
    ]


def _target_correlations(profile: DatasetProfile | None) -> dict[str, float]:
    """Map column name -> |correlation with target|."""
    if profile is None:
        return {}
    out: dict[str, float] = {}
    target = profile.target.name if profile.target else None
    for pair in profile.target_correlations:
        other = pair.right if pair.left == target else pair.left
        out[other] = max(out.get(other, 0.0), abs(pair.coefficient))
    return out


def _leakage_columns(profile: DatasetProfile | None, min_severity: Severity) -> list[str]:
    if profile is None:
        return []
    order = {
        Severity.INFO: 0,
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4,
    }
    floor = order[min_severity]
    return [f.column for f in profile.leakage_findings if order[f.severity] >= floor]


def _note_of(state: RunState) -> str:
    """The standing caveat attached to offline narrative fields."""
    return (
        "Produced by the deterministic rule engine (offline mode, no model "
        "calls). Every figure is measured; the reasoning is rule-derived rather "
        "than inferred, so it covers statistical structure but not domain "
        "meaning."
    )


# ---------------------------------------------------------------------------
# Dataset understanding
# ---------------------------------------------------------------------------

#: Column-name substrings that hint at a business domain. Crude by design — the
#: rationale says the inference is name-based so nobody mistakes it for insight.
_DOMAIN_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("customer churn / retention", ("churn", "retention", "cancel", "unsubscrib")),
    ("credit risk / lending", ("default", "credit", "loan", "delinquen", "fico")),
    ("fraud detection", ("fraud", "chargeback", "suspicious")),
    ("retail / e-commerce demand", ("sales", "order", "sku", "basket", "revenue")),
    ("real estate pricing", ("price", "sqft", "bedroom", "bathroom", "property")),
    ("marketing response", ("campaign", "click", "impression", "conversion", "lead")),
    ("healthcare outcomes", ("diagnos", "patient", "readmit", "symptom", "icd")),
    ("human resources / attrition", ("attrition", "employee", "salary", "tenure")),
    ("insurance claims", ("claim", "premium", "policy", "underwrit")),
    ("telemetry / sensors", ("sensor", "temperature", "voltage", "reading")),
)


def _infer_domain(profile: DatasetProfile | None) -> str:
    if profile is None:
        return "unknown domain"
    names = " ".join(c.name.lower() for c in profile.columns)
    for label, hints in _DOMAIN_HINTS:
        if any(hint in names for hint in hints):
            return label
    return "general tabular domain (no recognisable domain keywords in the column names)"


def _infer_grain(profile: DatasetProfile | None) -> str:
    if profile is None:
        return "one record"
    ids = profile.identifier_columns
    temporal = profile.temporal_columns
    if ids and temporal:
        return f"one `{ids[0]}` observed at one `{temporal[0]}`"
    if ids:
        return f"one `{ids[0]}`"
    if temporal:
        return f"one observation at one `{temporal[0]}`"
    return "one record (no identifier column found to name the entity)"


def _assess_column(
    col: ColumnProfile,
    state: RunState,
    corr: dict[str, float],
    leaky: set[str],
) -> ColumnAssessment:
    target = state.target
    concerns: list[str] = []

    if col.name == target:
        role = ColumnRole.TARGET
    elif col.name in leaky:
        role = ColumnRole.LEAKAGE_SUSPECT
    elif col.looks_like_id:
        role = ColumnRole.IDENTIFIER
    elif col.kind is ColumnKind.DATETIME:
        role = ColumnRole.TEMPORAL_INDEX
    elif col.is_constant:
        role = ColumnRole.IGNORED
    else:
        role = ColumnRole.FEATURE

    if col.missing_fraction > 0.2:
        concerns.append(f"{_pct(col.missing_fraction)} missing")
    if col.is_constant:
        concerns.append("constant — zero variance, no information")
    if col.looks_like_id:
        concerns.append(f"identifier-like ({col.n_unique:,} distinct values)")
    if col.is_near_zero_variance and not col.is_constant:
        concerns.append("near-zero variance")
    if col.skewness is not None and abs(col.skewness) > 2:
        concerns.append(f"strongly skewed (skew {_num(col.skewness)})")
    if col.outliers and col.outliers.fraction > 0.05:
        concerns.append(f"{_pct(col.outliers.fraction)} IQR outliers")
    if col.name in leaky:
        concerns.append("associates with the target strongly enough to suggest leakage")

    association = corr.get(col.name)
    if role in (ColumnRole.IDENTIFIER, ColumnRole.IGNORED):
        potential: str = "none"
    elif role is ColumnRole.LEAKAGE_SUSPECT:
        potential = "none"
    elif association is None:
        potential = "medium" if not col.is_near_zero_variance else "low"
    elif association >= 0.4:
        potential = "high"
    elif association >= 0.15:
        potential = "medium"
    elif association >= 0.05:
        potential = "low"
    else:
        potential = "none"

    bits = [f"{col.kind.value.replace('_', ' ')}, {col.n_unique:,} distinct"]
    if col.missing_fraction:
        bits.append(f"{_pct(col.missing_fraction)} missing")
    if association is not None:
        bits.append(f"|r| with target {_num(association)}")
    if col.mean is not None:
        bits.append(f"mean {_num(col.mean)}, sd {_num(col.std)}")
    notes = "; ".join(bits) + "."
    if role is ColumnRole.LEAKAGE_SUSPECT:
        notes += " Held out of modelling pending review."

    return ColumnAssessment(
        name=col.name,
        role=role,
        predictive_potential=potential,  # type: ignore[arg-type]
        concerns=concerns,
        notes=notes,
    )


def _understanding(state: RunState, agent: Any = None) -> DatasetUnderstanding:
    profile = _profile(state)
    if profile is None:
        return DatasetUnderstanding(
            headline="No profile was computed, so the dataset could not be assessed.",
            narrative=_note_of(state),
            likely_domain="unknown",
            grain="unknown",
            data_readiness="unusable",
            readiness_rationale="Profiling did not run; there is nothing to assess.",
        )

    corr = _target_correlations(profile)
    leaky = set(_leakage_columns(profile, Severity.HIGH))
    assessments = [_assess_column(c, state, corr, leaky) for c in profile.columns]

    n_numeric = sum(1 for c in profile.columns if c.kind in _NUMERIC_KINDS)
    n_categorical = sum(1 for c in profile.columns if c.kind in _CATEGORICAL_KINDS)
    domain = _infer_domain(profile)
    grain = _infer_grain(profile)

    # --- readiness ------------------------------------------------------
    blocking: list[str] = []
    if profile.missing_cell_fraction > 0.4:
        blocking.append(f"{_pct(profile.missing_cell_fraction)} of all cells missing")
    if profile.n_rows < 50:
        blocking.append(f"only {profile.n_rows:,} rows")
    dirty: list[str] = []
    if profile.total_missing_cells:
        dirty.append(f"{profile.total_missing_cells:,} missing cells")
    if profile.n_duplicate_rows:
        dirty.append(f"{profile.n_duplicate_rows:,} duplicate rows")
    if profile.constant_columns:
        dirty.append(f"{len(profile.constant_columns)} constant column(s)")
    if leaky:
        dirty.append(f"{len(leaky)} leakage suspect(s)")

    if blocking:
        readiness = "needs_major_work" if profile.n_rows >= 50 else "unusable"
        readiness_rationale = f"{MARK} Blocking issues: {'; '.join(blocking)}."
    elif dirty:
        readiness = "needs_cleaning"
        readiness_rationale = (
            f"{MARK} Usable once cleaned: {'; '.join(dirty)}. None of these is "
            "structural — each has a standard remedy scheduled in the cleaning plan."
        )
    else:
        readiness = "ready"
        readiness_rationale = (
            f"{MARK} No missing cells, no duplicate rows, no constant columns, and "
            "no leakage findings above the HIGH severity floor."
        )

    # --- narrative ------------------------------------------------------
    paragraphs = [
        f"This table holds {profile.n_rows:,} rows across {profile.n_columns:,} "
        f"columns ({_num(profile.memory_bytes / 1_048_576, 1)} MB in memory), of "
        f"which {n_numeric} are numeric and {n_categorical} categorical. Column "
        f"names suggest {domain}. Each row appears to represent {grain}.",
        f"Integrity: {profile.total_missing_cells:,} of "
        f"{profile.n_rows * max(profile.n_columns, 1):,} cells are missing "
        f"({_pct(profile.missing_cell_fraction)}), and {profile.n_duplicate_rows:,} "
        f"rows ({_pct(profile.duplicate_fraction)}) duplicate another row exactly.",
    ]

    worst = sorted(
        (c for c in profile.columns if c.missing_fraction > 0),
        key=lambda c: c.missing_fraction,
        reverse=True,
    )[:5]
    if worst:
        paragraphs.append(
            "Missingness concentrates in "
            + ", ".join(f"`{c.name}` ({_pct(c.missing_fraction)})" for c in worst)
            + ". Each is imputed by a rule matched to its measured distribution "
            "rather than by a single global strategy."
        )

    if profile.target:
        tgt = profile.target
        if tgt.n_classes:
            balance = (
                f"imbalanced at {_num(tgt.imbalance_ratio)}:1 majority-to-minority"
                if tgt.is_imbalanced
                else "reasonably balanced"
            )
            paragraphs.append(
                f"The target `{tgt.name}` has {tgt.n_classes} classes and is "
                f"{balance}. That drives both the metric choice and the decision "
                "to stratify the splits."
            )
        else:
            paragraphs.append(
                f"The target `{tgt.name}` is continuous (mean {_num(tgt.mean)}, "
                f"sd {_num(tgt.std)}, skew {_num(tgt.skewness)}), so this is a "
                "regression problem scored on error magnitude."
            )

    strong = sorted(corr.items(), key=lambda kv: kv[1], reverse=True)[:5]
    if strong:
        paragraphs.append(
            "Strongest measured associations with the target: "
            + ", ".join(f"`{n}` (|r| {_num(v)})" for n, v in strong)
            + ". These are linear correlations only; a tree ensemble may find "
            "structure this ranking cannot see."
        )

    if profile.highly_correlated_pairs:
        pair = profile.highly_correlated_pairs[0]
        paragraphs.append(
            f"{len(profile.highly_correlated_pairs)} feature pair(s) are highly "
            f"correlated with each other — for example `{pair.left}` and "
            f"`{pair.right}` at r={_num(pair.coefficient)}. Redundancy inflates "
            "variance in linear models and splits importance across duplicates in "
            "tree models."
        )

    if leaky:
        paragraphs.append(
            f"Leakage risk: {', '.join(f'`{c}`' for c in sorted(leaky))} associate "
            "with the target far more tightly than a genuine predictor normally "
            "would. They are dropped before modelling; a score that depends on "
            "them would not survive deployment."
        )

    paragraphs.append(_note_of(state))

    key_findings = [
        f"{profile.n_rows:,} rows x {profile.n_columns:,} columns; "
        f"{_pct(profile.missing_cell_fraction)} of cells missing.",
    ]
    if profile.n_duplicate_rows:
        key_findings.append(
            f"{profile.n_duplicate_rows:,} exact duplicate rows "
            f"({_pct(profile.duplicate_fraction)}) — deduplication scheduled."
        )
    if profile.constant_columns:
        key_findings.append(
            f"Constant columns carry no signal: "
            f"{', '.join(f'`{c}`' for c in profile.constant_columns[:6])}."
        )
    if profile.target and profile.target.is_imbalanced:
        key_findings.append(
            f"Target imbalance {_num(profile.target.imbalance_ratio)}:1 — accuracy "
            "would be misleading here."
        )
    for name, value in strong[:3]:
        key_findings.append(f"`{name}` correlates with the target at |r|={_num(value)}.")

    risks = [
        f"`{c}` is a leakage suspect and is excluded from modelling." for c in sorted(leaky)
    ]
    if profile.n_rows < 1000:
        risks.append(
            f"{profile.n_rows:,} rows is small; holdout estimates will carry wide "
            "confidence intervals and model choice matters less than variance control."
        )
    if profile.highly_correlated_pairs:
        risks.append(
            f"{len(profile.highly_correlated_pairs)} highly correlated feature pair(s) "
            "will distort per-feature importance attribution."
        )
    high_card = [
        c for c in profile.columns if c.kind in _CATEGORICAL_KINDS and c.n_unique > 50
    ]
    if high_card:
        risks.append(
            "High-cardinality categoricals ("
            + ", ".join(f"`{c.name}` ({c.n_unique:,})" for c in high_card[:4])
            + ") risk overfitting under naive encoding."
        )
    risks.append(
        "Domain framing is inferred from column names only — offline mode cannot "
        "reason about what these fields mean in the business."
    )

    # Target candidates: prefer the configured target, then low-cardinality or
    # well-named columns that are not identifiers.
    candidates: list[str] = []
    if state.target:
        candidates.append(state.target)
    for col in profile.columns:
        if col.name in candidates or col.looks_like_id or col.is_constant:
            continue
        lowered = col.name.lower()
        if any(
            hint in lowered
            for hint in ("target", "label", "churn", "outcome", "class", "y", "default")
        ):
            candidates.append(col.name)
    for col in profile.columns:
        if col.name in candidates or col.looks_like_id or col.is_constant:
            continue
        if col.kind in _CATEGORICAL_KINDS and 2 <= col.n_unique <= 10:
            candidates.append(col.name)

    headline = (
        f"{profile.n_rows:,} x {profile.n_columns:,} table in {domain}, "
        f"{readiness.replace('_', ' ')}"
        + (f", targeting `{state.target}`." if state.target else ".")
    )

    return DatasetUnderstanding(
        headline=headline,
        narrative="\n\n".join(paragraphs),
        likely_domain=domain,
        grain=grain,
        column_assessments=assessments,
        key_findings=key_findings,
        risks=risks,
        suggested_target_columns=candidates[:5],
        data_readiness=readiness,  # type: ignore[arg-type]
        readiness_rationale=readiness_rationale,
    )


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------


def _primary_metric(task: TaskType, imbalanced: bool) -> tuple[str, list[str], str]:
    """Metric, secondaries, and the reason — all canonical names."""
    if task is TaskType.BINARY_CLASSIFICATION:
        if imbalanced:
            return (
                "roc_auc",
                ["average_precision", "f1", "balanced_accuracy", "recall", "precision"],
                "The classes are imbalanced, so accuracy would reward a "
                "majority-class constant predictor. ROC AUC is threshold-free and "
                "insensitive to the base rate; average precision is reported "
                "alongside it because it is the more honest summary when the "
                "positive class is rare.",
            )
        return (
            "roc_auc",
            ["accuracy", "f1", "precision", "recall"],
            "Binary target with usable class balance. ROC AUC measures ranking "
            "quality independently of where the decision threshold lands, which "
            "keeps model comparison separate from threshold tuning.",
        )
    if task is TaskType.MULTICLASS_CLASSIFICATION:
        return (
            "f1_macro",
            ["accuracy", "balanced_accuracy", "f1_weighted"],
            "Multiclass target. Macro-F1 weights every class equally, so a small "
            "class cannot be ignored for free the way it can under plain accuracy.",
        )
    if task in (TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING):
        return (
            "rmse",
            ["mae", "r2", "mape"],
            "Continuous target. RMSE is in the target's own units and penalises "
            "large misses quadratically; MAE is carried alongside because it is "
            "robust to the outliers the profile measured.",
        )
    if task is TaskType.CLUSTERING:
        return (
            "silhouette",
            ["calinski_harabasz", "davies_bouldin"],
            "No target column, so quality is measured by cluster cohesion and "
            "separation rather than against labels.",
        )
    if task is TaskType.ANOMALY_DETECTION:
        return (
            "score_separation",
            ["outlier_fraction"],
            "Unlabelled anomaly detection; separation between normal and anomalous "
            "score distributions is the only available quality signal.",
        )
    return ("accuracy", [], "Default metric for this task type.")


def _problem(state: RunState, agent: Any = None) -> ProblemDefinition:
    profile = _profile(state)
    target_name = state.target
    target = profile.target if profile else None

    # Reuse the executor's own inference so offline and online agree on the rule.
    try:
        from ..agents.problem import _task_from_target

        task = _task_from_target(profile)
    except Exception:  # noqa: BLE001 - fall back to the local rule
        if target is None:
            task = TaskType.CLUSTERING
        elif target.kind in (ColumnKind.NUMERIC_CONTINUOUS, ColumnKind.DATETIME):
            task = TaskType.REGRESSION
        elif target.n_classes is None:
            task = TaskType.REGRESSION
        elif target.n_classes <= 2:
            task = TaskType.BINARY_CLASSIFICATION
        elif target.n_classes <= 50:
            task = TaskType.MULTICLASS_CLASSIFICATION
        else:
            task = TaskType.REGRESSION

    if state.config.task_type_override:
        task = state.config.task_type_override

    imbalanced = bool(target and target.is_imbalanced)
    metric, secondary, metric_rationale = _primary_metric(task, imbalanced)
    if state.config.primary_metric_override:
        metric = state.config.primary_metric_override
        metric_rationale = (
            f"{MARK} Metric fixed to `{metric}` by run configuration, overriding "
            "the rule-derived choice."
        )

    # --- rationale, citing what actually decided it ----------------------
    if target is None:
        rationale = (
            f"{MARK} No target column was supplied or detected, so no supervised "
            "objective exists. The task falls to clustering, which is the only "
            "framing the measured data supports."
        )
        confidence = "medium"
    elif task is TaskType.REGRESSION:
        rationale = (
            f"{MARK} `{target.name}` is {target.kind.value} with "
            f"{target.n_classes or 'many'} distinct values (mean {_num(target.mean)}, "
            f"sd {_num(target.std)}). A continuous, unbounded target is a "
            "regression problem: the quantity to predict is a magnitude, not a "
            "membership."
        )
        confidence = "high"
    elif task is TaskType.BINARY_CLASSIFICATION:
        classes = ", ".join(f"`{c.value}` ({c.count:,})" for c in target.class_counts[:2])
        rationale = (
            f"{MARK} `{target.name}` takes exactly {target.n_classes} distinct "
            f"values ({classes}), which is a binary classification problem. "
            + (
                f"Class balance is {_num(target.imbalance_ratio)}:1, past the 3:1 "
                "point where accuracy stops being informative, so the metric is "
                "chosen accordingly."
                if imbalanced
                else "Class balance is workable, so no resampling is forced."
            )
        )
        confidence = "high"
    elif task is TaskType.MULTICLASS_CLASSIFICATION:
        rationale = (
            f"{MARK} `{target.name}` is categorical with {target.n_classes} "
            "distinct classes — more than two, and few enough to model directly "
            "as multiclass rather than collapsing into groups."
        )
        confidence = "high"
    else:
        rationale = f"{MARK} Task inferred as {task.value} from the measured target."
        confidence = "medium"

    positive_class = None
    if task is TaskType.BINARY_CLASSIFICATION and target and target.class_counts:
        # The minority class is the event of interest in nearly every business
        # framing (churn, fraud, default, conversion).
        positive_class = min(target.class_counts, key=lambda c: c.count).value

    temporal = profile.temporal_columns[0] if profile and profile.temporal_columns else None

    constraints = [
        "Offline mode: reasoning is rule-derived, so the business framing below "
        "is a structural default rather than a stakeholder-informed objective.",
    ]
    if imbalanced:
        constraints.append(
            f"Class imbalance {_num(target.imbalance_ratio if target else None)}:1 — "
            "splits must be stratified and accuracy must not be the headline number."
        )
    if profile and profile.n_rows < 1000:
        constraints.append(
            f"{profile.n_rows:,} rows — prefer cross-validation over a single "
            "holdout, and prefer regularised models over high-capacity ones."
        )

    objective = (
        f"Predict `{target_name}` accurately enough to act on, and understand which "
        "measured factors drive it."
        if target_name
        else "Segment the records into coherent groups for downstream targeting."
    )

    alternatives: list[str] = []
    if target is not None and target.n_classes and 2 < target.n_classes <= 5:
        alternatives.append(
            f"Binary classification by collapsing the {target.n_classes} classes into "
            "an event/non-event split — rejected because it discards a distinction "
            "the data actually records."
        )
    if temporal and task is not TaskType.TIME_SERIES_FORECASTING:
        alternatives.append(
            f"Time-series forecasting on `{temporal}` — not selected because the "
            "rows carry per-entity features rather than a single ordered series, "
            "but the temporal column still forces a time-ordered split."
        )
    if target is not None:
        alternatives.append(
            "Anomaly detection — rejected because a labelled target is present, and "
            "supervised signal beats unsupervised proxies when it exists."
        )

    return ProblemDefinition(
        task_type=task,
        target_column=target_name,
        positive_class=positive_class,
        temporal_column=temporal,
        group_column=None,
        horizon=None,
        rationale=rationale,
        alternatives_considered=alternatives,
        confidence=confidence,  # type: ignore[arg-type]
        primary_metric=metric,
        secondary_metrics=secondary,
        metric_rationale=f"{MARK} {metric_rationale}",
        business_objective=objective,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _plan(state: RunState, agent: Any = None) -> ExecutionPlan:
    """Schedule the canonical pipeline, dropping what this dataset does not need."""
    from ..orchestrator.graph import CANONICAL_STEPS

    profile = _profile(state)
    task = state.task_type or TaskType.BINARY_CLASSIFICATION
    config = state.config

    needs_cleaning = bool(
        profile
        and (
            profile.total_missing_cells
            or profile.n_duplicate_rows
            or profile.constant_columns
            or profile.leakage_findings
        )
    )
    drops = list(profile.constant_columns) if profile else []
    drops += _leakage_columns(profile, Severity.HIGH)

    reasons: dict[str, str] = {
        "clean": (
            f"{MARK} The profile measured {profile.total_missing_cells:,} missing "
            f"cells, {profile.n_duplicate_rows:,} duplicate rows, and "
            f"{len(profile.constant_columns)} constant column(s). Each has a "
            "distinct remedy, so cleaning precedes any feature work."
            if profile
            else f"{MARK} Standard cleaning pass."
        ),
        "engineer_features": (
            f"{MARK} Encoding is mandatory — the models cannot consume raw "
            "categorical or datetime columns — and scaling is required by the "
            "distance- and gradient-based candidates."
        ),
        "split": (
            f"{MARK} Held-out data must be carved out before any fitting so that "
            "imputation and encoding statistics never see the test rows."
        ),
        "select_models": (
            f"{MARK} Candidate families are chosen from measured size and shape "
            f"({profile.n_rows:,} rows x {profile.n_columns:,} columns) rather "
            "than by default."
            if profile
            else f"{MARK} Choose candidates for this task."
        ),
        "run_experiments": (
            f"{MARK} Every candidate is trained and cross-validated on identical "
            "splits, so the comparison between them is fair."
        ),
        "tune": (
            f"{MARK} Tuning runs only if the leaderboard shows headroom worth the "
            "compute; the decision is made against measured scores, not assumed."
        ),
        "explain": (
            f"{MARK} Attribution is required output, not a nicety: a score without "
            "an explanation cannot be acted on or audited."
        ),
        "evaluate": (
            f"{MARK} The quality gate. Compares train against holdout to diagnose "
            "over- or under-fitting and decides whether the result is shippable."
        ),
        "insights": (
            f"{MARK} Translates attributions and metrics into business language."
        ),
        "visualise": (
            f"{MARK} Charts selected for {task.value}, covering both data structure "
            "and model behaviour."
        ),
        "report": (
            f"{MARK} Assembles the nine required sections into the deliverable."
        ),
    }

    # Only agent-owned steps belong in a plan. Pure-executor steps such as
    # `split` have no agent to assign, and every post-plan agent already owns a
    # step of its own — naming one twice would make the planner warn that the
    # second invocation overwrites the first. The orchestrator re-injects
    # non-skippable executor steps itself, so omitting them loses nothing.
    selected = [
        d
        for d in CANONICAL_STEPS
        if d.planned
        and d.agent is not None
        and not (d.step_id == "clean" and not needs_cleaning)
        and not (d.step_id == "tune" and not config.enable_tuning)
        and not (d.step_id == "explain" and not config.enable_explainability)
    ]

    # The planner forces the terminal agents to the tail, so a dependency
    # declared against one of them from an earlier step would become a forward
    # edge and be dropped with a warning. Mirror that ordering here, then wire
    # each step to its immediate predecessor — the orchestrator walks the plan
    # linearly, so a chain is both sufficient and always acyclic. The constant is
    # imported rather than restated so the two cannot drift apart.
    try:
        from ..agents.planner import TERMINAL_AGENTS as terminal
    except Exception:  # noqa: BLE001 - keep the engine usable if planner moves
        terminal = (AgentName.EVALUATION, AgentName.INSIGHT, AgentName.REPORT)
    body = [d for d in selected if d.agent not in terminal]
    tail = [d for d in selected if d.agent in terminal]
    tail.sort(key=lambda d: terminal.index(d.agent))  # type: ignore[arg-type]
    ordered_defs = body + tail

    steps: list[PlanStep] = []
    for index, definition in enumerate(ordered_defs):
        step_id = definition.step_id
        destructive = definition.destructive or (step_id == "clean" and bool(drops))
        steps.append(
            PlanStep(
                step_id=step_id,
                order=index + 1,
                title=definition.title,
                agent=definition.agent,  # type: ignore[arg-type]
                objective=definition.title,
                rationale=reasons.get(step_id, f"{MARK} Canonical pipeline step."),
                depends_on=[ordered_defs[index - 1].step_id] if index else [],
                optional=definition.skippable,
                destructive=destructive,
                estimated_seconds=definition.estimated_seconds,
                success_criteria=f"{step_id} completes without error",
            )
        )

    adaptations: list[str] = []
    if not needs_cleaning:
        adaptations.append(
            "Cleaning is skipped entirely: no missing cells, no duplicate rows, no "
            "constant columns, and no leakage findings were measured."
        )
    if profile and profile.temporal_columns:
        adaptations.append(
            f"`{profile.temporal_columns[0]}` is a datetime column, so the split is "
            "time-ordered rather than random — a shuffled split would train on the "
            "future and score on the past."
        )
    if profile and profile.target and profile.target.is_imbalanced:
        adaptations.append(
            f"Target imbalance of {_num(profile.target.imbalance_ratio)}:1 forces "
            "stratified folds and a ranking metric instead of accuracy."
        )
    if profile and profile.n_rows < 2000:
        adaptations.append(
            f"At {profile.n_rows:,} rows the plan favours cross-validation and "
            "regularised models; high-capacity boosting would memorise this set."
        )
    if drops:
        adaptations.append(
            f"{len(drops)} column(s) are dropped before modelling: "
            f"{', '.join(f'`{c}`' for c in drops[:6])}."
        )
    if not config.enable_tuning:
        adaptations.append("Hyperparameter tuning is disabled by run configuration.")

    risks = [
        "Offline mode: the plan follows the canonical pipeline with rule-based "
        "adaptations. It cannot invent a dataset-specific step the way the "
        "Planning Agent can.",
    ]
    if profile and profile.n_rows < 500:
        risks.append(
            f"{profile.n_rows:,} rows makes every holdout estimate noisy; treat "
            "small score differences between candidates as ties."
        )
    if drops:
        risks.append(
            "Dropping leakage suspects will lower the headline score. That is the "
            "correct outcome — the inflated score was not real."
        )

    return ExecutionPlan(
        summary=(
            f"{MARK} A {len(steps)}-step pipeline for {task.value} on "
            f"{profile.n_rows:,} rows x {profile.n_columns:,} columns. "
            if profile
            else f"{MARK} A {len(steps)}-step pipeline for {task.value}. "
        )
        + "Clean, engineer, split before fitting, train a baseline alongside real "
        "candidates, tune only if measurement justifies it, then explain, gate on "
        "quality, and report. Each step is included because the profile showed it "
        "was needed, and the ones that were not needed were dropped.",
        steps=steps,
        dataset_specific_adaptations=adaptations,
        risks=risks,
        fallback_strategy=(
            "If evaluation rejects the result, re-enter at feature engineering with "
            "interaction and ratio terms; if a second pass still fails, widen the "
            "model families rather than tuning the same family harder. Persistent "
            "failure after both means the measured signal is too weak and the "
            "honest recommendation is to collect better features."
        ),
        revision=0,
    )


# ---------------------------------------------------------------------------
# Cleaning — the flagship rule set
# ---------------------------------------------------------------------------


def _impute_decision(col: ColumnProfile, n_rows: int) -> CleaningDecision | None:
    """Pick an imputation strategy from the column's measured distribution."""
    if col.n_missing == 0:
        return None

    if col.missing_fraction >= DROP_MISSING_FRACTION:
        return CleaningDecision(
            action=CleaningAction.DROP_COLUMN,
            columns=[col.name],
            rationale=(
                f"{MARK} {_pct(col.missing_fraction)} of `{col.name}` is missing "
                f"({col.n_missing:,} of {n_rows:,} rows). Imputing would fabricate "
                "more of the column than the data provides, so any signal found in "
                "it afterwards would be an artefact of the imputer."
            ),
            expected_impact="Removes one column; prevents a fabricated feature.",
            destructive=True,
            severity_if_skipped=Severity.HIGH,
        )

    if col.kind in _NUMERIC_KINDS:
        skew = col.skewness
        if skew is not None and abs(skew) > SKEW_THRESHOLD:
            return CleaningDecision(
                action=CleaningAction.IMPUTE_MISSING,
                columns=[col.name],
                strategy=MissingStrategy.MEDIAN,
                rationale=(
                    f"{MARK} Highly skewed distribution: skewness {_num(skew)} "
                    f"(threshold {SKEW_THRESHOLD}), with mean {_num(col.mean)} "
                    f"against median {_num(col.quantiles.p50 if col.quantiles else None)}. "
                    "The mean is pulled toward the tail and would bias every "
                    "imputed row upward, so the median is the representative centre."
                ),
                expected_impact=(
                    f"Fills {col.n_missing:,} rows without shifting the distribution's centre."
                ),
                severity_if_skipped=Severity.MEDIUM,
            )
        return CleaningDecision(
            action=CleaningAction.IMPUTE_MISSING,
            columns=[col.name],
            strategy=MissingStrategy.MEAN,
            rationale=(
                f"{MARK} Distribution is approximately normal: skewness "
                f"{_num(skew)} is within +/-{SKEW_THRESHOLD}, and the mean "
                f"({_num(col.mean)}) sits close to the median "
                f"({_num(col.quantiles.p50 if col.quantiles else None)}). With no "
                "meaningful tail to distort it, the mean is the minimum-variance "
                "estimate of the centre."
            ),
            expected_impact=f"Fills {col.n_missing:,} rows; preserves the mean exactly.",
            severity_if_skipped=Severity.MEDIUM,
        )

    if col.kind is ColumnKind.DATETIME:
        return CleaningDecision(
            action=CleaningAction.IMPUTE_MISSING,
            columns=[col.name],
            strategy=MissingStrategy.FORWARD_FILL,
            rationale=(
                f"{MARK} `{col.name}` is a datetime column with {col.n_missing:,} "
                "gaps. Carrying the last observed timestamp forward respects the "
                "ordering of the series, which mean or mode imputation would destroy."
            ),
            severity_if_skipped=Severity.MEDIUM,
        )

    if col.kind in _CATEGORICAL_KINDS:
        if col.n_unique > ONEHOT_MAX_CARDINALITY or col.missing_fraction > 0.15:
            return CleaningDecision(
                action=CleaningAction.IMPUTE_MISSING,
                columns=[col.name],
                strategy=MissingStrategy.MISSING_CATEGORY,
                rationale=(
                    f"{MARK} Categorical with {col.n_unique:,} levels and "
                    f"{_pct(col.missing_fraction)} missing. At this rate the "
                    "absence is likely informative rather than random, so it is "
                    "encoded as its own level instead of being hidden inside the "
                    "most frequent one."
                ),
                severity_if_skipped=Severity.MEDIUM,
            )
        top = col.top_values[0] if col.top_values else None
        return CleaningDecision(
            action=CleaningAction.IMPUTE_MISSING,
            columns=[col.name],
            strategy=MissingStrategy.MODE,
            rationale=(
                f"{MARK} Categorical feature with {col.n_unique:,} levels; the "
                "mean and median are undefined for unordered categories. The mode"
                + (
                    f" (`{top.value}`, {_pct(top.fraction)} of rows)"
                    if top
                    else ""
                )
                + f" fills {col.n_missing:,} rows "
                f"({_pct(col.missing_fraction)}) — a low enough share that the "
                "distribution barely shifts."
            ),
            severity_if_skipped=Severity.MEDIUM,
        )

    if col.kind is ColumnKind.TEXT:
        return CleaningDecision(
            action=CleaningAction.IMPUTE_MISSING,
            columns=[col.name],
            strategy=MissingStrategy.CONSTANT,
            parameters=[Param(key="fill_value", value="")],
            rationale=(
                f"{MARK} Free-text column with {col.n_missing:,} nulls. An empty "
                "string is the natural absence value for text and keeps downstream "
                "vectorisers from failing on None."
            ),
            severity_if_skipped=Severity.LOW,
        )

    return None


def _cleaning(state: RunState, agent: Any = None) -> CleaningPlan:
    profile = _profile(state)
    if profile is None:
        return CleaningPlan(
            decisions=[],
            summary=f"{MARK} No profile available, so no cleaning could be reasoned about.",
        )

    target = state.target
    decisions: list[CleaningDecision] = []
    drops: list[str] = []
    drop_reasons: list[str] = []

    # --- whole-table actions --------------------------------------------
    if profile.n_duplicate_rows:
        decisions.append(
            CleaningDecision(
                action=CleaningAction.DROP_DUPLICATE_ROWS,
                rationale=(
                    f"{MARK} {profile.n_duplicate_rows:,} rows "
                    f"({_pct(profile.duplicate_fraction)}) duplicate another row "
                    "exactly. Duplicates leak across a random split — the same row "
                    "lands in train and test — which inflates the holdout score "
                    "without improving the model."
                ),
                expected_impact=f"Removes {profile.n_duplicate_rows:,} rows.",
                destructive=True,
                severity_if_skipped=Severity.HIGH,
            )
        )

    if target and profile.target and profile.target.n_missing:
        decisions.append(
            CleaningDecision(
                action=CleaningAction.DROP_ROWS_MISSING_TARGET,
                columns=[target],
                rationale=(
                    f"{MARK} {profile.target.n_missing:,} rows have no value for "
                    f"the target `{target}`. A supervised model cannot learn from "
                    "a row with no label, and imputing a target would be inventing "
                    "the answer."
                ),
                destructive=True,
                severity_if_skipped=Severity.CRITICAL,
            )
        )

    # --- constants -------------------------------------------------------
    for name in profile.constant_columns:
        if name == target:
            continue
        drops.append(name)
        drop_reasons.append(f"`{name}`: single distinct value, zero variance.")
        decisions.append(
            CleaningDecision(
                action=CleaningAction.DROP_CONSTANT_COLUMN,
                columns=[name],
                rationale=(
                    f"{MARK} `{name}` holds one distinct value across all "
                    f"{profile.n_rows:,} rows. A constant cannot discriminate "
                    "between outcomes; it only widens the feature matrix."
                ),
                destructive=True,
                severity_if_skipped=Severity.LOW,
            )
        )

    # --- leakage ---------------------------------------------------------
    for finding in profile.leakage_findings:
        if finding.column == target or finding.severity not in (
            Severity.HIGH,
            Severity.CRITICAL,
        ):
            continue
        drops.append(finding.column)
        drop_reasons.append(f"`{finding.column}`: leakage — {finding.reason}")
        decisions.append(
            CleaningDecision(
                action=CleaningAction.DROP_LEAKAGE_COLUMN,
                columns=[finding.column],
                rationale=(
                    f"{MARK} `{finding.column}` associates with the target at "
                    f"{_num(finding.score)} by {finding.method} "
                    f"({finding.severity.value} severity): {finding.reason}. A "
                    "genuine predictor rarely reaches this strength; keeping it "
                    "would produce a score that collapses in production, when the "
                    "column is not yet known."
                ),
                destructive=True,
                severity_if_skipped=Severity.CRITICAL,
            )
        )

    # --- identifiers -----------------------------------------------------
    for col in profile.columns:
        if col.name == target or not col.looks_like_id or col.name in drops:
            continue
        drops.append(col.name)
        drop_reasons.append(
            f"`{col.name}`: identifier ({col.n_unique:,} of {profile.n_rows:,} rows distinct)."
        )
        decisions.append(
            CleaningDecision(
                action=CleaningAction.DROP_COLUMN,
                columns=[col.name],
                rationale=(
                    f"{MARK} `{col.name}` has {col.n_unique:,} distinct values "
                    f"across {profile.n_rows:,} rows "
                    f"(cardinality ratio {_num(col.cardinality_ratio)}), which "
                    "makes it a row identifier. A tree can memorise the target "
                    "through it and learn nothing that generalises."
                ),
                destructive=True,
                severity_if_skipped=Severity.HIGH,
            )
        )

    # --- per-column imputation -------------------------------------------
    for col in profile.columns:
        if col.name in drops:
            continue
        if col.name == target:
            continue  # handled by DROP_ROWS_MISSING_TARGET
        decision = _impute_decision(col, profile.n_rows)
        if decision is None:
            continue
        if decision.action is CleaningAction.DROP_COLUMN:
            drops.append(col.name)
            drop_reasons.append(
                f"`{col.name}`: {_pct(col.missing_fraction)} missing, past the "
                f"{_pct(DROP_MISSING_FRACTION)} imputation ceiling."
            )
        decisions.append(decision)

    # --- outliers --------------------------------------------------------
    heavy = [
        c
        for c in profile.columns
        if c.name not in drops
        and c.name != target
        and c.outliers
        and c.outliers.fraction > 0.02
        and c.kind in _NUMERIC_KINDS
    ]
    if heavy:
        decisions.append(
            CleaningDecision(
                action=CleaningAction.CLIP_OUTLIERS,
                columns=[c.name for c in heavy],
                parameters=[
                    Param(key="method", value="iqr"),
                    Param(key="factor", value="1.5"),
                ],
                rationale=(
                    f"{MARK} "
                    + "; ".join(
                        f"`{c.name}` has {c.outliers.n_outliers:,} IQR outliers "
                        f"({_pct(c.outliers.fraction)})"
                        for c in heavy[:4]
                    )
                    + ". Clipping to the IQR fence caps their leverage on "
                    "scale-sensitive models while keeping the rows — deleting them "
                    "would discard real observations that carry signal elsewhere."
                ),
                expected_impact="Bounds extreme values; row count unchanged.",
                severity_if_skipped=Severity.LOW,
            )
        )

    # --- deliberately not done -------------------------------------------
    skipped: list[str] = []
    if not profile.n_duplicate_rows:
        skipped.append("Deduplication — no exact duplicate rows were measured.")
    if not heavy:
        skipped.append(
            "Outlier treatment — no numeric column exceeded 2% IQR outliers, so "
            "clipping would distort more than it protects."
        )
    skipped.append(
        "Row deletion for missingness — imputation preserves sample size, and at "
        "the measured rates the imputed values do not dominate any column."
    )
    skipped.append(
        "Rescaling and encoding — deferred to feature engineering so they are fit "
        "on the training split only and cannot leak test statistics."
    )
    if profile.highly_correlated_pairs:
        skipped.append(
            f"Dropping {len(profile.highly_correlated_pairs)} correlated pair(s) "
            "here — handled during feature engineering, where the choice can be "
            "made against the assembled matrix."
        )

    imputed = sum(1 for d in decisions if d.action is CleaningAction.IMPUTE_MISSING)
    summary = (
        f"{MARK} {len(decisions)} decision(s): {imputed} column(s) imputed by a "
        f"strategy matched to their measured distribution, {len(drops)} column(s) "
        f"dropped"
        + (f" ({', '.join(f'`{c}`' for c in drops[:5])})" if drops else "")
        + ". Numeric columns are split by skewness — median past "
        f"|skew| > {SKEW_THRESHOLD}, mean below it — and categoricals by "
        "cardinality, so no single global strategy is applied blindly."
    )

    return CleaningPlan(
        decisions=decisions,
        summary=summary,
        columns_to_drop=drops,
        drop_rationale=drop_reasons,
        skipped_considerations=skipped,
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _features(state: RunState, agent: Any = None) -> FeaturePlan:
    profile = _profile(state)
    if profile is None:
        return FeaturePlan(decisions=[], summary=f"{MARK} No profile available.")

    task = state.task_type or TaskType.BINARY_CLASSIFICATION
    columns = _feature_columns(state)
    dropped = set(state.dropped_columns) | set(
        state.cleaning.columns_to_drop if state.cleaning else []
    )
    columns = [c for c in columns if c.name not in dropped]

    numeric = [c for c in columns if c.kind in _NUMERIC_KINDS]
    categorical = [c for c in columns if c.kind in _CATEGORICAL_KINDS]
    datetimes = [c for c in columns if c.kind is ColumnKind.DATETIME]
    texts = [c for c in columns if c.kind is ColumnKind.TEXT or c.looks_like_text]

    decisions: list[FeatureDecision] = []
    delta = 0

    # --- datetime decomposition ------------------------------------------
    for col in datetimes:
        decisions.append(
            FeatureDecision(
                op=FeatureOp.DATE_DECOMPOSE,
                input_columns=[col.name],
                output_name_hint=f"{col.name}_parts",
                parameters=[Param(key="parts", value='["year","month","day","dayofweek"]')],
                rationale=(
                    f"{MARK} `{col.name}` spans {col.min_timestamp} to "
                    f"{col.max_timestamp}. As a raw timestamp it is a single "
                    "monotonic number that a tree can only split on chronologically; "
                    "decomposed, the recurring structure (month, day of week) "
                    "becomes learnable."
                ),
                hypothesis=(
                    "Behaviour varies by position in the week and year — weekends "
                    "and month boundaries do not look like ordinary days."
                ),
                risk=(
                    "Adds four columns per date. Harmless for trees; for linear "
                    "models the cyclical encoding below is the safer companion."
                ),
                priority="high",
            )
        )
        delta += 4
        decisions.append(
            FeatureDecision(
                op=FeatureOp.CYCLICAL_ENCODE,
                input_columns=[col.name],
                output_name_hint=f"{col.name}_cyclical",
                parameters=[Param(key="periods", value='["month","dayofweek"]')],
                rationale=(
                    f"{MARK} Calendar parts of `{col.name}` wrap around: December "
                    "(12) is adjacent to January (1), but numerically they are 11 "
                    "apart. Sine/cosine pairs restore that adjacency."
                ),
                hypothesis="Seasonal effects are continuous across the year boundary.",
                risk="Two extra columns per period; no leakage risk.",
                priority="medium",
            )
        )
        delta += 4

    # --- skew correction --------------------------------------------------
    skewed = [
        c
        for c in numeric
        if c.skewness is not None
        and abs(c.skewness) > 1.5
        and (c.minimum is not None and c.minimum >= 0)
    ]
    for col in skewed[:8]:
        decisions.append(
            FeatureDecision(
                op=FeatureOp.LOG_TRANSFORM,
                input_columns=[col.name],
                output_name_hint=f"log_{col.name}",
                parameters=[Param(key="offset", value="1")],
                rationale=(
                    f"{MARK} `{col.name}` is right-skewed at {_num(col.skewness)} "
                    f"(min {_num(col.minimum)}, median "
                    f"{_num(col.quantiles.p50 if col.quantiles else None)}, max "
                    f"{_num(col.maximum)}). log1p compresses the tail so the top "
                    "few percent of rows stop dominating any distance or gradient "
                    "computation."
                ),
                hypothesis=(
                    "The effect on the target is multiplicative rather than "
                    "additive — a doubling matters, not a fixed increment."
                ),
                risk=(
                    "Only valid because the measured minimum is non-negative; the "
                    "offset of 1 keeps zeros finite."
                ),
                priority="high" if abs(col.skewness or 0) > 3 else "medium",
            )
        )
        delta += 1

    # --- categorical encoding ---------------------------------------------
    low_card = [c for c in categorical if c.n_unique <= ONEHOT_MAX_CARDINALITY]
    high_card = [c for c in categorical if c.n_unique > ONEHOT_MAX_CARDINALITY]

    if low_card:
        decisions.append(
            FeatureDecision(
                op=FeatureOp.ONE_HOT_ENCODE,
                input_columns=[c.name for c in low_card],
                parameters=[
                    Param(key="handle_unknown", value="ignore"),
                    Param(key="drop", value="if_binary"),
                ],
                rationale=(
                    f"{MARK} {len(low_card)} categorical column(s) with at most "
                    f"{ONEHOT_MAX_CARDINALITY} levels ("
                    + ", ".join(f"`{c.name}`:{c.n_unique}" for c in low_card[:5])
                    + "). One-hot imposes no false ordering between levels, and at "
                    "this cardinality the added width is affordable."
                ),
                hypothesis="Each level shifts the outcome independently.",
                risk=(
                    f"Adds about {sum(c.n_unique for c in low_card)} columns. "
                    "`handle_unknown=ignore` keeps unseen levels from failing at "
                    "inference."
                ),
                priority="high",
            )
        )
        delta += sum(c.n_unique for c in low_card)

    for col in high_card:
        if task.is_classification or task is TaskType.REGRESSION:
            decisions.append(
                FeatureDecision(
                    op=FeatureOp.TARGET_ENCODE,
                    input_columns=[col.name],
                    output_name_hint=f"{col.name}_target_enc",
                    parameters=[
                        Param(key="cv", value="5"),
                        Param(key="smoothing", value="10"),
                    ],
                    rationale=(
                        f"{MARK} `{col.name}` has {col.n_unique:,} levels — "
                        f"one-hot would add {col.n_unique:,} mostly-empty columns. "
                        "Target encoding compresses it to one column carrying the "
                        "per-level outcome rate."
                    ),
                    hypothesis=(
                        "Levels differ systematically in their outcome rate, and "
                        "that rate is the useful part of the category."
                    ),
                    risk=(
                        "Target encoding leaks unless it is fitted out-of-fold. "
                        "Computed with 5-fold out-of-fold means and smoothing=10 so "
                        "rare levels shrink toward the global mean."
                    ),
                    priority="medium",
                )
            )
            delta += 1
        else:
            decisions.append(
                FeatureDecision(
                    op=FeatureOp.FREQUENCY_ENCODE,
                    input_columns=[col.name],
                    output_name_hint=f"{col.name}_freq",
                    rationale=(
                        f"{MARK} `{col.name}` has {col.n_unique:,} levels and there "
                        "is no target to encode against, so level frequency is the "
                        "available ordering."
                    ),
                    risk="No leakage risk; frequency uses no label information.",
                    priority="low",
                )
            )
            delta += 1

    # --- text -------------------------------------------------------------
    for col in texts[:3]:
        decisions.append(
            FeatureDecision(
                op=FeatureOp.TEXT_LENGTH,
                input_columns=[col.name],
                output_name_hint=f"{col.name}_len",
                rationale=(
                    f"{MARK} `{col.name}` averages "
                    f"{_num(col.mean_string_length)} characters. Length alone is "
                    "often predictive and costs one column, so it is worth taking "
                    "before the expense of vectorising."
                ),
                priority="low",
            )
        )
        delta += 1
        decisions.append(
            FeatureDecision(
                op=FeatureOp.TEXT_TFIDF,
                input_columns=[col.name],
                parameters=[
                    Param(key="max_features", value="200"),
                    Param(key="ngram_range", value="[1, 2]"),
                ],
                rationale=(
                    f"{MARK} `{col.name}` is free text averaging "
                    f"{_num(col.mean_token_count)} tokens. TF-IDF over the top 200 "
                    "uni- and bigrams exposes lexical signal without the dimension "
                    "explosion of a full vocabulary."
                ),
                risk="200 sparse columns; capped deliberately to protect the row-to-column ratio.",
                priority="medium",
            )
        )
        delta += 200

    # --- redundancy --------------------------------------------------------
    redundant = [
        p for p in profile.highly_correlated_pairs if abs(p.coefficient) >= CORRELATION_REDUNDANT
    ]
    if redundant:
        decisions.append(
            FeatureDecision(
                op=FeatureOp.DROP_CORRELATED,
                input_columns=sorted({p.left for p in redundant} | {p.right for p in redundant}),
                parameters=[Param(key="threshold", value=str(CORRELATION_REDUNDANT))],
                rationale=(
                    f"{MARK} {len(redundant)} pair(s) correlate at or above "
                    f"|r|={CORRELATION_REDUNDANT}, e.g. `{redundant[0].left}` and "
                    f"`{redundant[0].right}` at {_num(redundant[0].coefficient)}. "
                    "Near-duplicates inflate variance in linear models and split "
                    "one feature's importance across two names, which makes the "
                    "explainability output misleading."
                ),
                hypothesis="One of each pair carries the information; the other repeats it.",
                risk="Dropping the wrong member of a pair loses nothing — they are collinear.",
                priority="medium",
            )
        )
        delta -= len(redundant)

    near_zero = [c for c in columns if c.is_near_zero_variance and not c.is_constant]
    if near_zero:
        decisions.append(
            FeatureDecision(
                op=FeatureOp.VARIANCE_THRESHOLD,
                input_columns=[c.name for c in near_zero],
                parameters=[Param(key="threshold", value="0.0")],
                rationale=(
                    f"{MARK} {len(near_zero)} column(s) measured near-zero "
                    "variance ("
                    + ", ".join(f"`{c.name}`" for c in near_zero[:5])
                    + "). A feature that barely varies cannot separate outcomes."
                ),
                priority="low",
            )
        )
        delta -= len(near_zero)

    # --- scaling -----------------------------------------------------------
    if numeric:
        has_outliers = any(c.outliers and c.outliers.fraction > 0.02 for c in numeric)
        op = FeatureOp.ROBUST_SCALE if has_outliers else FeatureOp.STANDARD_SCALE
        decisions.append(
            FeatureDecision(
                op=op,
                input_columns=[c.name for c in numeric],
                rationale=(
                    f"{MARK} {len(numeric)} numeric column(s) span very different "
                    "ranges ("
                    + ", ".join(
                        f"`{c.name}` {_num(c.minimum)}..{_num(c.maximum)}"
                        for c in numeric[:3]
                    )
                    + "). Logistic regression, SVM, and KNN are all distance- or "
                    "gradient-based and would let the widest-range column dominate. "
                    + (
                        "Robust scaling is used because outliers were measured, and "
                        "it centres on the median and scales by the IQR so extreme "
                        "values do not set the scale."
                        if has_outliers
                        else "Standard scaling is sufficient — no column carries a "
                        "heavy outlier fraction."
                    )
                ),
                risk=(
                    "Must be fitted on the training split only; fitting on the full "
                    "table would leak test statistics into training."
                ),
                priority="high",
            )
        )

    strategy = (
        f"Expected width change {delta:+,} columns against "
        f"{profile.n_columns:,} original. "
    )
    n_rows = profile.n_rows
    projected = max(profile.n_columns + delta, 1)
    if n_rows / projected < 10:
        strategy += (
            f"That leaves roughly {n_rows / projected:.1f} rows per column, which is "
            "thin — PCA or SelectKBest should follow if the leaderboard shows "
            "overfitting."
        )
    else:
        strategy += (
            f"That leaves roughly {n_rows / projected:.0f} rows per column, "
            "comfortable enough that no dimensionality reduction is scheduled."
        )

    return FeaturePlan(
        decisions=decisions,
        summary=(
            f"{MARK} {len(decisions)} operation(s): "
            f"{len(datetimes)} datetime column(s) decomposed, {len(skewed)} skewed "
            f"column(s) log-transformed, {len(low_card)} one-hot encoded, "
            f"{len(high_card)} high-cardinality column(s) encoded compactly, and "
            f"{len(numeric)} numeric column(s) scaled. Each operation is triggered "
            "by a measured property of the column it touches, not by a template."
        ),
        expected_feature_count_delta=delta,
        dimensionality_strategy=strategy,
        selection_strategy=(
            "No supervised selection is applied up front: with the measured "
            "row-to-column ratio, selection would risk discarding weak-but-real "
            "signal. Redundancy is removed structurally (correlation and variance) "
            "instead, which needs no label and therefore cannot leak."
        ),
    )


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def _model_selection(state: RunState, agent: Any = None) -> ModelSelection:
    profile = _profile(state)
    task = state.task_type or TaskType.BINARY_CLASSIFICATION
    n_rows = profile.n_rows if profile else 0
    n_cols = profile.n_columns if profile else 0
    imbalanced = bool(profile and profile.target and profile.target.is_imbalanced)

    candidates: list[ModelCandidate] = []
    excluded: list[str] = []
    exclusion_rationale: list[str] = []
    rank = 1

    def add(
        family: ModelFamily,
        suitability: str,
        rationale: str,
        strengths: list[str],
        weaknesses: list[str],
        *,
        baseline: bool = False,
        tune: str = "medium",
        params: list[Param] | None = None,
    ) -> None:
        nonlocal rank
        candidates.append(
            ModelCandidate(
                family=family,
                rank=rank,
                suitability=suitability,  # type: ignore[arg-type]
                rationale=f"{MARK} {rationale}",
                expected_strengths=strengths,
                expected_weaknesses=weaknesses,
                initial_params=params or [],
                is_baseline=baseline,
                tune_priority=tune,  # type: ignore[arg-type]
            )
        )
        rank += 1

    if task is TaskType.CLUSTERING:
        add(
            ModelFamily.KMEANS,
            "good",
            f"{n_rows:,} rows with numeric structure. K-means is the standard "
            "starting point: it scales linearly and its centroids are directly "
            "interpretable as segment profiles.",
            ["Fast on this size", "Centroids readable as segments"],
            ["Assumes roughly spherical, equally sized clusters"],
            tune="high",
        )
        add(
            ModelFamily.GAUSSIAN_MIXTURE,
            "fair",
            "Relaxes k-means' equal-variance assumption, so elongated or "
            "differently-sized clusters remain findable.",
            ["Soft assignment", "Handles elliptical clusters"],
            ["More parameters to fit on the same data"],
        )
        validation = "silhouette across a sweep of k"
        validation_rationale = (
            f"{MARK} No labels exist, so quality is internal: silhouette compares "
            "within-cluster cohesion against nearest-cluster separation."
        )
    elif task is TaskType.ANOMALY_DETECTION:
        add(
            ModelFamily.ISOLATION_FOREST,
            "good",
            f"Unlabelled anomaly detection over {n_cols:,} columns. Isolation "
            "Forest isolates rare points in few splits and does not assume a "
            "distribution.",
            ["Scales well", "No distributional assumption"],
            ["Contamination rate must be assumed"],
            tune="high",
        )
        validation = "holdout score separation"
        validation_rationale = f"{MARK} No labels; separation of score distributions is the signal."
    else:
        is_clf = task.is_classification
        # Baseline first — it is what every other number is judged against.
        add(
            ModelFamily.BASELINE_DUMMY,
            "poor",
            "Not a contender — the floor. Predicts the majority class or the mean, "
            "so any candidate that fails to beat it has learned nothing from the "
            "features. Without this number the leaderboard cannot be interpreted"
            + (
                f", and at {_num(profile.target.imbalance_ratio)}:1 imbalance a "
                "constant predictor already looks deceptively strong on accuracy."
                if imbalanced and profile and profile.target
                else "."
            ),
            ["Establishes the no-skill floor"],
            ["Zero predictive value by construction"],
            baseline=True,
            tune="none",
        )

        small = n_rows < 2000
        linear_family = ModelFamily.LOGISTIC if is_clf else ModelFamily.RIDGE
        if small:
            add(
                linear_family,
                "good",
                f"Only {n_rows:,} rows. A regularised linear model has the lowest "
                "variance of any candidate here, and at this sample size variance "
                "control matters more than the flexibility a boosted ensemble buys.",
                ["Stable on small samples", "Coefficients are directly readable"],
                ["Misses interactions unless they are engineered in"],
                tune="medium",
                params=[Param(key="max_iter", value="2000")],
            )
            add(
                ModelFamily.RANDOM_FOREST,
                "good",
                "Bagging averages away much of the variance that hurts a single "
                "tree on a small sample, and it captures the interactions the "
                "linear model cannot.",
                ["Captures interactions", "Little tuning needed to be reasonable"],
                ["Larger artifact", "Extrapolates poorly"],
                tune="medium",
                params=[Param(key="n_estimators", value="300")],
            )
        else:
            if _installed("lightgbm"):
                add(
                    ModelFamily.LIGHTGBM,
                    "excellent",
                    f"{n_rows:,} rows x {n_cols:,} columns of tabular data is "
                    "exactly where leaf-wise gradient boosting wins: it handles "
                    "mixed types and non-linear interactions with the best "
                    "accuracy-per-second of the available families.",
                    ["Strongest expected accuracy on tabular data", "Fast training"],
                    ["Overfits without early stopping", "Less interpretable than linear"],
                    tune="high",
                    params=[
                        Param(key="n_estimators", value="400"),
                        Param(key="learning_rate", value="0.05"),
                    ],
                )
            elif _installed("xgboost"):
                add(
                    ModelFamily.XGBOOST,
                    "excellent",
                    f"{n_rows:,} rows of tabular data; XGBoost is the strongest "
                    "boosting implementation installed in this environment.",
                    ["Strong tabular accuracy", "Handles missing values natively"],
                    ["Needs tuning to reach its ceiling"],
                    tune="high",
                )
            else:
                add(
                    ModelFamily.HIST_GRADIENT_BOOSTING,
                    "excellent",
                    f"{n_rows:,} rows of tabular data. Neither LightGBM nor XGBoost "
                    "is installed, and scikit-learn's histogram gradient boosting "
                    "is the same algorithm class with comparable accuracy.",
                    ["Strong tabular accuracy", "No extra dependency"],
                    ["Slightly slower than LightGBM at this size"],
                    tune="high",
                )
                excluded += ["lightgbm", "xgboost"]
                exclusion_rationale.append(
                    "LightGBM and XGBoost are not installed in this environment; "
                    "hist_gradient_boosting covers the same ground without the "
                    "dependency."
                )

            add(
                ModelFamily.RANDOM_FOREST,
                "good",
                "An independent second opinion with a different bias: bagging "
                "rather than boosting. If it lands close to the boosted model the "
                "result is likely real; if it lands far below, the boosting run is "
                "probably overfitting.",
                ["Robust default", "Different error profile from boosting"],
                ["Rarely the top scorer on tabular data"],
                tune="medium",
                params=[Param(key="n_estimators", value="300")],
            )
            add(
                linear_family,
                "fair",
                "A linear reference point. If it comes close to the ensembles, the "
                "relationship is mostly additive and the simpler, faster, more "
                "explainable model should ship instead.",
                ["Fast", "Interpretable coefficients", "Tiny artifact"],
                ["Cannot represent interactions unaided"],
                tune="low",
                params=[Param(key="max_iter", value="2000")],
            )

        if n_rows > 50_000:
            excluded.append("svm")
            exclusion_rationale.append(
                f"SVM excluded: kernel fitting is roughly quadratic in rows, and at "
                f"{n_rows:,} rows it would consume the run's entire time budget."
            )
        if n_rows > 20_000:
            excluded.append("knn")
            exclusion_rationale.append(
                f"KNN excluded: it stores all {n_rows:,} training rows and pays the "
                "cost at prediction time, which is the wrong trade for a deployed model."
            )

        if profile and profile.temporal_columns:
            validation = f"time-ordered split on `{profile.temporal_columns[0]}`"
            validation_rationale = (
                f"{MARK} `{profile.temporal_columns[0]}` orders the rows in time. A "
                "shuffled fold would train on later rows and score on earlier ones, "
                "reporting an accuracy that cannot exist in production."
            )
        elif is_clf:
            folds = state.config.cv_folds
            validation = f"stratified {folds}-fold cross-validation"
            validation_rationale = (
                f"{MARK} Stratification holds the class ratio constant in every "
                "fold"
                + (
                    f" — necessary at {_num(profile.target.imbalance_ratio)}:1 "
                    "imbalance, where an unstratified fold could contain almost no "
                    "positives."
                    if imbalanced and profile and profile.target
                    else "."
                )
            )
        else:
            validation = f"{state.config.cv_folds}-fold cross-validation"
            validation_rationale = (
                f"{MARK} Averaging over {state.config.cv_folds} folds gives a more "
                f"stable error estimate than one holdout at {n_rows:,} rows."
            )

    reasoning = (
        f"{MARK} The shortlist follows from size and shape rather than habit. At "
        f"{n_rows:,} rows and {n_cols:,} columns, "
        + (
            "the sample is small enough that variance dominates bias, so a "
            "regularised linear model leads and bagging provides the non-linear "
            "check."
            if n_rows < 2000
            else "there is enough data to support a boosted ensemble, which is the "
            "strongest family for tabular problems of this size."
        )
        + " A dummy baseline is included deliberately: it is the only way to tell "
        "whether the features contribute anything at all. Families whose cost "
        "scales badly at this row count are excluded by name and reason rather "
        "than silently omitted."
    )

    return ModelSelection(
        candidates=candidates,
        summary=(
            f"{MARK} {len(candidates)} candidate(s) ranked for {task.value}: "
            + ", ".join(f"{c.rank}. {c.family.value}" for c in candidates)
            + "."
        ),
        reasoning=reasoning,
        excluded_families=excluded,
        exclusion_rationale=exclusion_rationale,
        validation_strategy=validation,
        validation_rationale=validation_rationale,
    )


# ---------------------------------------------------------------------------
# Experiments (hybrid — measurements already computed)
# ---------------------------------------------------------------------------


def _experiments(state: RunState, agent: Any = None) -> ExperimentLog:
    computed = getattr(agent, "_computed", None) or state.experiments
    if computed is None:
        return ExperimentLog(
            primary_metric=state.primary_metric,
            leaderboard_notes=f"{MARK} No experiments were recorded.",
        )

    log = computed.model_copy(deep=True)
    ok = [r for r in log.results if not r.failed]
    failed = [r for r in log.results if r.failed]
    best = next((r for r in log.results if r.experiment_id == log.best_experiment_id), None)
    baseline = next((r for r in log.results if r.is_baseline), None)

    lines: list[str] = []
    if best is not None:
        lines.append(
            f"{MARK} `{best.family.value}` leads on {log.primary_metric} at "
            f"{_num(best.primary_score, 4)}, trained in "
            f"{_num(best.train_seconds, 1)}s on {best.n_features_in:,} features."
        )
        if baseline is not None and baseline.primary_score is not None and best.primary_score is not None:
            delta = best.primary_score - baseline.primary_score
            direction = 1 if log.higher_is_better else -1
            improved = delta * direction > 0
            lines.append(
                f"Against the dummy baseline ({_num(baseline.primary_score, 4)}) that "
                f"is a {'gain' if improved else 'LOSS'} of {_num(abs(delta), 4)}. "
                + (
                    "The features carry real signal."
                    if improved
                    else "The features are not contributing — this result is not usable."
                )
            )
    if len(ok) > 1:
        spread = [r for r in ok if r.primary_score is not None]
        if len(spread) > 1:
            scores = sorted(
                (r.primary_score for r in spread if r.primary_score is not None),
                reverse=log.higher_is_better,
            )
            lines.append(
                f"{len(ok)} candidates completed; scores span "
                f"{_num(scores[0], 4)} to {_num(scores[-1], 4)}. "
                + (
                    "A narrow spread means the choice of family matters less than "
                    "the features do."
                    if abs(scores[0] - scores[-1]) < 0.05
                    else "The spread is wide enough that family choice is doing real work."
                )
            )
    if failed:
        lines.append(
            f"{len(failed)} candidate(s) failed to train: "
            + "; ".join(f"{r.family.value} ({r.error})" for r in failed[:3])
            + "."
        )

    log.leaderboard_notes = " ".join(lines) if lines else f"{MARK} Leaderboard recorded."
    return log


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


_SEARCH_SPACES: dict[ModelFamily, list[SearchSpaceEntry]] = {
    ModelFamily.LIGHTGBM: [
        SearchSpaceEntry(name="n_estimators", kind="int", low=100, high=1000, rationale="Boosting rounds trade fit against overfitting."),
        SearchSpaceEntry(name="learning_rate", kind="log_float", low=0.01, high=0.3, rationale="Interacts inversely with n_estimators."),
        SearchSpaceEntry(name="num_leaves", kind="int", low=15, high=255, rationale="Primary capacity control for leaf-wise growth."),
        SearchSpaceEntry(name="min_child_samples", kind="int", low=5, high=100, rationale="Guards against leaves fitted to a handful of rows."),
        SearchSpaceEntry(name="subsample", kind="float", low=0.6, high=1.0, rationale="Row sampling decorrelates trees."),
        SearchSpaceEntry(name="colsample_bytree", kind="float", low=0.6, high=1.0, rationale="Column sampling decorrelates trees."),
    ],
    ModelFamily.XGBOOST: [
        SearchSpaceEntry(name="n_estimators", kind="int", low=100, high=1000, rationale="Boosting rounds."),
        SearchSpaceEntry(name="learning_rate", kind="log_float", low=0.01, high=0.3, rationale="Step size per round."),
        SearchSpaceEntry(name="max_depth", kind="int", low=3, high=10, rationale="Primary capacity control for depth-wise growth."),
        SearchSpaceEntry(name="subsample", kind="float", low=0.6, high=1.0, rationale="Row sampling."),
        SearchSpaceEntry(name="colsample_bytree", kind="float", low=0.6, high=1.0, rationale="Column sampling."),
        SearchSpaceEntry(name="reg_lambda", kind="log_float", low=0.1, high=10.0, rationale="L2 penalty on leaf weights."),
    ],
    ModelFamily.HIST_GRADIENT_BOOSTING: [
        SearchSpaceEntry(name="max_iter", kind="int", low=100, high=800, rationale="Boosting rounds."),
        SearchSpaceEntry(name="learning_rate", kind="log_float", low=0.01, high=0.3, rationale="Step size per round."),
        SearchSpaceEntry(name="max_leaf_nodes", kind="int", low=15, high=127, rationale="Capacity control."),
        SearchSpaceEntry(name="min_samples_leaf", kind="int", low=5, high=100, rationale="Leaf-size floor."),
    ],
    ModelFamily.RANDOM_FOREST: [
        SearchSpaceEntry(name="n_estimators", kind="int", low=100, high=800, rationale="More trees reduce variance monotonically."),
        SearchSpaceEntry(name="max_depth", kind="int", low=3, high=30, rationale="Depth cap limits memorisation."),
        SearchSpaceEntry(name="min_samples_leaf", kind="int", low=1, high=20, rationale="Leaf-size floor."),
        SearchSpaceEntry(name="max_features", kind="categorical", choices=["sqrt", "log2", "0.5"], rationale="Controls tree decorrelation."),
    ],
    ModelFamily.LOGISTIC: [
        SearchSpaceEntry(name="C", kind="log_float", low=0.001, high=100.0, rationale="Inverse regularisation strength."),
        SearchSpaceEntry(name="penalty", kind="categorical", choices=["l1", "l2"], rationale="L1 also performs selection."),
    ],
    ModelFamily.RIDGE: [
        SearchSpaceEntry(name="alpha", kind="log_float", low=0.001, high=100.0, rationale="Regularisation strength."),
    ],
    ModelFamily.KMEANS: [
        SearchSpaceEntry(name="n_clusters", kind="int", low=2, high=12, rationale="The only structural choice k-means makes."),
    ],
}


def _tuning(state: RunState, agent: Any = None) -> TuningDecision:
    log = state.experiments
    profile = _profile(state)
    n_rows = profile.n_rows if profile else 0
    remaining = state.time_remaining

    best = None
    baseline = None
    if log:
        best = next((r for r in log.results if r.experiment_id == log.best_experiment_id), None)
        baseline = next((r for r in log.results if r.is_baseline), None)

    if best is None:
        return TuningDecision(
            worthwhile=False,
            rationale=(
                f"{MARK} No candidate trained successfully, so there is no model to "
                "tune. Tuning a model that does not exist would burn the remaining "
                "budget for nothing."
            ),
            method=TuningMethod.NONE,
            n_trials=0,
            timeout_seconds=0,
        )

    family = best.family
    space = _SEARCH_SPACES.get(family, [])

    # --- is it worth it? -------------------------------------------------
    reasons_against: list[str] = []
    if not state.config.enable_tuning:
        reasons_against.append("tuning is disabled by run configuration")
    if remaining < 60:
        reasons_against.append(
            f"only {_num(remaining, 0)}s of the time budget remains, below the 60s "
            "floor a search needs to complete even a handful of trials"
        )
    if not space:
        reasons_against.append(
            f"no search space is defined for `{family.value}`, so a search would "
            "have nothing to vary"
        )
    if n_rows < 300:
        reasons_against.append(
            f"at {n_rows:,} rows, cross-validated scores vary more between folds "
            "than between hyperparameter settings — a search would be fitting noise"
        )

    lift_note = ""
    if baseline is not None and baseline.primary_score is not None and best.primary_score is not None:
        delta = abs(best.primary_score - baseline.primary_score)
        lift_note = (
            f" The winner beats the dummy baseline by {_num(delta, 4)} on "
            f"{log.primary_metric if log else state.primary_metric}, so the features "
            "carry signal worth refining."
        )

    if reasons_against:
        return TuningDecision(
            worthwhile=False,
            rationale=(
                f"{MARK} Tuning is not worth the compute here: "
                + "; ".join(reasons_against)
                + f". `{family.value}` ships at its default settings."
            ),
            method=TuningMethod.NONE,
            target_family=family,
            n_trials=0,
            timeout_seconds=0,
            expected_gain="None — no search will be run.",
        )

    if _installed("optuna"):
        method = TuningMethod.OPTUNA_TPE
        method_rationale = (
            f"{MARK} Optuna's TPE sampler concentrates trials in the promising "
            "region instead of sampling uniformly, which matters because the "
            f"budget only allows a few dozen trials over {len(space)} dimensions. "
            "Grid search over the same space would need thousands."
        )
    else:
        method = TuningMethod.RANDOM_SEARCH
        method_rationale = (
            f"{MARK} Optuna is not installed, so random search is used. Over "
            f"{len(space)} dimensions random search still beats grid search at "
            "equal budget, because most dimensions matter far less than one or two."
        )

    budget = max(60, int(min(remaining * 0.5, 600)))
    trials = max(10, min(50, int(budget / 6)))

    return TuningDecision(
        worthwhile=True,
        rationale=(
            f"{MARK} `{family.value}` won the leaderboard at "
            f"{_num(best.primary_score, 4)} and trained in "
            f"{_num(best.train_seconds, 1)}s, so a trial is cheap enough that "
            f"{trials} of them fit inside the {_num(remaining, 0)}s remaining."
            + lift_note
            + " Boosting and forest defaults are chosen for generality, not for "
            "this dataset, so there is normally headroom in capacity and "
            "regularisation."
        ),
        method=method,
        method_rationale=method_rationale,
        target_family=family,
        n_trials=trials,
        timeout_seconds=budget,
        early_stopping=True,
        search_space=space,
        expected_gain=(
            "Typically a small single-digit-percent improvement on the primary "
            "metric. The tuned model is only adopted if it actually beats the "
            "untuned score on held-out data."
        ),
    )


# ---------------------------------------------------------------------------
# Explainability (hybrid)
# ---------------------------------------------------------------------------


def _explain(state: RunState, agent: Any = None) -> ExplainabilityReport:
    computed = getattr(agent, "_computed", None) or state.explainability
    report = (computed or ExplainabilityReport()).model_copy(deep=True)

    attributions = report.global_attributions or report.permutation_importance
    target = state.target or "the outcome"
    method = "SHAP" if report.shap_available else "permutation importance"

    plain: list[str] = []
    for attribution in attributions[:8]:
        share = _pct(attribution.importance, 0)
        direction = {
            "increases": f"higher values push predictions toward a higher {target}",
            "decreases": f"higher values push predictions toward a lower {target}",
            "mixed": "its effect changes direction depending on the rest of the row",
            "unknown": "the direction of its effect was not measured",
        }[attribution.direction]
        plain.append(
            f"`{attribution.feature}` contributes approximately {share} of the "
            f"model's {target} prediction importance — {direction}."
        )

    if attributions:
        top = attributions[0]
        concentration = sum(a.importance for a in attributions[:3])
        narrative = (
            f"{MARK} Attribution was computed by {method} over the fitted model. "
            f"`{top.feature}` is the strongest single driver at "
            f"{_pct(top.importance, 0)} of total importance, and the top three "
            f"features together account for {_pct(concentration, 0)}. "
            + (
                "That concentration means the model is effectively reasoning from a "
                "handful of signals, which is good for explainability but leaves it "
                "exposed if one of those inputs degrades in production."
                if concentration > 0.6
                else "Importance is spread broadly across the feature set, so no "
                "single input failure would break the model — though it also means "
                "no small set of factors tells the whole story."
            )
            + " Importance measures how much the model relies on a feature, not "
            "that the feature causes the outcome; acting on these as levers "
            "requires a causal design this run did not perform."
        )
    else:
        narrative = (
            f"{MARK} No attributions were computed, so the model's behaviour cannot "
            "be explained from this run."
        )

    report.plain_language_explanations = plain
    report.narrative = narrative
    report.method_notes = (
        f"{MARK} Method: {method}."
        + (
            " SHAP values are computed on a sample of the holdout set."
            if report.shap_available
            else " SHAP was unavailable, so permutation importance on held-out data "
            "was used instead; it measures the score drop when a feature is shuffled."
        )
        + " Narration is rule-generated in offline mode."
    )
    return report


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

#: Metrics bounded on [0, 1] where an absolute grade is meaningful.
_BOUNDED_HIGHER = {
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "average_precision",
    "precision",
    "recall",
    "r2",
}


def _evaluation(state: RunState, agent: Any = None) -> EvaluationVerdict:
    log = state.experiments
    metric = state.primary_metric
    best = None
    baseline = None
    if log:
        best = next((r for r in log.results if r.experiment_id == log.best_experiment_id), None)
        baseline = next((r for r in log.results if r.is_baseline), None)

    if best is None or best.primary_score is None:
        return EvaluationVerdict(
            acceptable=False,
            verdict_rationale=(
                f"{MARK} No candidate produced a score, so there is nothing to "
                "accept. This is a pipeline failure, not a modelling result."
            ),
            overall_grade="F",
            recommended_action="retry_model_selection",
            action_rationale=(
                f"{MARK} Re-select model families; the current shortlist did not "
                "yield a trainable candidate in this environment."
            ),
            weaknesses=["No model trained successfully."],
        )

    score = best.primary_score
    try:
        from ..execution.metrics import metric_higher_is_better

        higher_better = metric_higher_is_better(metric)
    except Exception:  # noqa: BLE001
        higher_better = log.higher_is_better if log else True

    baseline_score = baseline.primary_score if baseline else None
    beats_baseline = True
    lift_text = ""
    if baseline_score is not None:
        delta = score - baseline_score
        beats_baseline = (delta > 0) if higher_better else (delta < 0)
        lift_text = (
            f" Against the no-skill baseline of {_num(baseline_score, 4)} that is a "
            f"{'gain' if beats_baseline else 'shortfall'} of {_num(abs(delta), 4)}."
        )

    # --- grade ------------------------------------------------------------
    if metric in _BOUNDED_HIGHER and higher_better:
        if score >= 0.90:
            grade, quality = "A", "strong"
        elif score >= 0.80:
            grade, quality = "B", "solid"
        elif score >= 0.70:
            grade, quality = "C", "modest"
        elif score >= 0.60:
            grade, quality = "D", "weak"
        else:
            grade, quality = "F", "no better than guessing"
        grade_basis = (
            f"{metric} of {_num(score, 4)} on held-out data, graded against the "
            "conventional bands for a bounded ranking metric"
        )
    elif baseline_score is not None and baseline_score != 0:
        relative = abs(score - baseline_score) / abs(baseline_score)
        if not beats_baseline:
            grade, quality = "F", "worse than the baseline"
        elif relative >= 0.40:
            grade, quality = "A", "strong"
        elif relative >= 0.25:
            grade, quality = "B", "solid"
        elif relative >= 0.10:
            grade, quality = "C", "modest"
        else:
            grade, quality = "D", "marginal"
        grade_basis = (
            f"{_pct(relative)} improvement over the baseline on {metric} — an "
            "unbounded metric has no absolute scale, so the baseline is the only "
            "honest reference"
        )
    else:
        grade, quality = "C", "unrated"
        grade_basis = (
            f"{metric} of {_num(score, 4)} with no baseline to compare against, so "
            "the grade is provisional"
        )

    if not beats_baseline:
        grade = "F"

    acceptable = grade in ("A", "B", "C") and beats_baseline

    weaknesses: list[str] = []
    improvements: list[str] = []
    if not beats_baseline:
        weaknesses.append(
            "The model does not beat a constant predictor — the features contribute "
            "nothing the model can use."
        )
        improvements.append(
            "Revisit feature engineering: add interaction and ratio terms, and "
            "confirm the target is not effectively random with respect to the inputs."
        )
    if grade in ("C", "D"):
        improvements.append(
            "Add interaction features between the top attributed columns, and "
            "consider whether a domain-derived feature is missing entirely."
        )
    profile = _profile(state)
    if profile and profile.n_rows < 1000:
        weaknesses.append(
            f"{profile.n_rows:,} rows makes the holdout estimate noisy; the true "
            "score could differ materially from the measured one."
        )
        improvements.append("Collect more rows before trusting this estimate.")
    if profile and profile.target and profile.target.is_imbalanced:
        weaknesses.append(
            f"Class imbalance of {_num(profile.target.imbalance_ratio)}:1 means "
            "minority-class recall deserves separate scrutiny before deployment."
        )
    if state.warnings:
        weaknesses.append(
            f"{len(state.warnings)} warning(s) were raised during the run; review "
            "them before relying on this result."
        )

    if not acceptable:
        action = (
            "retry_feature_engineering"
            if not beats_baseline or grade == "D"
            else "collect_more_data"
        )
        action_rationale = (
            f"{MARK} Grade {grade} is below the acceptance floor. "
            + (
                "The features are the binding constraint — the model cannot beat a "
                "constant predictor, so no amount of tuning or model swapping will "
                "help."
                if not beats_baseline
                else "Feature engineering is the highest-leverage retry: the model "
                "families have already been compared and differ less than the "
                "feature set would."
            )
        )
    else:
        action = "accept"
        action_rationale = (
            f"{MARK} Grade {grade} clears the acceptance floor and the model beats "
            "the no-skill baseline, so the result is fit to report."
        )

    drift_risk = "unknown"
    drift_rationale = (
        f"{MARK} No temporal holdout was constructed, so drift cannot be measured "
        "from this run."
    )
    if profile and profile.temporal_columns:
        drift_risk = "medium"
        drift_rationale = (
            f"{MARK} The data carries a time column (`{profile.temporal_columns[0]}`), "
            "which means the relationships learned here are anchored to a period "
            "and can decay. Monitor the primary metric on fresh data and retrain "
            "when it degrades."
        )

    return EvaluationVerdict(
        acceptable=acceptable,
        verdict_rationale=(
            f"{MARK} `{best.family.value}` scores {_num(score, 4)} on {metric} — "
            f"{quality}.{lift_text} Grade {grade} is assigned on {grade_basis}."
        ),
        overall_grade=grade,  # type: ignore[arg-type]
        bias_variance=BiasVarianceDiagnosis(
            test_score=score,
            detail=(
                f"{MARK} Train/holdout comparison is restored from the measured "
                "diagnostics bundle where available."
            ),
        ),
        generalisation_notes=(
            f"{MARK} The score is measured on data held out before any fitting, "
            "including imputation and encoding, so it is not inflated by "
            "preprocessing leakage."
        ),
        drift_risk=drift_risk,  # type: ignore[arg-type]
        drift_rationale=drift_rationale,
        residual_notes=f"{MARK} Residual structure was not analysed in offline mode.",
        error_analysis=[
            f"{MARK} Per-slice error analysis requires reasoning about which slices "
            "matter to the business, which offline mode cannot supply."
        ],
        weaknesses=weaknesses,
        recommended_action=action,  # type: ignore[arg-type]
        action_rationale=action_rationale,
        specific_improvements=improvements,
    )


# ---------------------------------------------------------------------------
# Business insight
# ---------------------------------------------------------------------------


def _insight(state: RunState, agent: Any = None) -> InsightReport:
    verdict = state.evaluation
    log = state.experiments
    explain = state.explainability
    target = state.target or "the outcome"
    metric = state.primary_metric

    best = None
    if log:
        best = next((r for r in log.results if r.experiment_id == log.best_experiment_id), None)

    attributions = []
    if explain:
        attributions = explain.global_attributions or explain.permutation_importance

    insights: list[BusinessInsight] = []

    if best is not None and best.primary_score is not None:
        usable = bool(verdict and verdict.acceptable)
        insights.append(
            BusinessInsight(
                headline=(
                    f"{target} can be predicted well enough to act on."
                    if usable
                    else f"{target} cannot yet be predicted reliably enough to act on."
                ),
                detail=(
                    f"The best model, {best.family.value}, scores "
                    f"{_num(best.primary_score, 4)} on {metric} against data it "
                    "never saw during training. "
                    + (
                        "That is strong enough to prioritise where attention goes: "
                        "the highest-scoring records are materially more likely to "
                        "be the ones that matter."
                        if usable
                        else "That is not strong enough to drive decisions. Acting "
                        "on it would mean sending real resources after predictions "
                        "that are close to guesses."
                    )
                ),
                supporting_evidence=(
                    f"Holdout {metric} = {_num(best.primary_score, 4)}; evaluation "
                    f"grade {verdict.overall_grade if verdict else 'n/a'}."
                ),
                recommended_action=(
                    "Pilot the model on one segment, measure the lift against "
                    "current practice, and expand only if the pilot holds."
                    if usable
                    else "Do not deploy. Invest in richer inputs before retrying."
                ),
                confidence="medium" if usable else "high",
                audience="executive",
            )
        )

    for position, attribution in enumerate(attributions[:3]):
        insights.append(
            BusinessInsight(
                headline=(
                    f"{attribution.feature.replace('_', ' ').title()} is the "
                    f"strongest measured driver of {target}."
                    if position == 0
                    else f"{attribution.feature.replace('_', ' ').title()} is a "
                    f"significant secondary driver of {target}."
                ),
                detail=(
                    f"It accounts for roughly {_pct(attribution.importance, 0)} of "
                    f"the model's decision. In practice this means records are "
                    f"separated more by `{attribution.feature}` than by almost "
                    "anything else recorded — the model reaches its conclusion "
                    "largely by reading this field."
                ),
                supporting_evidence=(
                    f"{attribution.method} importance {_num(attribution.importance)} "
                    f"(rank {position + 1} of {len(attributions)})."
                ),
                recommended_action=(
                    f"Check that `{attribution.feature}` is captured reliably at "
                    "the point of decision — if it is missing or stale in "
                    "production, the model loses most of its accuracy."
                ),
                confidence="medium",
                audience="operations",
            )
        )

    profile = _profile(state)
    if profile and profile.target and profile.target.is_imbalanced:
        insights.append(
            BusinessInsight(
                headline=(
                    f"The event being predicted is rare, so precision and recall "
                    "must be traded deliberately."
                ),
                detail=(
                    f"The target is imbalanced at "
                    f"{_num(profile.target.imbalance_ratio)}:1. Chasing every "
                    "flagged record wastes effort on false positives; a tight "
                    "threshold misses real cases. The right cut-off depends on what "
                    "an intervention costs versus what a miss costs — a business "
                    "input this run does not have."
                ),
                supporting_evidence=(
                    f"Measured class ratio {_num(profile.target.imbalance_ratio)}:1 "
                    f"across {profile.n_rows:,} rows."
                ),
                recommended_action=(
                    "Supply the cost of an intervention and the cost of a miss, "
                    "then set the threshold to minimise expected cost rather than "
                    "to maximise a generic metric."
                ),
                confidence="high",
                audience="operations",
            )
        )

    if not insights:
        insights.append(
            BusinessInsight(
                headline="This run produced no result that supports a business decision.",
                detail=(
                    "No model completed training, so there is nothing to translate "
                    "into a recommendation."
                ),
                supporting_evidence="Empty experiment log.",
                recommended_action="Investigate the recorded warnings and re-run.",
                confidence="high",
                audience="data_team",
            )
        )

    drivers = [
        f"`{a.feature}` accounts for about {_pct(a.importance, 0)} of the model's "
        f"{target} decision."
        for a in attributions[:5]
    ]

    summary_parts = []
    if best is not None and best.primary_score is not None:
        summary_parts.append(
            f"A {best.family.value} model predicts {target} at {_num(best.primary_score, 4)} "
            f"{metric} on held-out data"
            + (
                f", graded {verdict.overall_grade} and "
                + ("cleared" if verdict.acceptable else "NOT cleared")
                + " for use."
                if verdict
                else "."
            )
        )
    if attributions:
        summary_parts.append(
            f"Its decisions rest mainly on `{attributions[0].feature}`"
            + (
                f" and `{attributions[1].feature}`"
                if len(attributions) > 1
                else ""
            )
            + "."
        )
    summary_parts.append(
        "These conclusions are drawn from measured model behaviour by a rule "
        "engine; offline mode cannot supply the domain context that turns a "
        "statistical driver into a business explanation."
    )

    return InsightReport(
        executive_summary=f"{MARK} " + " ".join(summary_parts),
        insights=insights,
        key_drivers_plain_language=drivers,
        caveats=[
            "Feature importance shows what the model relies on, not what causes "
            "the outcome. Changing a high-importance field will not move the real "
            "outcome unless the relationship is genuinely causal.",
            "Scores are measured on historical data. They hold in production only "
            "while the underlying population behaves as it did in this sample.",
            "Offline mode: the business framing is rule-derived and has not been "
            "checked against domain knowledge.",
        ],
        suggested_next_experiments=[
            "Run a holdout pilot against current practice to measure real lift "
            "rather than modelled lift.",
            "Add features capturing recent behaviour if the source systems record "
            "it — recency is usually the strongest missing signal.",
            "Re-run with a live model to obtain domain-aware framing of these "
            "same measurements.",
        ],
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _visualization(state: RunState, agent: Any = None) -> VisualizationPlan:
    task = state.task_type or TaskType.BINARY_CLASSIFICATION
    profile = _profile(state)
    target = state.target
    charts: list[ChartSpec] = []

    def add(kind: ChartKind, title: str, rationale: str, columns: list[str] | None = None, priority: str = "medium") -> None:
        charts.append(
            ChartSpec(
                kind=kind,
                title=title,
                columns=columns or [],
                rationale=f"{MARK} {rationale}",
                priority=priority,  # type: ignore[arg-type]
            )
        )

    if profile and profile.total_missing_cells:
        add(
            ChartKind.MISSINGNESS,
            "Missing values by column",
            f"{_pct(profile.missing_cell_fraction)} of cells are missing; the "
            "pattern shows whether absence is concentrated or spread, which "
            "determines whether it is likely informative.",
            priority="high",
        )

    if profile and profile.target:
        if profile.target.n_classes:
            add(
                ChartKind.CLASS_BALANCE,
                f"Class balance of {target}",
                f"The target splits {_num(profile.target.imbalance_ratio)}:1; the "
                "reader needs this before any accuracy figure is meaningful.",
                columns=[target] if target else [],
                priority="high",
            )
        else:
            add(
                ChartKind.HISTOGRAM,
                f"Distribution of {target}",
                f"The target is continuous with skewness "
                f"{_num(profile.target.skewness)}; its shape determines whether "
                "error is dominated by a tail.",
                columns=[target] if target else [],
                priority="high",
            )

    if profile and profile.top_correlations:
        add(
            ChartKind.CORRELATION_HEATMAP,
            "Feature correlation matrix",
            f"{len(profile.highly_correlated_pairs)} pair(s) exceed the redundancy "
            "threshold; the matrix shows which blocks of features move together.",
            priority="medium",
        )

    # Charting a column that cleaning removed would illustrate a feature the
    # model never saw — worse than omitting the chart.
    gone = set(state.dropped_columns) | set(
        state.cleaning.columns_to_drop if state.cleaning else []
    )
    numeric = [
        c
        for c in _feature_columns(state)
        if c.kind in _NUMERIC_KINDS and c.name not in gone
    ]
    skewed = sorted(
        (c for c in numeric if c.skewness is not None),
        key=lambda c: abs(c.skewness or 0),
        reverse=True,
    )[:3]
    for col in skewed:
        add(
            ChartKind.HISTOGRAM,
            f"Distribution of {col.name}",
            f"Skewness {_num(col.skewness)} drove the imputation and transform "
            "choices for this column; the histogram is the evidence.",
            columns=[col.name],
            priority="low",
        )

    if state.explainability and (
        state.explainability.global_attributions or state.explainability.permutation_importance
    ):
        add(
            ChartKind.FEATURE_IMPORTANCE,
            "Feature importance",
            "Ranks what the model actually relies on — the single most useful "
            "chart for anyone deciding whether to trust it.",
            priority="high",
        )
        if state.explainability.shap_available:
            add(
                ChartKind.SHAP_SUMMARY,
                "SHAP summary",
                "Shows direction as well as magnitude per feature, which bare "
                "importance cannot.",
                priority="high",
            )

    if state.experiments and len(state.experiments.results) > 1:
        add(
            ChartKind.LEADERBOARD,
            "Model leaderboard",
            f"{len(state.experiments.results)} candidates were trained on identical "
            "splits; the comparison is only fair because the splits were shared.",
            priority="high",
        )

    if task.is_classification:
        add(ChartKind.ROC_CURVE, "ROC curve", "Shows the full precision/recall trade-off rather than one threshold.", priority="high")
        add(ChartKind.CONFUSION_MATRIX, "Confusion matrix", "Exposes which class the errors fall on — the asymmetry that matters operationally.", priority="high")
        add(ChartKind.PR_CURVE, "Precision-recall curve", "More informative than ROC when the positive class is rare.", priority="medium")
        add(ChartKind.CALIBRATION_CURVE, "Calibration curve", "Shows whether a predicted 0.8 actually happens 80% of the time — required before scores are read as probabilities.", priority="medium")
    elif task in (TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING):
        add(ChartKind.RESIDUALS, "Residuals vs predicted", "Reveals heteroscedasticity and systematic bias that a single RMSE hides.", priority="high")
        add(ChartKind.RESIDUAL_HISTOGRAM, "Residual distribution", "Shows whether errors are centred and symmetric or dominated by a tail.", priority="medium")
        add(ChartKind.PREDICTION_DISTRIBUTION, "Predicted vs actual", "Shows whether the model compresses toward the mean.", priority="medium")
        if task is TaskType.TIME_SERIES_FORECASTING:
            add(ChartKind.TIME_SERIES_FORECAST, "Forecast vs actual over time", "The only view that exposes lag and phase error.", priority="high")

    add(
        ChartKind.LEARNING_CURVE,
        "Learning curve",
        "Separates a data problem from a model problem: if both curves plateau "
        "together, more rows will not help.",
        priority="low",
    )

    return VisualizationPlan(
        charts=charts,
        dashboard_narrative=(
            f"{MARK} {len(charts)} chart(s), ordered from data structure to model "
            "behaviour. The data charts justify the cleaning and feature decisions; "
            "the model charts justify — or undermine — the headline score. Chart "
            "selection is driven by the measured profile and the task type."
        ),
    )


# ---------------------------------------------------------------------------
# Deployment + final report
# ---------------------------------------------------------------------------


def _deployment(state: RunState) -> DeploymentRecommendation:
    log = state.experiments
    best = None
    if log:
        best = next((r for r in log.results if r.experiment_id == log.best_experiment_id), None)

    size_mb = (best.model_size_bytes / 1_048_576) if best and best.model_size_bytes else None
    predict_s = best.predict_seconds if best else None
    n_rows = _profile(state).n_rows if _profile(state) else 0
    latency_ms = None
    if predict_s and n_rows:
        latency_ms = (predict_s / max(n_rows, 1)) * 1000

    profile = _profile(state)
    temporal = bool(profile and profile.temporal_columns)

    if latency_ms is not None and latency_ms < 5 and (size_mb or 0) < 100:
        pattern = DeploymentPattern.REST_API
        rationale = (
            f"Measured inference cost is about {_num(latency_ms, 3)} ms per row and "
            f"the serialised model is {_num(size_mb, 1)} MB. Both fit comfortably "
            "inside a synchronous request budget, so a REST endpoint gives the "
            "widest usefulness without special infrastructure."
        )
    elif size_mb is not None and size_mb > 500:
        pattern = DeploymentPattern.BATCH_INFERENCE
        rationale = (
            f"The model serialises to {_num(size_mb, 1)} MB, which is heavy to hold "
            "resident behind a request path. Scoring on a schedule amortises the "
            "load cost across the whole batch."
        )
    else:
        pattern = DeploymentPattern.BATCH_INFERENCE
        rationale = (
            "Nightly batch scoring is the lowest-risk starting point: it needs no "
            "latency guarantee, makes every prediction auditable before it is "
            "used, and is the simplest thing that delivers value. Move to a "
            "synchronous endpoint only when a use case genuinely needs a "
            "prediction inside a user interaction."
        )
        if latency_ms is not None:
            rationale += (
                f" Measured inference is about {_num(latency_ms, 3)} ms per row, so "
                "the option stays open."
            )

    monitoring = [
        f"Track {state.primary_metric} on labelled outcomes as they arrive; alert "
        "on sustained degradation rather than single-day noise.",
        "Monitor the input distribution of the top attributed features — drift "
        "there precedes score decay.",
        "Alert on missing-value rates rising above what was measured in training; "
        "the imputers were fitted to those rates.",
        "Log every prediction with its inputs and model version so any decision "
        "can be reconstructed.",
    ]
    risks = [
        "Training data is historical; performance holds only while the population "
        "resembles this sample.",
    ]
    if temporal:
        risks.append(
            "The data is time-ordered, so relationships are anchored to a period "
            "and will decay. Retraining cadence should be treated as a requirement, "
            "not an optimisation."
        )
    if profile and profile.n_rows < 1000:
        risks.append(
            f"Only {profile.n_rows:,} training rows — the deployed model may behave "
            "differently from the measured estimate."
        )
    risks.append(
        "Offline mode: deployment sizing is derived from measured artifact size and "
        "inference time only, with no knowledge of the actual traffic profile."
    )

    return DeploymentRecommendation(
        pattern=pattern,
        rationale=f"{MARK} {rationale}",
        estimated_latency_ms=latency_ms,
        model_size_mb=size_mb,
        infrastructure_notes=(
            f"{MARK} The fitted preprocessing and the estimator ship as one "
            "pipeline artifact, so inference applies exactly the transformations "
            "that were fitted at training time. Deploying the estimator alone is "
            "the most common cause of a model that scores well and then fails in "
            "production."
        ),
        monitoring_plan=monitoring,
        retraining_cadence=(
            "Monthly, or whenever monitored performance drops materially below the "
            "recorded holdout score — whichever comes first."
            if temporal
            else "Quarterly, or on a monitored performance drop."
        ),
        rollout_strategy=(
            "Shadow first: score live traffic without acting on it and compare "
            "against current practice. Promote to a small share of decisions, then "
            "widen once observed lift matches the holdout estimate."
        ),
        risks=risks,
    )


def _report(state: RunState, agent: Any = None) -> FinalReport:
    """Author the summary and deployment view; let the agent synthesise sections.

    ``ReportAgent.postprocess`` rebuilds any of the nine required sections that
    are absent, using the same deterministic builders it uses online. Returning
    an empty section list is therefore not a gap — it routes section assembly
    through one tested code path instead of duplicating it here.
    """
    log = state.experiments
    verdict = state.evaluation
    profile = _profile(state)
    best = None
    if log:
        best = next((r for r in log.results if r.experiment_id == log.best_experiment_id), None)

    target = state.target or "the outcome"
    task = state.task_type.value.replace("_", " ") if state.task_type else "analysis"

    if best is not None and best.primary_score is not None:
        headline = (
            f"A {best.family.value} model predicts `{target}` at "
            f"{_num(best.primary_score, 4)} {state.primary_metric} on data held out "
            "from training."
        )
        if verdict:
            headline += (
                f" The evaluation gate graded it {verdict.overall_grade} and "
                + (
                    "cleared it for use."
                    if verdict.acceptable
                    else "did NOT clear it for deployment."
                )
            )
    else:
        headline = "No model completed training in this run, so there is no predictive result to report."

    summary = (
        f"{MARK} {headline}\n\n"
        + (
            f"The analysis started from {profile.n_rows:,} rows across "
            f"{profile.n_columns:,} columns, with "
            f"{_pct(profile.missing_cell_fraction)} of cells missing and "
            f"{profile.n_duplicate_rows:,} duplicate rows. "
            if profile
            else ""
        )
        + "Cleaning strategy was chosen per column from its measured distribution, "
        "features were engineered against measured structure, and candidate models "
        "were compared on identical splits against a no-skill baseline so the "
        "leaderboard means something.\n\n"
        + _note_of(state)
    )

    return FinalReport(
        title=f"{task.title()} of {target}",
        subtitle=(
            f"{profile.n_rows:,} rows x {profile.n_columns:,} columns"
            if profile
            else "AutoML Architect run"
        ),
        executive_summary=summary,
        sections=[],
        deployment=_deployment(state),
        appendix_notes=[
            "Generated in offline mode: no model calls were made and no API "
            "credentials were used. Every number in this report was measured by "
            "deterministic code; the prose around them was assembled by rules.",
            f"Run id: {state.run_id}.",
        ],
    )


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------


def _qa(state: RunState, agent: Any = None) -> QuestionAnswer:
    question = str(state.extras.get("question", "")).strip()
    log = state.experiments
    verdict = state.evaluation
    explain = state.explainability

    evidence: list[str] = []
    parts: list[str] = []

    best = None
    if log:
        best = next((r for r in log.results if r.experiment_id == log.best_experiment_id), None)
    if best is not None and best.primary_score is not None:
        parts.append(
            f"The winning model was {best.family.value}, scoring "
            f"{_num(best.primary_score, 4)} on {state.primary_metric}."
        )
        evidence.append(
            f"Leaderboard winner: {best.family.value} at "
            f"{_num(best.primary_score, 4)} {state.primary_metric}."
        )
    if verdict:
        parts.append(
            f"Evaluation graded the run {verdict.overall_grade} and recommended "
            f"'{verdict.recommended_action}'."
        )
        evidence.append(f"Evaluation verdict: {verdict.verdict_rationale}")
    if explain and (explain.global_attributions or explain.permutation_importance):
        attributions = explain.global_attributions or explain.permutation_importance
        parts.append(
            "The strongest drivers were "
            + ", ".join(
                f"`{a.feature}` ({_pct(a.importance, 0)})" for a in attributions[:3]
            )
            + "."
        )
        evidence.extend(
            f"{a.method} importance: {a.feature} = {_num(a.importance)}"
            for a in attributions[:3]
        )
    if state.cleaning:
        evidence.append(f"Cleaning: {state.cleaning.summary}")
    if state.warnings:
        evidence.append(f"{len(state.warnings)} warning(s) recorded during the run.")

    answer = (
        f"{MARK} Offline mode cannot interpret the question, so this is a summary "
        "of the recorded run state rather than a targeted answer.\n\n"
        + (" ".join(parts) if parts else "No results were recorded for this run.")
    )

    return QuestionAnswer(
        question=question or "(no question supplied)",
        answer=answer,
        evidence=evidence,
        confidence="low",
        caveats=[
            "Offline mode answers with a fixed run summary; it does not parse the "
            "question. Re-run with credentials for a real answer.",
        ],
        suggested_followups=[
            "Which features drove the prediction?",
            "Why was this model selected over the alternatives?",
            "What would improve the score most?",
        ],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_DECIDERS: dict[str, Callable[[RunState, Any], BaseModel]] = {
    "DatasetUnderstanding": _understanding,
    "ProblemDefinition": _problem,
    "ExecutionPlan": _plan,
    "CleaningPlan": _cleaning,
    "FeaturePlan": _features,
    "ModelSelection": _model_selection,
    "ExperimentLog": _experiments,
    "TuningDecision": _tuning,
    "ExplainabilityReport": _explain,
    "EvaluationVerdict": _evaluation,
    "InsightReport": _insight,
    "VisualizationPlan": _visualization,
    "FinalReport": _report,
    "QuestionAnswer": _qa,
}


def supports(output_model: type[BaseModel]) -> bool:
    """Whether a deterministic decider exists for this output type."""
    return output_model.__name__ in _DECIDERS


def decide(
    output_model: type[BaseModel], state: RunState, agent: Any = None
) -> BaseModel:
    """Produce ``output_model`` from measured state, with no model call.

    Args:
        output_model: The schema the agent would have asked Claude for.
        state: The run blackboard, already carrying the profile and whatever the
            execution layer has computed for this step.
        agent: The calling agent, when available. Hybrid agents stash their
            computed measurements on ``_computed``, which the deciders read.

    Returns:
        A validated instance of ``output_model``.

    Raises:
        KeyError: If no decider is registered for the model.
    """
    try:
        decider = _DECIDERS[output_model.__name__]
    except KeyError as exc:
        raise KeyError(
            f"offline mode has no rule engine for {output_model.__name__}; "
            "register one in automl_architect.core.offline"
        ) from exc
    value = decider(state, agent)
    logger.debug("offline decision produced %s", output_model.__name__)
    return value


__all__ = ["MARK", "decide", "supports"]
