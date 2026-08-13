"""Dataset memory: recognise a dataset you have seen before, and say what worked.

The premise is that a data scientist who has already modelled fifty tables does
not start from scratch on the fifty-first. Neither should the planner. After each
run we store a :class:`DatasetFingerprint` — a small structural signature plus
the outcome — and before planning we score the new dataset against that history.

Similarity is a weighted blend of seven signals, each normalised to ``[0, 1]``:

============  ======  ====================================================
signal        weight  what it captures
============  ======  ====================================================
task_type      0.24   a churn model is not evidence about a forecast
shape          0.18   log row and column counts; 10k x 20 != 10M x 800
column_names   0.18   Jaccard overlap; the strongest signal for "same table"
type_mix       0.16   numeric/categorical/datetime/text proportions
target_kind    0.10   continuous target vs 3-class label
missingness    0.08   a 40%-missing table needs a different plan
imbalance      0.06   99:1 changes metric and sampling choices
============  ======  ====================================================

Signals that either side cannot supply are dropped and the remaining weights
renormalised, so a fingerprint recorded before a target was known still matches
on structure instead of being penalised for a missing field.

Nothing here calls a model. The suggestion object is *evidence* handed to the
Planning Agent, which remains free to argue against it.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING

from ..config import Settings, get_settings
from ..core.schemas import (
    ColumnKind,
    DatasetFingerprint,
    DatasetProfile,
    ExperimentLog,
    FeatureOp,
    FeaturePlan,
    MemorySuggestion,
    ModelFamily,
    ProblemDefinition,
    RunSummary,
    SimilarRun,
    TaskType,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import RunState
    from .repository import RunRepository

logger = logging.getLogger(__name__)

#: Component weights. Must sum to 1.0; the blend renormalises over whatever is
#: actually comparable, so the absolute values only matter relative to each other.
WEIGHTS: dict[str, float] = {
    "task_type": 0.24,
    "shape": 0.18,
    "column_names": 0.18,
    "type_mix": 0.16,
    "target_kind": 0.10,
    "missingness": 0.08,
    "imbalance": 0.06,
}

#: Below this, "similar" is not a claim worth making to the planner.
MIN_SIMILARITY = 0.45

#: How many similar runs get their full summary loaded for cautions.
DEEP_INSPECT_LIMIT = 3

_NUMERIC_KINDS = {ColumnKind.NUMERIC_CONTINUOUS, ColumnKind.NUMERIC_DISCRETE}
_CATEGORICAL_KINDS = {
    ColumnKind.CATEGORICAL_NOMINAL,
    ColumnKind.CATEGORICAL_ORDINAL,
    ColumnKind.BOOLEAN,
}
_CLASSIFICATION_TASKS = {
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
}
_CONTINUOUS_TASKS = {TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING}
_UNSUPERVISED_TASKS = {TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION}

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


@lru_cache(maxsize=100_000)
def _normalise_column_name(name: str) -> str:
    """Fold a column name so ``Customer ID``, ``customer_id`` and ``CustomerId`` match.

    Cached because scoring one dataset against a few hundred stored fingerprints
    re-normalises the same names thousands of times.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return "_".join(part for part in _TOKEN_SPLIT.split(spaced.lower()) if part)


def build_fingerprint(
    profile: DatasetProfile,
    problem: ProblemDefinition | None = None,
    *,
    run_id: str,
    project: str = "default",
    experiments: ExperimentLog | None = None,
    features: FeaturePlan | None = None,
    primary_metric: str = "",
) -> DatasetFingerprint:
    """Build a structural signature from a measured profile.

    Args:
        profile: The deterministic dataset profile.
        problem: Task definition, when it has already been decided.
        run_id: Run this fingerprint belongs to.
        project: Project namespace.
        experiments: Completed experiments, to record what actually won.
        features: The feature plan, to record which operations were applied.
        primary_metric: Metric the best score is expressed in.

    Returns:
        A :class:`DatasetFingerprint` ready to persist.
    """
    counts = {"numeric": 0, "categorical": 0, "datetime": 0, "text": 0}
    for column in profile.columns:
        if column.kind in _NUMERIC_KINDS:
            counts["numeric"] += 1
        elif column.kind in _CATEGORICAL_KINDS:
            counts["categorical"] += 1
        elif column.kind is ColumnKind.DATETIME:
            counts["datetime"] += 1
        elif column.kind is ColumnKind.TEXT:
            counts["text"] += 1

    best = experiments.best() if experiments else None
    metric = primary_metric or (experiments.primary_metric if experiments else "")
    if not metric and problem:
        metric = problem.primary_metric

    return DatasetFingerprint(
        run_id=run_id,
        project=project,
        n_rows=profile.n_rows,
        n_columns=profile.n_columns,
        n_numeric=counts["numeric"],
        n_categorical=counts["categorical"],
        n_datetime=counts["datetime"],
        n_text=counts["text"],
        missing_fraction=profile.missing_cell_fraction,
        duplicate_fraction=profile.duplicate_fraction,
        task_type=problem.task_type if problem else None,
        target_kind=profile.target.kind if profile.target else None,
        imbalance_ratio=profile.target.imbalance_ratio if profile.target else None,
        column_names=[column.name for column in profile.columns],
        primary_metric=metric,
        best_score=best.primary_score if best else None,
        best_family=best.family if best else None,
        winning_feature_ops=_winning_feature_ops(features),
    )


def fingerprint_from_summary(summary: RunSummary) -> DatasetFingerprint | None:
    """Build a fingerprint from a finished run, or ``None`` if it never profiled."""
    if summary.profile is None:
        return None
    return build_fingerprint(
        summary.profile,
        summary.problem,
        run_id=summary.run_id,
        project=summary.project,
        experiments=summary.experiments,
        features=summary.features,
        primary_metric=(
            summary.experiments.primary_metric if summary.experiments else ""
        ),
    )


def _winning_feature_ops(features: FeaturePlan | None) -> list[str]:
    if features is None:
        return []
    seen: list[str] = []
    for decision in features.decisions:
        value = decision.op.value
        if value not in seen:
            seen.append(value)
    return seen


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _log_closeness(left: float, right: float, tolerance: float) -> float:
    """Similarity of two magnitudes on a log scale.

    ``tolerance`` is the number of orders of magnitude at which similarity hits
    zero, which is what makes 8k-vs-12k rows near-identical while 8k-vs-8M is not.
    """
    gap = abs(math.log10(max(left, 0.0) + 1.0) - math.log10(max(right, 0.0) + 1.0))
    return max(0.0, 1.0 - gap / tolerance)


def _shape_similarity(a: DatasetFingerprint, b: DatasetFingerprint) -> float:
    rows = _log_closeness(a.n_rows, b.n_rows, tolerance=2.0)
    columns = _log_closeness(a.n_columns, b.n_columns, tolerance=1.2)
    return 0.5 * rows + 0.5 * columns


def _type_mix_similarity(a: DatasetFingerprint, b: DatasetFingerprint) -> float | None:
    def mix(fp: DatasetFingerprint) -> list[float] | None:
        total = fp.n_numeric + fp.n_categorical + fp.n_datetime + fp.n_text
        if total <= 0:
            return None
        return [
            fp.n_numeric / total,
            fp.n_categorical / total,
            fp.n_datetime / total,
            fp.n_text / total,
        ]

    left, right = mix(a), mix(b)
    if left is None or right is None:
        return None
    # Total-variation distance between the two proportion vectors.
    distance = 0.5 * sum(abs(x - y) for x, y in zip(left, right, strict=True))
    return 1.0 - distance


def _missingness_similarity(a: DatasetFingerprint, b: DatasetFingerprint) -> float:
    return max(0.0, 1.0 - abs(a.missing_fraction - b.missing_fraction) / 0.4)


def _task_similarity(a: DatasetFingerprint, b: DatasetFingerprint) -> float | None:
    if a.task_type is None or b.task_type is None:
        return None
    if a.task_type is b.task_type:
        return 1.0
    for group in (_CLASSIFICATION_TASKS, _CONTINUOUS_TASKS, _UNSUPERVISED_TASKS):
        if a.task_type in group and b.task_type in group:
            return 0.6
    return 0.0


def _target_kind_similarity(a: DatasetFingerprint, b: DatasetFingerprint) -> float | None:
    if a.target_kind is None or b.target_kind is None:
        return None
    if a.target_kind is b.target_kind:
        return 1.0
    for group in (_NUMERIC_KINDS, _CATEGORICAL_KINDS):
        if a.target_kind in group and b.target_kind in group:
            return 0.7
    return 0.0


def _imbalance_similarity(a: DatasetFingerprint, b: DatasetFingerprint) -> float | None:
    if a.imbalance_ratio is None or b.imbalance_ratio is None:
        return None
    return _log_closeness(a.imbalance_ratio, b.imbalance_ratio, tolerance=1.0)


def _name_overlap(a: DatasetFingerprint, b: DatasetFingerprint) -> tuple[float | None, int]:
    left = {_normalise_column_name(name) for name in a.column_names}
    right = {_normalise_column_name(name) for name in b.column_names}
    left.discard("")
    right.discard("")
    if not left or not right:
        return None, 0
    shared = left & right
    union = left | right
    return len(shared) / len(union), len(shared)


def similarity_components(
    a: DatasetFingerprint, b: DatasetFingerprint
) -> dict[str, float]:
    """Per-signal similarity scores, omitting signals neither side can supply.

    Exposed separately from :func:`score_similarity` so the blend is inspectable
    — a similarity number nobody can decompose is a number nobody will trust.
    """
    jaccard, _ = _name_overlap(a, b)
    raw: dict[str, float | None] = {
        "task_type": _task_similarity(a, b),
        "shape": _shape_similarity(a, b),
        "column_names": jaccard,
        "type_mix": _type_mix_similarity(a, b),
        "target_kind": _target_kind_similarity(a, b),
        "missingness": _missingness_similarity(a, b),
        "imbalance": _imbalance_similarity(a, b),
    }
    return {
        key: max(0.0, min(1.0, value))
        for key, value in raw.items()
        if value is not None
    }


def score_similarity(
    a: DatasetFingerprint, b: DatasetFingerprint
) -> tuple[float, str]:
    """Score two fingerprints and explain the score in one sentence.

    Args:
        a: The dataset being planned.
        b: A stored fingerprint from a past run.

    Returns:
        ``(similarity, why_similar)`` with similarity in ``[0, 1]``.
    """
    components = similarity_components(a, b)
    if not components:
        return 0.0, "no comparable structural signals were recorded"

    available = sum(WEIGHTS[key] for key in components)
    score = sum(WEIGHTS[key] * value for key, value in components.items()) / available
    return score, _explain(a, b, components)


def _explain(
    a: DatasetFingerprint, b: DatasetFingerprint, components: dict[str, float]
) -> str:
    """Name the two or three signals that actually drove the score."""
    ranked = sorted(
        components.items(),
        key=lambda item: WEIGHTS[item[0]] * item[1],
        reverse=True,
    )
    _, shared = _name_overlap(a, b)
    phrases: list[str] = []
    for key, value in ranked[:3]:
        if value < 0.35:
            continue
        if key == "task_type":
            same = a.task_type is b.task_type
            label = b.task_type.value if b.task_type else "unknown"
            phrases.append(f"same task ({label})" if same else f"related task ({label})")
        elif key == "shape":
            phrases.append(
                f"comparable shape ({a.n_rows:,}x{a.n_columns} vs {b.n_rows:,}x{b.n_columns})"
            )
        elif key == "column_names":
            phrases.append(f"{value:.0%} column-name overlap ({shared} shared names)")
        elif key == "type_mix":
            phrases.append(f"similar column-type mix ({value:.0%} agreement)")
        elif key == "target_kind":
            label = b.target_kind.value if b.target_kind else "unknown"
            phrases.append(f"target of the same kind ({label})")
        elif key == "missingness":
            phrases.append(
                f"similar missingness ({a.missing_fraction:.1%} vs {b.missing_fraction:.1%})"
            )
        elif key == "imbalance":
            phrases.append("similar class imbalance")
    if not phrases:
        phrases.append("weak structural resemblance only")
    return "; ".join(phrases)


def rank_fingerprints(
    target: DatasetFingerprint,
    candidates: list[DatasetFingerprint],
    *,
    limit: int = 5,
    minimum: float = MIN_SIMILARITY,
) -> list[tuple[SimilarRun, DatasetFingerprint]]:
    """Score candidates against ``target`` and return the best matches.

    Args:
        target: Fingerprint of the dataset being planned.
        candidates: Stored fingerprints to compare against.
        limit: Maximum matches returned.
        minimum: Similarity floor; matches below it are discarded.

    Returns:
        ``(SimilarRun, fingerprint)`` pairs, most similar first.
    """
    scored: list[tuple[SimilarRun, DatasetFingerprint]] = []
    for candidate in candidates:
        if candidate.run_id == target.run_id:
            continue
        similarity, why = score_similarity(target, candidate)
        if similarity < minimum:
            continue
        scored.append(
            (
                SimilarRun(
                    run_id=candidate.run_id,
                    similarity=round(similarity, 4),
                    task_type=candidate.task_type,
                    best_family=candidate.best_family,
                    best_score=candidate.best_score,
                    primary_metric=candidate.primary_metric,
                    why_similar=why,
                ),
                candidate,
            )
        )
    scored.sort(key=lambda item: item[0].similarity, reverse=True)
    return scored[: max(1, limit)]


# ---------------------------------------------------------------------------
# The memory itself
# ---------------------------------------------------------------------------


class DatasetMemory:
    """Records dataset fingerprints and answers "have we seen this before?".

    Args:
        repository: Storage to read and write. Constructed from settings if omitted.
        settings: Settings override.
    """

    def __init__(
        self,
        repository: RunRepository | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if repository is None:
            from .repository import RunRepository as _RunRepository

            repository = _RunRepository(settings=self.settings)
        self.repository = repository

    # -- writing -----------------------------------------------------------

    def remember(self, summary: RunSummary) -> DatasetFingerprint | None:
        """Persist a fingerprint for a finished run.

        Returns ``None`` (and logs) when the run has no profile to fingerprint —
        a run that failed during ingestion teaches nothing about datasets.
        """
        fingerprint = fingerprint_from_summary(summary)
        if fingerprint is None:
            logger.debug("run %s has no profile; nothing to remember", summary.run_id)
            return None
        try:
            self.repository.save_fingerprint(fingerprint)
        except Exception:  # memory is a nice-to-have, never a failure path
            logger.exception("could not persist fingerprint for run %s", summary.run_id)
            return None
        return fingerprint

    def remember_state(self, state: RunState) -> DatasetFingerprint | None:
        """Convenience wrapper: fingerprint a live :class:`RunState`."""
        return self.remember(state.to_summary())

    # -- reading -----------------------------------------------------------

    def fingerprint_for(
        self,
        profile: DatasetProfile,
        problem: ProblemDefinition | None = None,
        *,
        run_id: str,
        project: str = "default",
    ) -> DatasetFingerprint:
        """Fingerprint a dataset that has not finished modelling yet."""
        return build_fingerprint(
            profile, problem, run_id=run_id, project=project
        )

    def lookup(
        self,
        profile: DatasetProfile,
        problem: ProblemDefinition | None = None,
        *,
        run_id: str,
        project: str = "default",
        limit: int = 5,
    ) -> MemorySuggestion:
        """Find precedent for a dataset and turn it into planner-ready advice."""
        fingerprint = self.fingerprint_for(
            profile, problem, run_id=run_id, project=project
        )
        return self.suggest(fingerprint, limit=limit)

    def lookup_for_state(self, state: RunState, *, limit: int = 5) -> MemorySuggestion:
        """Look up precedent for a live run. Degrades to "no precedent" if unprofiled."""
        if state.profile is None:
            return _no_precedent("the dataset has not been profiled yet")
        return self.lookup(
            state.profile,
            state.problem,
            run_id=state.run_id,
            project=state.config.project,
            limit=limit,
        )

    def suggest(
        self, fingerprint: DatasetFingerprint, *, limit: int = 5
    ) -> MemorySuggestion:
        """Assemble a :class:`MemorySuggestion` from stored history.

        Args:
            fingerprint: The dataset being planned.
            limit: Maximum similar runs to report.

        Returns:
            A suggestion with ``has_precedent=False`` and an explanatory
            narrative when history is empty or nothing clears the similarity
            floor. This method never raises on a storage failure.
        """
        try:
            candidates = self.repository.list_fingerprints(
                exclude_run_id=fingerprint.run_id
            )
        except Exception:
            logger.exception("dataset-memory lookup failed; continuing without precedent")
            return _no_precedent("the run history could not be read")

        if not candidates:
            return _no_precedent(
                "this is the first run recorded in this workspace, so there is no "
                "precedent to draw on"
            )

        matches = rank_fingerprints(fingerprint, candidates, limit=limit)
        if not matches:
            return _no_precedent(
                f"{len(candidates)} past run(s) were compared, but none scored above "
                f"{MIN_SIMILARITY:.0%} structural similarity — treat this dataset as new"
            )

        similar = [item for item, _ in matches]
        families = self._rank_families(matches)
        feature_ops = self._rank_feature_ops(matches)
        cautions = self._collect_cautions(matches)

        return MemorySuggestion(
            has_precedent=True,
            similar_runs=similar,
            recommended_families=families,
            recommended_feature_ops=feature_ops,
            cautions=cautions,
            narrative=_narrative(fingerprint, similar, families, feature_ops),
        )

    # -- aggregation -------------------------------------------------------

    @staticmethod
    def _rank_families(
        matches: list[tuple[SimilarRun, DatasetFingerprint]],
    ) -> list[ModelFamily]:
        """Families that won on similar data, weighted by how similar it was."""
        weights: dict[ModelFamily, float] = defaultdict(float)
        for similar, fingerprint in matches:
            if fingerprint.best_family is None:
                continue
            weights[fingerprint.best_family] += similar.similarity
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        return [family for family, _ in ranked[:4]]

    @staticmethod
    def _rank_feature_ops(
        matches: list[tuple[SimilarRun, DatasetFingerprint]],
    ) -> list[FeatureOp]:
        """Feature operations used on similar data, weighted by similarity."""
        weights: dict[FeatureOp, float] = defaultdict(float)
        for similar, fingerprint in matches:
            for name in fingerprint.winning_feature_ops:
                try:
                    op = FeatureOp(name)
                except ValueError:  # an op removed from the enum since it was stored
                    continue
                weights[op] += similar.similarity
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        return [op for op, _ in ranked[:8]]

    def _collect_cautions(
        self, matches: list[tuple[SimilarRun, DatasetFingerprint]]
    ) -> list[str]:
        """What went wrong last time, read from the stored run summaries."""
        cautions: list[str] = []
        for similar, _ in matches[:DEEP_INSPECT_LIMIT]:
            try:
                summary = self.repository.get_run(similar.run_id)
            except Exception:
                logger.debug("could not load run %s for cautions", similar.run_id)
                continue
            if summary is None:
                continue
            label = similar.run_id

            if summary.experiments:
                failed = sorted(
                    {r.family.value for r in summary.experiments.results if r.failed}
                )
                if failed:
                    cautions.append(
                        f"on similar run {label}, these families failed to fit: "
                        f"{', '.join(failed)}"
                    )
            if summary.evaluation:
                for weakness in summary.evaluation.weaknesses[:2]:
                    cautions.append(f"{label} noted: {weakness}")
                if summary.evaluation.recommended_action != "accept":
                    cautions.append(
                        f"{label} was not accepted on the first pass "
                        f"(action: {summary.evaluation.recommended_action})"
                    )
            if (
                summary.tuning
                and summary.tuning.ran
                and summary.tuning.improvement is not None
                and summary.tuning.improvement <= 0
            ):
                cautions.append(
                    f"tuning did not improve the score on {label}; budget it cautiously"
                )
        return cautions[:6]


def _no_precedent(reason: str) -> MemorySuggestion:
    """A well-formed "nothing to go on" suggestion."""
    return MemorySuggestion(
        has_precedent=False,
        similar_runs=[],
        recommended_families=[],
        recommended_feature_ops=[],
        cautions=[],
        narrative=(
            f"No usable precedent: {reason}. Plan this dataset on its measured "
            "properties alone."
        ),
    )


def _narrative(
    fingerprint: DatasetFingerprint,
    similar: list[SimilarRun],
    families: list[ModelFamily],
    feature_ops: list[FeatureOp],
) -> str:
    """A paragraph the Planning Agent can reason over."""
    best = similar[0]
    lines = [
        f"Found {len(similar)} structurally similar past run(s). The closest is "
        f"{best.run_id} at {best.similarity:.0%} similarity ({best.why_similar})."
    ]
    scored = [item for item in similar if item.best_score is not None]
    if scored:
        rendered = ", ".join(
            f"{item.run_id} scored {item.best_score:.4g} "
            f"{item.primary_metric or 'on its primary metric'}"
            f"{f' with {item.best_family.value}' if item.best_family else ''}"
            for item in scored[:3]
        )
        lines.append(f"Outcomes on those runs: {rendered}.")
    if families:
        lines.append(
            "Model families that won on similar data, most-supported first: "
            + ", ".join(family.value for family in families)
            + "."
        )
    if feature_ops:
        lines.append(
            "Feature operations those runs applied: "
            + ", ".join(op.value for op in feature_ops)
            + "."
        )
    lines.append(
        "This is precedent, not instruction — the current dataset "
        f"({fingerprint.n_rows:,} rows x {fingerprint.n_columns} columns) still "
        "governs the plan."
    )
    return " ".join(lines)


__all__ = [
    "DEEP_INSPECT_LIMIT",
    "MIN_SIMILARITY",
    "WEIGHTS",
    "DatasetMemory",
    "build_fingerprint",
    "fingerprint_from_summary",
    "rank_fingerprints",
    "score_similarity",
    "similarity_components",
]
