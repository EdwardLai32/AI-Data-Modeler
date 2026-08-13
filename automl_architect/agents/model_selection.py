"""Model Selection Agent.

The failure mode this module exists to prevent is the reflex: reach for gradient
boosting, add a random forest for contrast, ship. That reflex produces a model
that is sometimes right and an argument that is never checkable. The prompt below
forces a comparative case grounded in this dataset's n, p, n/p ratio, feature
mix, class balance, missingness, and explainability requirement — and it forbids
including a family "for completeness" without saying why it belongs.

It also mandates a trivial baseline. A model that cannot beat the majority class
or the mean has learned nothing, and without that comparison in the leaderboard
an ROC-AUC of 0.71 reads as competence rather than as whatever it actually is.

:meth:`ModelSelectionAgent.postprocess` reconciles the agent's choices with the
environment: families whose library is not installed are dropped with a warning
rather than exploding inside the trainer, a baseline is injected if the agent
forgot one, and the candidate list is truncated to the operator's experiment
budget with ranks renumbered densely.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..core.agent import BaseAgent
from ..core.schemas import (
    AgentName,
    ColumnKind,
    ModelCandidate,
    ModelFamily,
    ModelSelection,
    TaskType,
)
from ..core.state import RunState

logger = logging.getLogger(__name__)

#: Prompt-side fallback used only when ``execution.model_zoo`` cannot be
#: imported. Keeping the model zoo authoritative matters, but an agent told
#: nothing about legality will happily propose a regressor for a classification
#: task, so a coarse map is better than silence.
_TASK_FAMILIES: dict[TaskType, tuple[ModelFamily, ...]] = {
    TaskType.BINARY_CLASSIFICATION: (
        ModelFamily.LOGISTIC,
        ModelFamily.RIDGE,
        ModelFamily.DECISION_TREE,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.EXTRA_TREES,
        ModelFamily.GRADIENT_BOOSTING,
        ModelFamily.HIST_GRADIENT_BOOSTING,
        ModelFamily.XGBOOST,
        ModelFamily.LIGHTGBM,
        ModelFamily.SVM,
        ModelFamily.KNN,
        ModelFamily.NAIVE_BAYES,
        ModelFamily.NEURAL_NETWORK,
        ModelFamily.BASELINE_DUMMY,
    ),
    TaskType.REGRESSION: (
        ModelFamily.LINEAR,
        ModelFamily.RIDGE,
        ModelFamily.LASSO,
        ModelFamily.ELASTIC_NET,
        ModelFamily.DECISION_TREE,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.EXTRA_TREES,
        ModelFamily.GRADIENT_BOOSTING,
        ModelFamily.HIST_GRADIENT_BOOSTING,
        ModelFamily.XGBOOST,
        ModelFamily.LIGHTGBM,
        ModelFamily.SVM,
        ModelFamily.KNN,
        ModelFamily.NEURAL_NETWORK,
        ModelFamily.BASELINE_DUMMY,
    ),
    TaskType.TIME_SERIES_FORECASTING: (
        ModelFamily.SEASONAL_NAIVE,
        ModelFamily.THETA,
        ModelFamily.EXPONENTIAL_SMOOTHING,
        ModelFamily.SARIMAX,
        ModelFamily.RIDGE,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.HIST_GRADIENT_BOOSTING,
        ModelFamily.LIGHTGBM,
        ModelFamily.BASELINE_DUMMY,
    ),
    TaskType.CLUSTERING: (
        ModelFamily.KMEANS,
        ModelFamily.DBSCAN,
        ModelFamily.GAUSSIAN_MIXTURE,
    ),
    TaskType.ANOMALY_DETECTION: (
        ModelFamily.ISOLATION_FOREST,
        ModelFamily.LOCAL_OUTLIER_FACTOR,
        ModelFamily.ONE_CLASS_SVM,
    ),
}
_TASK_FAMILIES[TaskType.MULTICLASS_CLASSIFICATION] = _TASK_FAMILIES[
    TaskType.BINARY_CLASSIFICATION
]

#: Families whose backing library is optional in this deployment. Named in the
#: prompt so the agent does not spend its argument on something unbuildable.
_OPTIONAL_LIBRARIES = {
    ModelFamily.CATBOOST: "catboost",
    ModelFamily.XGBOOST: "xgboost",
    ModelFamily.LIGHTGBM: "lightgbm",
    ModelFamily.SARIMAX: "statsmodels",
    ModelFamily.THETA: "statsmodels",
    ModelFamily.EXPONENTIAL_SMOOTHING: "statsmodels",
}


def _num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        if value != value:  # NaN
            return "n/a"
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.{digits}g}"


def _pct(fraction: float | None) -> str:
    if fraction is None:
        return "n/a"
    return f"{fraction * 100:.2f}%"


def _availability() -> dict[ModelFamily, bool] | None:
    """Ask the model zoo which families can actually be built here.

    Returns ``None`` when the executor module is unavailable, so callers can
    fall back rather than treat "unknown" as "unavailable".
    """
    try:
        from ..execution.model_zoo import is_available
    except Exception as exc:  # noqa: BLE001 - optional at prompt-build time
        logger.debug("model_zoo unavailable for availability probe: %s", exc)
        return None

    result: dict[ModelFamily, bool] = {}
    for family in ModelFamily:
        try:
            result[family] = bool(is_available(family))
        except Exception as exc:  # noqa: BLE001 - one bad probe must not stop the rest
            logger.debug("is_available(%s) raised: %s", family.value, exc)
            result[family] = True
    return result


def _legal_families(task: TaskType | None) -> list[ModelFamily]:
    """Families the executor can train for this task, best effort."""
    if task is None:
        return []
    try:
        from ..execution.model_zoo import available_families

        families = list(available_families(task))
        if families:
            return families
    except Exception as exc:  # noqa: BLE001 - fall back to the coarse map
        logger.debug("available_families(%s) unavailable: %s", task.value, exc)
    return list(_TASK_FAMILIES.get(task, ()))


def _feature_shape(state: RunState) -> tuple[int, int]:
    """Rows and feature-column count as they will be at training time."""
    n_rows = 0
    n_features = 0

    frame = state.feature_frame if state.feature_frame is not None else state.working_df
    if frame is None:
        frame = state.raw_df
    if frame is not None:
        try:
            n_rows = int(len(frame))
            n_features = int(frame.shape[1])
        except (TypeError, AttributeError):  # not a dataframe
            n_rows = n_features = 0

    if state.feature_names:
        n_features = len(state.feature_names)
    elif n_features and state.target:
        n_features = max(n_features - 1, 0)

    if not n_rows and state.profile:
        n_rows = state.profile.n_rows
    if not n_features and state.profile:
        n_features = max(state.profile.n_columns - (1 if state.target else 0), 0)
    return n_rows, n_features


class ModelSelectionAgent(BaseAgent[ModelSelection]):
    """Chooses and ranks model families with a comparative, dataset-grounded case."""

    name = AgentName.MODEL_SELECTION
    title = "Model Selection Agent"
    output_model = ModelSelection
    effort = "xhigh"
    max_tokens = 16_000

    # -- prompts -----------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        """The comparative-selection method, adapted to this run's shape."""
        task = state.task_type
        n_rows, n_features = _feature_shape(state)
        ratio = n_rows / n_features if n_features else 0.0

        situational: list[str] = []
        if n_features and ratio < 10:
            situational.append(
                f"There are only about {ratio:.1f} rows per feature "
                f"(n={n_rows:,}, p={n_features}). This is the regime where "
                "unregularised models memorise. Regularised linear models (ridge, "
                "lasso, elastic net) and shallow, heavily-constrained trees have the "
                "advantage; deep boosted ensembles will fit the noise. Argue from this "
                "number explicitly."
            )
        elif n_features and ratio > 500:
            situational.append(
                f"There are roughly {ratio:.0f} rows per feature (n={n_rows:,}, "
                f"p={n_features}). Data is plentiful relative to width, so boosted "
                "ensembles have room to earn their capacity — but that is an argument "
                "you still have to make against a linear baseline, not a default."
            )
        if n_rows and n_rows < 1_000:
            situational.append(
                f"With {n_rows:,} rows, cross-validated estimates will be noisy and "
                "any difference between two decent models is likely inside the error "
                "bars. Prefer simple, stable, interpretable families and say that the "
                "sample size — not the algorithm — is the binding constraint."
            )
        if n_rows and n_rows > 200_000:
            situational.append(
                f"With {n_rows:,} rows, per-model fit time matters against the run's "
                "time budget. SVM with a non-linear kernel is roughly quadratic in n "
                "and KNN pays at prediction time; exclude them on cost and say so "
                "rather than proposing them and hoping."
            )
        if task and task.is_classification and state.profile and state.profile.target:
            summary = state.profile.target
            if summary.is_imbalanced:
                situational.append(
                    f"The target is imbalanced (majority/minority "
                    f"{_num(summary.imbalance_ratio, 3)}). Validation must be "
                    "stratified or the minority class will be unevenly distributed "
                    "across folds. Consider class weighting in `initial_params` and "
                    "note that accuracy is not a usable comparison here."
                )
        if task is TaskType.TIME_SERIES_FORECASTING:
            situational.append(
                "This is forecasting. Random k-fold is invalid: it trains on the "
                "future to predict the past. Choose an expanding-window (or rolling-"
                "origin) time-series split and say so in `validation_rationale`. A "
                "seasonal-naive forecast is the honest baseline here, not just a dummy."
            )
        if state.problem and state.problem.group_column:
            situational.append(
                f"A group key exists (`{state.problem.group_column}`). If the same "
                "entity can appear in several rows, a random split puts the same entity "
                "in train and test and the score measures memorisation of that entity "
                "rather than generalisation. Choose a grouped split and justify it."
            )

        situational_block = "\n\n".join(f"- {item}" for item in situational)

        return f"""\
You are the model selection specialist. Your output is judged on the *argument*,
not on the list. A ranked set of families with no comparative case is worth less
than three families where you can say why the first beats the second on this
data.

## The one thing you must not do

Do not default to gradient boosting. It is a strong general performer and it is
also the reflex answer, which means proposing it costs you nothing and tells the
reader nothing. If boosted trees are right here, prove it from the row count,
the feature mix, and the interaction structure — and name the specific model it
should beat.

## The comparative argument, made from measured quantities

Reason from n (rows), p (features), the n/p ratio, the feature-type mix, class
balance, residual missingness, and the explainability requirement:

-   **Linear and logistic regression** are the interpretable strong baseline, and
    they *win outright* when n is small, when p approaches n, or when the signal
    is close to linear — which the target correlations will show you. Coefficients
    map directly to business language, which for many stakeholders is worth more
    than a small accuracy gain. If you are not proposing one, say why not.
-   **Regularised linear (ridge, lasso, elastic net)** when p is large relative
    to n or when features are collinear. Lasso additionally selects, which is a
    feature-count answer as well as a model answer.
-   **Decision tree** rarely wins, but a shallow one is the most explainable
    model that exists. Propose it when a human needs to read the rules.
-   **Random forest / extra trees** for mixed types, non-linearities, and
    interactions, with low tuning sensitivity and honest out-of-the-box behaviour.
    They are the right answer surprisingly often on medium-sized tabular data.
-   **Boosted trees (hist gradient boosting, XGBoost, LightGBM)** when n is large
    enough that their capacity is supported rather than spent on noise, when
    interactions matter, and when residual missingness needs native handling —
    all three split on NaN directly, which can beat any imputation. Sensitive to
    hyperparameters, so say whether the time budget allows tuning.
-   **SVM** only at modest n: a non-linear kernel scales roughly quadratically,
    so it becomes untenable as rows grow. Requires scaled features.
-   **KNN** only when a distance in this feature space means something real and
    the features are scaled. High-cardinality one-hot columns and mixed types
    usually destroy that meaning — if so, exclude it and say why.
-   **Naive Bayes** for high-dimensional sparse count data (text), where its
    independence assumption is wrong but harmless. Elsewhere it is usually
    dominated.
-   **Neural networks** are rarely justified on tabular data at this scale; they
    need more rows than a typical table has, cost interpretability, and are
    routinely beaten by gradient boosting on exactly this kind of problem. Say
    that plainly in `exclusion_rationale` rather than including one for
    completeness.

## The mandatory baseline

You must always include `baseline_dummy` with `is_baseline` set true. A model
that cannot beat the majority class (classification) or the mean (regression)
has learned nothing, and the report needs that comparison in order to be honest
about what was achieved. Rank it last. Set its `tune_priority` to none.

## Validation strategy

Choose it from the data, and put the reason in `validation_rationale`:

-   Imbalanced classification: stratified k-fold, so every fold contains the
    minority class in proportion.
-   A group or entity key present, with repeated entities: grouped k-fold, so the
    same entity never spans train and test. Without this the score measures
    memorisation.
-   Temporal data or forecasting: expanding-window / rolling-origin split. Random
    folds train on the future.
-   Small n: more folds buys a less noisy estimate at proportional compute cost —
    say which side of that trade you chose.
-   Otherwise: plain k-fold, and say why nothing more elaborate is needed.

## Output discipline

-   Rank the candidates. Rank 1 is the model you would ship if you could only
    train one.
-   Populate `excluded_families` and `exclusion_rationale` as index-aligned
    lists: the family you rejected and the specific reason you rejected it. A
    rejection is as informative as a selection, and it proves you considered the
    option rather than forgetting it.
-   `reasoning` must be the comparative argument — "X over Y because …" — not a
    restatement of each candidate's rationale.
-   Put concrete starting hyperparameters in `initial_params` where the data
    shape implies them (a max_depth suited to n, class weights for imbalance, a
    learning rate matched to the number of estimators). Leave it empty when the
    library default is genuinely the right start.
-   Propose only families that appear in the legal list you are given. Do not
    invent a family name.
-   Fewer, better-argued candidates beat a long shortlist: the experiment budget
    is finite and every slot spent on a model you cannot argue for is a slot not
    spent on tuning one you can.
{situational_block}
"""

    def build_prompt(self, state: RunState) -> str:
        profile = state.profile
        n_rows, n_features = _feature_shape(state)
        lines: list[str] = []
        add = lines.append

        add("# MODEL SELECTION BRIEF")
        add("")
        add(
            "The measured dataset facts are already in your context. Below is the "
            "post-preparation shape — which is what the models will actually see — "
            "plus the environment's constraints."
        )
        add("")

        add("## Problem framing")
        problem = state.problem
        if problem:
            add(f"- task: {problem.task_type.value}")
            add(f"- target: `{problem.target_column or 'none (unsupervised)'}`")
            if problem.positive_class:
                add(f"- positive class: {problem.positive_class}")
            add(f"- primary metric: {problem.primary_metric}")
            add(f"- metric rationale: {problem.metric_rationale}")
            if problem.secondary_metrics:
                add(f"- secondary metrics: {', '.join(problem.secondary_metrics)}")
            if problem.temporal_column:
                add(f"- temporal column: `{problem.temporal_column}`")
            if problem.group_column:
                add(f"- group / entity key: `{problem.group_column}`")
            if problem.horizon:
                add(f"- forecast horizon: {problem.horizon} period(s)")
            add(f"- business objective: {problem.business_objective}")
            if problem.constraints:
                add("- constraints stated for this problem:")
                for constraint in problem.constraints:
                    add(f"  - {constraint}")
        else:
            add(
                f"- task: {state.task_type.value if state.task_type else 'not yet determined'}"
            )
            add(f"- target: `{state.target or 'unknown'}`")
            add(f"- primary metric: {state.primary_metric}")
        add("")

        add("## Shape the models will see")
        add(f"- training rows available: {n_rows:,}")
        add(f"- feature columns after preparation: {n_features:,}")
        if n_features:
            add(
                f"- n/p ratio: {n_rows / n_features:.1f} rows per feature "
                "(below ~10 favours regularisation; above ~100 supports capacity)"
            )
        sizes = state.splits.sizes()
        if any(sizes.values()):
            add(
                f"- split sizes already materialised: train={sizes['train']:,} "
                f"validation={sizes['validation']:,} test={sizes['test']:,}"
            )
            if state.splits.strategy:
                add(f"- split strategy in force: {state.splits.strategy}")
        add(f"- cross-validation folds configured: {state.config.cv_folds}")
        add(f"- experiment budget (max models to train): {state.config.max_experiments}")
        add(
            f"- time budget remaining for the whole run: "
            f"{state.time_remaining:.0f}s of {state.config.time_budget_seconds}s"
        )
        add(f"- hyperparameter tuning enabled: {str(state.config.enable_tuning).lower()}")
        add(
            f"- explainability required: {str(state.config.enable_explainability).lower()}"
        )
        add("")

        if profile is not None:
            kinds: dict[str, int] = {}
            for col in profile.columns:
                if col.name == state.target:
                    continue
                kinds[col.kind.value] = kinds.get(col.kind.value, 0) + 1
            add("## Feature-type mix (pre-encoding)")
            for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
                add(f"- {kind}: {count}")
            high_card = [
                c
                for c in profile.columns
                if c.kind
                in {ColumnKind.CATEGORICAL_NOMINAL, ColumnKind.CATEGORICAL_ORDINAL}
                and c.n_unique > 20
                and c.name != state.target
            ]
            if high_card:
                add(
                    "- high-cardinality categoricals (>20 levels): "
                    + ", ".join(f"`{c.name}`({c.n_unique})" for c in high_card[:10])
                )
            add(
                f"- residual missing cells across the table: "
                f"{_pct(profile.missing_cell_fraction)} "
                "(non-zero favours learners with native NaN handling)"
            )
            if profile.highly_correlated_pairs:
                add(
                    f"- collinear feature pairs detected: "
                    f"{len(profile.highly_correlated_pairs)} "
                    "(argues for regularised linear over plain OLS)"
                )
            add("")

            if profile.target:
                target_summary = profile.target
                add("## Target distribution")
                if target_summary.n_classes is not None:
                    add(f"- classes: {target_summary.n_classes}")
                if target_summary.class_counts:
                    add(
                        "- class shares: "
                        + ", ".join(
                            f"{c.value!r}={_pct(c.fraction)}"
                            for c in target_summary.class_counts[:10]
                        )
                    )
                if target_summary.imbalance_ratio is not None:
                    add(
                        f"- imbalance ratio: {_num(target_summary.imbalance_ratio, 3)} "
                        f"({'IMBALANCED' if target_summary.is_imbalanced else 'reasonably balanced'})"
                    )
                if target_summary.mean is not None:
                    add(
                        f"- mean={_num(target_summary.mean)} "
                        f"std={_num(target_summary.std)} "
                        f"skew={_num(target_summary.skewness, 3)}"
                    )
                add("")

            if profile.target_correlations:
                add("## Linearity evidence: strongest target correlations")
                add(
                    "Strong linear correlations argue that a linear model is competitive; "
                    "weak linear correlation with obvious domain structure argues for "
                    "trees finding interactions."
                )
                for pair in profile.target_correlations[:12]:
                    add(
                        f"- `{pair.left}` vs `{pair.right}`: "
                        f"{_num(pair.coefficient, 4)} ({pair.method})"
                    )
                add("")

        if state.features and state.features.decisions:
            add("## Feature engineering that was applied")
            add(f"- {state.features.summary}")
            if state.features.dimensionality_strategy:
                add(f"- dimensionality strategy: {state.features.dimensionality_strategy}")
            scaled = sorted(
                {
                    d.op.value
                    for d in state.features.decisions
                    if "scale" in d.op.value or d.op.value == "quantile_transform"
                }
            )
            add(
                f"- scaling applied: {', '.join(scaled) if scaled else 'none'} "
                "(scale-sensitive families need this)"
            )
            add("")

        legal = _legal_families(state.task_type)
        availability = _availability()
        add("## Families you may choose from")
        if legal:
            for family in legal:
                note = ""
                if availability is not None and not availability.get(family, True):
                    note = " — UNAVAILABLE in this environment, do not propose it"
                elif family in _OPTIONAL_LIBRARIES:
                    note = f" (backed by optional library '{_OPTIONAL_LIBRARIES[family]}')"
                add(f"- {family.value}{note}")
        else:
            add(
                "- the executor could not report a family list for this task; propose "
                "only families that plainly fit the task type"
            )
        if availability is not None:
            missing = [f.value for f in ModelFamily if not availability.get(f, True)]
            if missing:
                add(
                    f"- not installed in this deployment: {', '.join(missing)}. "
                    "Proposing one wastes an experiment slot; put it in "
                    "`excluded_families` instead if it would otherwise have been a "
                    "contender."
                )
        add("")

        if state.memory and state.memory.recommended_families:
            add("## Precedent from structurally similar past runs")
            add(
                "- families that won before: "
                + ", ".join(f.value for f in state.memory.recommended_families)
            )
            for run in state.memory.similar_runs[:3]:
                add(
                    f"- run {run.run_id}: best={run.best_family.value if run.best_family else 'n/a'} "
                    f"{run.primary_metric}={_num(run.best_score)} "
                    f"(similarity {_num(run.similarity, 3)}) — {run.why_similar}"
                )
            if state.memory.cautions:
                add(f"- cautions from those runs: {'; '.join(state.memory.cautions)}")
            add("- this is a prior, not an instruction. The evidence here outranks it.")
            add("")

        if state.plan:
            steps = [
                s
                for s in state.plan.ordered()
                if s.agent is AgentName.MODEL_SELECTION
            ]
            if steps:
                add("## What the plan asked of you")
                for step in steps:
                    add(f"- {step.title}: {step.objective} (because: {step.rationale})")
                add("")

        add("## Your task")
        add(
            "Select and rank the model families to train on this dataset. Make the "
            "comparative case in `reasoning`: which family you expect to win, which is "
            "the credible challenger, and what measured quantity separates them. "
            "Include `baseline_dummy` with is_baseline true, ranked last. Populate "
            "`excluded_families` with index-aligned `exclusion_rationale` entries. "
            "Choose `validation_strategy` from the data and defend it in "
            "`validation_rationale`. Stay within the experiment budget of "
            f"{state.config.max_experiments} model(s)."
        )
        return "\n".join(lines)

    # -- grounding ---------------------------------------------------------

    def postprocess(self, value: ModelSelection, state: RunState) -> ModelSelection:
        """Reconcile the selection with the environment and the budget.

        Drops families whose library is not installed, deduplicates repeated
        families, guarantees a baseline candidate, truncates to
        ``config.max_experiments`` while protecting the baseline, and renumbers
        ranks densely from 1.
        """
        is_available = self._availability_probe(state)

        kept: list[ModelCandidate] = []
        seen: set[ModelFamily] = set()
        for candidate in sorted(value.candidates, key=lambda c: c.rank):
            if not is_available(candidate.family):
                state.add_warning(
                    f"{self.title}: dropped candidate {candidate.family.value} — it is "
                    "not available in this environment (optional dependency missing or "
                    "unsupported for this task)."
                )
                continue
            if candidate.family in seen:
                state.add_warning(
                    f"{self.title}: removed a duplicate {candidate.family.value} "
                    "candidate, keeping the better-ranked one."
                )
                continue
            seen.add(candidate.family)
            # A dummy is a baseline whether or not the flag was set; the
            # leaderboard's honesty depends on that label being right.
            if (
                candidate.family is ModelFamily.BASELINE_DUMMY
                and not candidate.is_baseline
            ):
                candidate = candidate.model_copy(update={"is_baseline": True})
            kept.append(candidate)

        kept = self._ensure_baseline(kept, state, is_available)
        kept = self._truncate(kept, state)

        renumbered = [
            candidate
            if candidate.rank == position
            else candidate.model_copy(update={"rank": position})
            for position, candidate in enumerate(
                sorted(kept, key=lambda c: c.rank), start=1
            )
        ]

        return value.model_copy(update={"candidates": renumbered})

    def _availability_probe(self, state: RunState) -> Callable[[ModelFamily], bool]:
        """Return a predicate answering "can this family be built here?".

        The model zoo is imported lazily so that a missing or broken executor
        module degrades into "assume available" rather than failing the run.
        """
        try:
            from ..execution.model_zoo import (
                is_available as zoo_is_available,
                supports_task as zoo_supports_task,
            )
        except Exception as exc:  # noqa: BLE001 - executor is optional at this point
            state.add_warning(
                f"{self.title}: could not import execution.model_zoo to verify model "
                f"availability ({exc}); accepting every proposed family unchecked."
            )
            return lambda _family: True

        task = state.task_type

        def probe(family: ModelFamily) -> bool:
            try:
                if not zoo_is_available(family):
                    return False
                # Task legality matters as much as installation: the trainer
                # filters on installation alone, so a clustering family proposed
                # for a classification task would consume an experiment slot and
                # then raise TrainingError inside build_estimator.
                if task is not None and not zoo_supports_task(family, task):
                    return False
                return True
            except Exception as exc:  # noqa: BLE001 - a bad probe must not drop a model
                logger.debug("availability probe for %s raised: %s", family.value, exc)
                return True

        return probe

    def _ensure_baseline(
        self,
        candidates: list[ModelCandidate],
        state: RunState,
        is_available: Callable[[ModelFamily], bool],
    ) -> list[ModelCandidate]:
        """Inject a trivial baseline if the agent did not provide one."""
        if any(c.is_baseline for c in candidates):
            return candidates
        if not is_available(ModelFamily.BASELINE_DUMMY):
            state.add_warning(
                f"{self.title}: no baseline candidate was proposed and baseline_dummy "
                "is unavailable; the leaderboard will have no trivial reference point."
            )
            return candidates

        task = state.task_type
        reference = (
            "the majority class"
            if task is not None and task.is_classification
            else "the mean of the target"
        )
        next_rank = max((c.rank for c in candidates), default=0) + 1
        state.add_warning(
            f"{self.title}: no trivial baseline was proposed; injected baseline_dummy "
            "so the leaderboard has an honest reference point."
        )
        return candidates + [
            ModelCandidate(
                family=ModelFamily.BASELINE_DUMMY,
                rank=next_rank,
                suitability="poor",
                rationale=(
                    f"Injected by the orchestrator. A model that cannot beat {reference} "
                    "has learned nothing from the features, so every reported score is "
                    "read against this floor."
                ),
                expected_strengths=["Establishes the no-skill floor for every metric"],
                expected_weaknesses=["No predictive power by construction"],
                is_baseline=True,
                tune_priority="none",
            )
        ]

    def _truncate(
        self, candidates: list[ModelCandidate], state: RunState
    ) -> list[ModelCandidate]:
        """Cut to the operator's experiment budget, preserving rank and baseline."""
        limit = max(1, int(state.config.max_experiments))
        if len(candidates) <= limit:
            return candidates

        ordered = sorted(candidates, key=lambda c: c.rank)
        baselines = [c for c in ordered if c.is_baseline]
        others = [c for c in ordered if not c.is_baseline]

        if not baselines:
            kept = ordered[:limit]
        else:
            # Reserve a slot for the baseline. When the budget is 1 this yields two
            # candidates: a dummy costs milliseconds to fit, and a leaderboard with
            # no floor cannot be reported honestly.
            keep_others = max(1, limit - 1)
            kept = sorted(others[:keep_others] + baselines[:1], key=lambda c: c.rank)
            if len(kept) > limit:
                state.add_warning(
                    f"{self.title}: kept {len(kept)} candidates against a budget of "
                    f"{limit} so that the mandatory baseline survives."
                )

        surviving = {id(c) for c in kept}
        dropped = [c.family.value for c in ordered if id(c) not in surviving]
        if dropped:
            state.add_warning(
                f"{self.title}: experiment budget is {limit}; dropped lower-ranked "
                f"candidate(s) {dropped}."
            )
        return kept

    # -- state -------------------------------------------------------------

    def apply(self, state: RunState, value: ModelSelection) -> None:
        """Record the selection for the trainer and the report."""
        state.model_selection = value

    def decision_summary(self, value: ModelSelection) -> str:
        ranked = ", ".join(
            f"{c.rank}:{c.family.value}"
            for c in sorted(value.candidates, key=lambda c: c.rank)
        )
        return (
            f"{len(value.candidates)} candidate(s) [{ranked}]; "
            f"validation={value.validation_strategy or 'unspecified'}; "
            f"{len(value.excluded_families)} family/families explicitly excluded"
        )
