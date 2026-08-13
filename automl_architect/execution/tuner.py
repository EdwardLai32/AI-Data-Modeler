"""Hyperparameter search, with the result reported honestly.

Two rules make this module trustworthy rather than merely functional.

**Selection and measurement are separated.** The search optimises a
cross-validated score computed on the training partition only. The number
reported as ``best_score`` is then measured by refitting on train and scoring the
held-out partition — the same partition, the same code path, and the same metric
that produced ``baseline_score``. Reporting the search's own best CV score
against a held-out baseline would compare two different measurements and flatter
the tuner every time.

**A tuned model that lost stays lost.** If the search does not beat the untuned
score, ``improvement`` is negative or zero, ``state.best_model`` is left alone,
and the report says so. Promoting a worse model because it was expensive to find
is the single most tempting lie an AutoML tool can tell.

The agent's ``search_space`` is treated as a suggestion: entries naming a
parameter the estimator does not accept are dropped with a warning, and if
nothing usable survives, the zoo's own default space is used.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ..core.schemas import (
    ModelFamily,
    Param,
    SearchSpaceEntry,
    TuningDecision,
    TuningMethod,
    TuningResult,
    dict_to_params,
    params_to_dict,
)
from ..core.state import RunState
from .metrics import higher_is_better, is_better
from .model_zoo import (
    build_estimator,
    default_search_space,
    is_available,
    unsupported_params,
)
from .trainer import (
    TrainingContext,
    build_training_context,
    fit_measured,
    record_experiment,
    score_fitted_model,
    wrap_with_preprocessor,
)

logger = logging.getLogger(__name__)

__all__ = ["run_tuning"]

_MODEL_STEP = "model"
_PREFIXES = ("model__", "estimator__", "regressor__", "classifier__")

# Tuning is a refinement, not the run's purpose: it gets a hard trial ceiling and
# has to leave time for explainability, evaluation, and reporting.
_MAX_TRIALS = 200
_MIN_TUNING_SECONDS = 20.0
_MAX_GRID_POINTS = 400
# Points to materialise per continuous parameter when a grid backend needs
# discrete values.
_GRID_RESOLUTION = 5


@dataclass
class _Space:
    """One search space, rendered for each backend that might consume it."""

    entries: list[SearchSpaceEntry] = field(default_factory=list)
    grid: dict[str, list[Any]] = field(default_factory=dict)
    distributions: dict[str, Any] = field(default_factory=dict)
    source: str = "agent"

    def __bool__(self) -> bool:
        return bool(self.grid or self.distributions)

    @property
    def grid_points(self) -> int:
        total = 1
        for values in self.grid.values():
            total *= max(1, len(values))
        return total


def run_tuning(state: RunState, decision: TuningDecision) -> TuningResult:
    """Run the hyperparameter search the Tuning Agent asked for.

    Args:
        state: The run blackboard. Reads ``experiments`` for the baseline and
            the winning family; may update ``best_model``, ``best_pipeline`` and
            ``experiments`` when tuning wins.
        decision: The agent's typed decision, including its budget and space.

    Returns:
        A :class:`TuningResult`. ``ran=False`` with a ``skipped_reason`` when
        tuning was declined or impossible; ``error`` set when the search itself
        failed. Never raises: a failed search leaves the untuned winner in place.
    """
    result = TuningResult(method=decision.method, family=decision.target_family)
    state.tuning = result

    skip = _skip_reason(state, decision)
    if skip is not None:
        result.skipped_reason = skip
        state.bus.log(f"tuning skipped: {skip}")
        return result

    started = time.perf_counter()
    try:
        _tune(state, decision, result)
    except Exception as exc:  # noqa: BLE001 - tuning must never end the run
        result.ran = False
        result.error = f"{type(exc).__name__}: {_brief(exc, limit=600)}"
        state.add_warning(f"hyperparameter tuning failed ({_brief(exc)})")
        logger.warning("tuning failed: %s", _brief(exc))
        logger.debug("tuning traceback", exc_info=True)
    result.seconds = time.perf_counter() - started
    return result


def _skip_reason(state: RunState, decision: TuningDecision) -> str | None:
    """Every reason not to tune, checked before any compute is spent."""
    if not decision.worthwhile:
        return (
            decision.rationale
            or "the Tuning Agent judged the expected gain not worth the compute"
        )
    if decision.method is TuningMethod.NONE:
        return "the Tuning Agent selected no tuning method"
    if not state.config.enable_tuning:
        return "tuning is disabled for this run (config.enable_tuning=False)"
    if state.time_remaining < _MIN_TUNING_SECONDS:
        return (
            f"only {state.time_remaining:.0f}s of the time budget remains, which is "
            "not enough to tune and still report"
        )
    family = _resolve_family(state, decision)
    if family is None:
        return "no trained model is available to tune"
    if not is_available(family):
        return f"the package behind '{family.value}' is not installed"
    return None


def _resolve_family(state: RunState, decision: TuningDecision) -> ModelFamily | None:
    """The agent's target family, or the leaderboard winner."""
    if decision.target_family is not None:
        return decision.target_family
    log = state.experiments
    if log is None:
        return None
    best = log.best()
    if best is not None and not best.failed:
        return best.family
    viable = [r for r in log.results if not r.failed]
    return viable[0].family if viable else None


def _tune(state: RunState, decision: TuningDecision, result: TuningResult) -> None:
    """The search proper. Populates ``result`` in place."""
    ctx = build_training_context(state)
    family = _resolve_family(state, decision)
    assert family is not None  # guaranteed by _skip_reason
    result.family = family

    base_params, result.baseline_score = _baseline(state, family)
    space = _build_space(state, ctx, family, decision)
    if not space:
        result.skipped_reason = (
            f"no tunable hyperparameter survived validation for {family.value}"
        )
        state.bus.log(f"tuning skipped: {result.skipped_reason}")
        return

    timeout = _timeout(state, decision)
    n_trials, per_trial = _trial_budget(state, decision, family, timeout)
    method = decision.method

    if ctx.scorer is None or ctx.cv is None:
        result.skipped_reason = (
            f"'{ctx.metric}' cannot be cross-validated for {ctx.task.value}, so a "
            "search would have nothing trustworthy to optimise"
        )
        state.bus.log(f"tuning skipped: {result.skipped_reason}")
        return

    state.bus.log(
        f"tuning {family.value} with {method.value}: up to {n_trials} trial(s), "
        f"{timeout:.0f}s, {len(space.grid or space.distributions)} parameter(s) "
        f"from the {space.source} space"
    )

    if method in (TuningMethod.BAYESIAN, TuningMethod.OPTUNA_TPE):
        best_params, trial_scores, completed = _search_optuna(
            state, ctx, family, base_params, space, decision, n_trials, timeout
        )
    else:
        best_params, trial_scores, completed = _search_sklearn(
            state, ctx, family, base_params, space, method, n_trials, per_trial
        )

    result.trial_scores = trial_scores
    result.n_trials_completed = completed
    if not best_params:
        result.skipped_reason = "no trial produced a usable score"
        state.add_warning(f"tuning {family.value}: {result.skipped_reason}")
        return

    merged = {**base_params, **best_params}
    result.best_params = dict_to_params(best_params)
    result.ran = True

    # The honest comparison: refit on train alone, score the same held-out
    # partition the baseline was scored on.
    estimator = build_estimator(
        family, ctx.task, merged, random_state=ctx.random_state, **ctx.zoo_ctx
    )
    pipeline = wrap_with_preprocessor(state, estimator, family)
    fitted, train_seconds, peak_mb = fit_measured(pipeline, ctx)
    metrics, _ = score_fitted_model(state, ctx, fitted, family)
    result.best_score = metrics.get(ctx.metric)

    if result.best_score is None and trial_scores:
        result.best_score = max(trial_scores) if ctx.higher_better else min(trial_scores)
        state.add_warning(
            f"tuning {family.value}: the held-out partition produced no "
            f"{ctx.metric}; reporting the best cross-validated score instead"
        )

    result.improvement = _improvement(
        result.best_score, result.baseline_score, ctx.metric
    )
    _settle(state, ctx, result, family, merged, fitted, train_seconds, peak_mb)


def _settle(
    state: RunState,
    ctx: TrainingContext,
    result: TuningResult,
    family: ModelFamily,
    params: dict[str, Any],
    fitted: Any,
    train_seconds: float,
    peak_mb: float,
) -> None:
    """Record the tuned model, and promote it only if it actually leads.

    Two separate comparisons, because they answer different questions. Beating
    ``baseline_score`` means the search found better hyperparameters for *this
    family* — that is what ``improvement`` reports. Becoming ``best_model``
    requires beating the *whole leaderboard*: tuning the second-best family into
    third place is still a real result, but it is not a new winner.
    """
    improvement = result.improvement
    if improvement is None or improvement <= 0:
        message = (
            f"tuning {family.value} did not improve {ctx.metric}: "
            f"{_fmt(result.best_score)} vs baseline {_fmt(result.baseline_score)}; "
            "keeping the untuned model as the winner"
        )
        state.add_warning(message)
        state.bus.log(message)
        return

    log = state.experiments
    incumbent = log.best() if log is not None else None
    incumbent_score = incumbent.primary_score if incumbent else None

    experiment = record_experiment(
        state,
        ctx,
        family,
        params,
        fitted,
        tuned=True,
        train_seconds=train_seconds,
        peak_memory_mb=peak_mb,
    )
    if log is not None:
        log.results.append(experiment)

    leads = is_better(result.best_score, incumbent_score, ctx.metric)
    if not leads:
        message = (
            f"tuning improved {family.value} {ctx.metric} from "
            f"{_fmt(result.baseline_score)} to {_fmt(result.best_score)}, but "
            f"{incumbent.family.value if incumbent else 'the incumbent'} still "
            f"leads at {_fmt(incumbent_score)}; the winner is unchanged"
        )
        state.add_warning(message)
        state.bus.log(message)
        if log is not None:
            log.leaderboard_notes = (log.leaderboard_notes or "") + f"\n{message}."
        return

    if log is not None:
        log.best_experiment_id = experiment.experiment_id
        log.leaderboard_notes = (
            (log.leaderboard_notes or "")
            + f"\nTuned {family.value} is the new winner: {ctx.metric} "
            f"{_fmt(incumbent_score)} -> {_fmt(result.best_score)}."
        )
    state.best_pipeline = fitted
    state.best_model = (
        fitted.named_steps[_MODEL_STEP]
        if hasattr(fitted, "named_steps") and _MODEL_STEP in fitted.named_steps
        else fitted
    )
    state.bus.log(
        f"tuning improved {ctx.metric} by {improvement:.5g} "
        f"({_fmt(result.baseline_score)} -> {_fmt(result.best_score)})"
    )


# ---------------------------------------------------------------------------
# Baseline, budget
# ---------------------------------------------------------------------------


def _baseline(
    state: RunState, family: ModelFamily
) -> tuple[dict[str, Any], float | None]:
    """The untuned params and score for ``family``, from the leaderboard."""
    log = state.experiments
    if log is None:
        return {}, None
    matches = [
        r
        for r in log.results
        if r.family is family
        and not r.failed
        and not r.tuned
        and r.primary_score is not None
    ]
    if not matches:
        best = log.best()
        return {}, best.primary_score if best else None
    chooser = max if log.higher_is_better else min
    reference = chooser(matches, key=lambda r: r.primary_score)
    return params_to_dict(reference.params), reference.primary_score


def _timeout(state: RunState, decision: TuningDecision) -> float:
    """Seconds the search may use, bounded by what the run has left."""
    requested = float(decision.timeout_seconds or 300)
    # Keep a slice of the budget back for explainability and reporting.
    available = max(_MIN_TUNING_SECONDS, state.time_remaining * 0.6)
    return max(5.0, min(requested, available))


def _trial_budget(
    state: RunState,
    decision: TuningDecision,
    family: ModelFamily,
    timeout: float,
) -> tuple[int, float]:
    """How many trials fit in ``timeout``, from the measured baseline fit cost.

    sklearn's search estimators cannot be interrupted mid-run, so the timeout has
    to be enforced by not starting trials that will not finish. The per-trial
    estimate comes from the untuned model's measured fit time multiplied by the
    fold count — a real measurement rather than a guess.
    """
    requested = max(1, min(int(decision.n_trials or 25), _MAX_TRIALS))
    log = state.experiments
    fit_seconds = 0.0
    if log is not None:
        for record in log.results:
            if record.family is family and not record.failed:
                fit_seconds = max(fit_seconds, record.train_seconds)
    n_splits = max(1, _splits_of(state))
    per_trial = max(0.02, fit_seconds) * n_splits
    affordable = max(1, int(timeout / per_trial))
    if affordable < requested:
        state.add_warning(
            f"tuning {family.value}: {requested} trials would need about "
            f"{requested * per_trial:.0f}s but only {timeout:.0f}s is available; "
            f"running {affordable}"
        )
    return min(requested, affordable), per_trial


def _splits_of(state: RunState) -> int:
    return max(2, int(state.config.cv_folds or 5))


def _improvement(
    tuned: float | None, baseline: float | None, metric: str
) -> float | None:
    """Signed gain in the metric's better direction; positive means tuning won."""
    if tuned is None or baseline is None:
        return None
    if not (np.isfinite(tuned) and np.isfinite(baseline)):
        return None
    delta = float(tuned) - float(baseline)
    return delta if higher_is_better(metric) else -delta


# ---------------------------------------------------------------------------
# Search space translation
# ---------------------------------------------------------------------------


def _build_space(
    state: RunState,
    ctx: TrainingContext,
    family: ModelFamily,
    decision: TuningDecision,
) -> _Space:
    """Turn the agent's entries into backend-ready spaces, dropping the unusable."""
    entries = [e for e in decision.search_space if e.name]
    normalised: list[SearchSpaceEntry] = []
    for entry in entries:
        name = entry.name
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalised.append(entry.model_copy(update={"name": name}))

    if normalised:
        rejected = set(
            unsupported_params(
                family, ctx.task, {e.name: None for e in normalised}, **ctx.zoo_ctx
            )
        )
        usable: list[SearchSpaceEntry] = []
        for entry in normalised:
            if entry.name in rejected:
                state.add_warning(
                    f"tuning {family.value}: dropped search-space entry "
                    f"'{entry.name}': the estimator does not accept it"
                )
                continue
            if not _entry_is_sane(entry):
                state.add_warning(
                    f"tuning {family.value}: dropped search-space entry "
                    f"'{entry.name}': its range is unusable "
                    f"(kind={entry.kind}, low={entry.low}, high={entry.high})"
                )
                continue
            usable.append(entry)
        if usable:
            space = _Space(entries=usable, source="agent")
            for entry in usable:
                space.grid[entry.name] = _entry_values(entry)
                space.distributions[entry.name] = _entry_distribution(entry)
            return space
        state.add_warning(
            f"tuning {family.value}: no agent search-space entry was usable; "
            "falling back to the zoo's default space"
        )

    fallback = default_search_space(family, ctx.task)
    if not fallback:
        return _Space(source="default")
    rejected = set(
        unsupported_params(
            family, ctx.task, {k: None for k in fallback}, **ctx.zoo_ctx
        )
    )
    space = _Space(source="default")
    for name, values in fallback.items():
        if name in rejected or not values:
            continue
        space.grid[name] = list(values)
        space.distributions[name] = list(values)
        space.entries.append(
            SearchSpaceEntry(
                name=name,
                kind="categorical",
                choices=[_render(v) for v in values],
                rationale="Zoo default space.",
            )
        )
    return space


def _entry_is_sane(entry: SearchSpaceEntry) -> bool:
    if entry.kind == "categorical":
        return len(entry.choices) >= 2
    if entry.low is None or entry.high is None:
        return False
    if not (np.isfinite(entry.low) and np.isfinite(entry.high)):
        return False
    if entry.high <= entry.low:
        return False
    if entry.kind == "log_float" and entry.low <= 0:
        return False
    if entry.kind == "int" and int(entry.high) - int(entry.low) < 1:
        return False
    return True


def _entry_values(entry: SearchSpaceEntry) -> list[Any]:
    """Discrete values for a grid backend."""
    if entry.kind == "categorical":
        return [_coerce_choice(c) for c in entry.choices]
    low = float(entry.low or 0.0)
    high = float(entry.high or 0.0)
    if entry.kind == "int":
        span = int(high) - int(low)
        count = min(_GRID_RESOLUTION, span + 1)
        values = np.unique(
            np.linspace(int(low), int(high), num=max(2, count)).round().astype(int)
        )
        return [int(v) for v in values]
    if entry.kind == "log_float":
        values = np.logspace(np.log10(low), np.log10(high), num=_GRID_RESOLUTION)
    else:
        values = np.linspace(low, high, num=_GRID_RESOLUTION)
    return [float(v) for v in values]


def _entry_distribution(entry: SearchSpaceEntry) -> Any:
    """A scipy distribution (or choice list) for randomised backends."""
    from scipy import stats

    if entry.kind == "categorical":
        return [_coerce_choice(c) for c in entry.choices]
    low = float(entry.low or 0.0)
    high = float(entry.high or 0.0)
    if entry.kind == "int":
        return stats.randint(int(low), int(high) + 1)
    if entry.kind == "log_float":
        return stats.loguniform(low, high)
    return stats.uniform(loc=low, scale=high - low)


def _coerce_choice(choice: str) -> Any:
    """Parse a categorical choice with the same rules as ``Param`` values."""
    return params_to_dict([Param(key="v", value=choice)])["v"]


def _render(value: Any) -> str:
    """Inverse of :func:`_coerce_choice`, matching ``schemas.dict_to_params``.

    ``repr`` is not that inverse: ``repr((128, 64))`` is ``'(128, 64)'``, which
    ``_coerce_choice`` cannot parse and hands back as a *string*. That silently
    turned every Optuna trial on the zoo's ``neural_network`` and ``sarimax``
    default spaces into an ``InvalidParameterError``. JSON round-trips a tuple to
    a list, which sklearn and the statsmodels wrapper both accept.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _prefixed(space: dict[str, Any], pipeline: Any) -> dict[str, Any]:
    """Address parameters through the Pipeline's model step when there is one."""
    if not hasattr(pipeline, "named_steps"):
        return dict(space)
    return {f"{_MODEL_STEP}__{name}": value for name, value in space.items()}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _search_sklearn(
    state: RunState,
    ctx: TrainingContext,
    family: ModelFamily,
    base_params: dict[str, Any],
    space: _Space,
    method: TuningMethod,
    n_trials: int,
    per_trial: float,
) -> tuple[dict[str, Any], list[float], int]:
    """Grid, random, or halving-random search via sklearn."""
    estimator = build_estimator(
        family, ctx.task, base_params, random_state=ctx.random_state, **ctx.zoo_ctx
    )
    pipeline = wrap_with_preprocessor(state, estimator, family)
    temporal = "TimeSeries" in type(ctx.cv).__name__

    if method is TuningMethod.GRID_SEARCH:
        points = space.grid_points
        if points <= min(n_trials, _MAX_GRID_POINTS):
            search = GridSearchCV(
                pipeline,
                param_grid=_prefixed(space.grid, pipeline),
                scoring=ctx.scorer.scorer,
                cv=ctx.cv,
                n_jobs=1,
                error_score=np.nan,
                refit=False,
            )
        else:
            state.add_warning(
                f"tuning {family.value}: the grid has {points} combinations but "
                f"only {n_trials} trials fit the budget; sampling it randomly "
                "instead of enumerating it"
            )
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=_prefixed(space.grid, pipeline),
                n_iter=n_trials,
                scoring=ctx.scorer.scorer,
                cv=ctx.cv,
                n_jobs=1,
                random_state=ctx.random_state,
                error_score=np.nan,
                refit=False,
            )
    elif method is TuningMethod.HALVING_RANDOM and not temporal:
        # Successive halving is imported through the experimental gate; sklearn
        # refuses the plain import.
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV

        search = HalvingRandomSearchCV(
            pipeline,
            param_distributions=_prefixed(space.distributions, pipeline),
            n_candidates=max(2, n_trials),
            factor=3,
            scoring=ctx.scorer.scorer,
            cv=ctx.cv,
            n_jobs=1,
            random_state=ctx.random_state,
            error_score=np.nan,
            refit=False,
        )
    else:
        if method is TuningMethod.HALVING_RANDOM:
            state.add_warning(
                "tuning: successive halving subsamples rows at random, which "
                "breaks a temporal split; using random search instead"
            )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=_prefixed(space.distributions, pipeline),
            n_iter=n_trials,
            scoring=ctx.scorer.scorer,
            cv=ctx.cv,
            n_jobs=1,
            random_state=ctx.random_state,
            error_score=np.nan,
            refit=False,
        )

    fit_kwargs: dict[str, Any] = {}
    if ctx.cv_groups is not None:
        fit_kwargs["groups"] = ctx.cv_groups
    search.fit(ctx.X_train, ctx.y_train, **fit_kwargs)

    results = getattr(search, "cv_results_", {}) or {}
    raw = np.asarray(results.get("mean_test_score", []), dtype=float)
    trial_scores = [ctx.scorer.to_natural(v) for v in raw if np.isfinite(v)]
    completed = int(len(results.get("params", [])))
    best = _strip_prefix(getattr(search, "best_params_", {}) or {})
    return best, trial_scores, completed


def _search_optuna(
    state: RunState,
    ctx: TrainingContext,
    family: ModelFamily,
    base_params: dict[str, Any],
    space: _Space,
    decision: TuningDecision,
    n_trials: int,
    timeout: float,
) -> tuple[dict[str, Any], list[float], int]:
    """Bayesian search via Optuna, falling back to random search when absent."""
    try:
        import optuna
    except ImportError:
        state.add_warning(
            "optuna is not installed; falling back to random search for "
            f"{family.value}"
        )
        return _search_sklearn(
            state,
            ctx,
            family,
            base_params,
            space,
            TuningMethod.RANDOM_SEARCH,
            n_trials,
            0.0,
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splits = list(ctx.cv.split(ctx.X_train, ctx.y_train, ctx.cv_groups))

    def objective(trial: Any) -> float:
        suggested = {
            entry.name: _suggest(trial, entry) for entry in space.entries
        }
        estimator = build_estimator(
            family,
            ctx.task,
            {**base_params, **suggested},
            random_state=ctx.random_state,
            **ctx.zoo_ctx,
        )
        pipeline = wrap_with_preprocessor(state, estimator, family)
        fold_scores: list[float] = []
        for step, (train_idx, valid_idx) in enumerate(splits):
            candidate = clone(pipeline)
            X_tr = _rows(ctx.X_train, train_idx)
            X_va = _rows(ctx.X_train, valid_idx)
            y_tr = _rows(ctx.y_train, train_idx)
            y_va = _rows(ctx.y_train, valid_idx)
            candidate.fit(X_tr, y_tr)
            fold_scores.append(float(ctx.scorer.scorer(candidate, X_va, y_va)))
            # Report the running mean so a hopeless trial can be cut short
            # instead of burning the remaining folds.
            trial.report(float(np.mean(fold_scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
        if decision.early_stopping
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="maximize",  # the scorer is already greater-is-better
        sampler=optuna.samplers.TPESampler(seed=ctx.random_state),
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        catch=(Exception,),  # one bad parameter combination is not a run failure
        gc_after_trial=True,
    )

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    if pruned or failed:
        # n_trials_completed counts only trials that produced a score, so say
        # what happened to the rest rather than letting the count look short.
        state.bus.log(
            f"optuna ran {len(study.trials)} trial(s) for {family.value}: "
            f"{len(completed)} scored, {pruned} pruned early, {failed} failed"
        )
    trial_scores = [
        ctx.scorer.to_natural(t.value) for t in completed if t.value is not None
    ]
    if not completed:
        return {}, trial_scores, 0
    return dict(study.best_trial.params), trial_scores, len(completed)


def _suggest(trial: Any, entry: SearchSpaceEntry) -> Any:
    if entry.kind == "categorical":
        return trial.suggest_categorical(
            entry.name, [_coerce_choice(c) for c in entry.choices]
        )
    low = float(entry.low or 0.0)
    high = float(entry.high or 0.0)
    if entry.kind == "int":
        return trial.suggest_int(entry.name, int(low), int(high))
    if entry.kind == "log_float":
        return trial.suggest_float(entry.name, low, high, log=True)
    return trial.suggest_float(entry.name, low, high)


def _strip_prefix(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        name = key
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
        out[name] = value
    return out


def _rows(data: Any, index: np.ndarray) -> Any:
    if data is None:
        return None
    if hasattr(data, "iloc"):
        return data.iloc[index]
    return np.asarray(data)[index]


def _fmt(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.5g}"


def _brief(exc: BaseException, limit: int = 240) -> str:
    """One-line, length-capped exception text; the full trace goes to the log."""
    text = " ".join(str(exc).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
