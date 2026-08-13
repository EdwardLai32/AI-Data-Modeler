"""The Tuning Agent: decide whether to tune at all, then how.

Most AutoML systems tune because tuning is a step in their pipeline. This one
makes it a judgement call, because on a 3,000-row table with a 0.013
cross-validation standard deviation, a 40-trial search mostly discovers which
hyperparameters happen to suit the validation folds. That is not an improvement;
it is an overfit dressed as one, and it costs the time budget that explainability
and evaluation need.

So the agent's first output is ``worthwhile``, argued from measured evidence:
the size of the best-vs-baseline gap, the fold-to-fold noise floor, the row
count, and the seconds actually left in the budget. :meth:`TuningAgent.postprocess`
then enforces the constraints the model cannot be trusted to respect on its own —
a disabled tuning flag, an exhausted clock, a target family that never trained,
a trial count the budget cannot pay for.

The agent decides; ``execution.tuner.run_tuning`` executes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..core.agent import BaseAgent
from ..core.llm import Effort
from ..core.schemas import (
    AgentName,
    ExperimentLog,
    ExperimentResult,
    ModelFamily,
    SearchSpaceEntry,
    TuningDecision,
    TuningMethod,
)
from ..core.state import RunState

# Imported for the cache key only: on a replan, a previous evaluation pass has
# already measured the fit, and that diagnosis should steer the search space.
from .evaluation import DIAGNOSTICS_EXTRA_KEY
from .experiment import (
    rank_results,
    render_leaderboard,
    render_margin_analysis,
    successful_results,
)

logger = logging.getLogger(__name__)

#: Below this many seconds left, tuning cannot finish anything useful and the
#: remaining budget belongs to evaluation and reporting.
MIN_TUNING_SECONDS = 90.0

#: Seconds held back from tuning for explainability, diagnostics, and the report.
DOWNSTREAM_RESERVE_SECONDS = 60.0

#: A search with fewer trials than this is noise, not a search.
MIN_USEFUL_TRIALS = 5

#: Hard ceiling regardless of what the budget arithmetic allows.
MAX_TRIALS_CAP = 300

_CONTINUOUS_KINDS = {"float", "log_float"}


def _g(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    return f"{number:.{digits}g}"


@dataclass(frozen=True)
class TuningBudget:
    """What the clock actually allows, measured rather than guessed."""

    time_remaining: float
    reserve_seconds: float
    budget_seconds: float
    seconds_per_trial: float
    affordable_trials: int
    basis: str


def estimate_tuning_budget(state: RunState, family: ModelFamily | None = None) -> TuningBudget:
    """Estimate how many tuning trials the remaining wall clock can pay for.

    One trial is priced as one full cross-validation of the candidate, using the
    measured single-fit time from the experiment log.

    Args:
        state: The live run state, for the clock and CV configuration.
        family: Family whose measured fit time to price. Defaults to the winner.

    Returns:
        A :class:`TuningBudget` describing the arithmetic, including the basis
        string so it can be shown to the agent as a measured fact.
    """
    log = state.experiments
    reference: ExperimentResult | None = None
    if log is not None:
        candidates = [r for r in successful_results(log) if not r.is_baseline]
        if family is not None:
            reference = next((r for r in candidates if r.family is family), None)
        if reference is None:
            reference = next(
                (r for r in rank_results(log) if not r.is_baseline), None
            )

    folds = max(1, int(state.config.cv_folds or 1))
    fit_seconds = float(reference.train_seconds) if reference else 0.0
    if fit_seconds > 0:
        seconds_per_trial = max(fit_seconds * folds, 0.05)
        basis = (
            f"{fit_seconds:.2f}s measured single fit for "
            f"`{reference.family.value}` x {folds} folds"  # type: ignore[union-attr]
        )
    else:
        # No timing was recorded; assume a cheap fit rather than blocking tuning.
        seconds_per_trial = 1.0 * folds
        basis = f"no measured fit time; assumed 1.00s x {folds} folds"

    remaining = float(state.time_remaining)
    budget = max(0.0, remaining - DOWNSTREAM_RESERVE_SECONDS)
    affordable = int(budget // seconds_per_trial)
    return TuningBudget(
        time_remaining=remaining,
        reserve_seconds=DOWNSTREAM_RESERVE_SECONDS,
        budget_seconds=budget,
        seconds_per_trial=seconds_per_trial,
        affordable_trials=min(affordable, MAX_TRIALS_CAP),
        basis=basis,
    )


def _tunable_families(log: ExperimentLog | None) -> list[ModelFamily]:
    """Families that trained successfully and are worth tuning (not baselines)."""
    if log is None:
        return []
    seen: list[ModelFamily] = []
    for result in rank_results(log):
        if result.is_baseline:
            continue
        if result.family not in seen:
            seen.append(result.family)
    return seen


def _render_params(result: ExperimentResult, limit: int = 10) -> str:
    if not result.params:
        return "defaults (none recorded)"
    rendered = ", ".join(f"{p.key}={p.value}" for p in result.params[:limit])
    if len(result.params) > limit:
        rendered += f", (+{len(result.params) - limit} more)"
    return rendered


_INSTRUCTIONS = """\
You are the hyperparameter-optimisation strategist. Your first decision is not \
which search to run — it is whether searching is worth the compute at all, on \
this dataset, with this leaderboard and this much clock left. A senior \
practitioner declines to tune far more often than an AutoML product does, and \
declining with a good argument is a correct answer here, not a failure.

METHOD — work the cost/benefit in this order:

1.  Is there anything to tune? At least one non-baseline candidate must have \
trained successfully. Never target the baseline: it exists to be beaten, not \
improved.
2.  Estimate the headroom. If the best model already sits far above the \
baseline and near the metric's practical ceiling (ROC AUC above ~0.97, R^2 above \
~0.95, accuracy above ~0.98), the remaining distance is mostly irreducible noise \
and tuning buys decimals. If the best model barely beats the baseline, the \
problem is usually features or the model class — not hyperparameters — and \
tuning will not rescue it. Say which of those two situations you are in.
3.  Find the noise floor. Realistic tuning gains are roughly 1-3% relative for \
tree ensembles from sensible defaults, less for linear models. Compare that \
against the measured cross-validation standard deviation and against the \
confidence the row count can support. When the plausible gain is smaller than \
the fold-to-fold spread, you would be selecting on noise; on a few thousand rows \
that is the common case.
4.  Check for a tie at the top. When the leading candidates sit within one \
pooled CV standard deviation, tuning may simply reorder noise. If you tune \
anyway, pick the target on structural grounds — which family's inductive bias \
fits this data's shape, cardinality, and interaction structure — not on which \
one is currently a hair ahead, and say that is what you are doing.
5.  Price it against the clock. You are given the measured seconds per trial \
and the number of trials the remaining budget can actually pay for. If the \
affordable count is small, either narrow the space to the one or two parameters \
that dominate this family's behaviour and run a short honest search, or decline \
and leave the time to evaluation and reporting. Never propose a search the \
budget cannot finish.

CHOOSING THE METHOD:

-   grid_search: only when the entire discrete product is small enough to \
enumerate inside the trial budget — two or three parameters with a handful of \
values each. A grid over a continuous or log-scaled range is not a grid; it is \
an arbitrary discretisation.
-   random_search: the default for a broad, cheap sweep, and clearly better than \
a grid once the space has four or more dimensions, because most dimensions do \
not matter and random sampling spends its budget on the ones that do.
-   optuna_tpe: when each trial is expensive (roughly a second or more of fit \
time) and the space is continuous or log-scaled, so that a model of the \
objective earns back the bookkeeping. This is the right choice for gradient \
boosting on a non-trivial table.
-   halving_random: when fits are cheap to abandon early and the data is large \
enough that a fraction of it still ranks candidates — successive halving spends \
its budget on survivors.
-   bayesian: only if you specifically want a Gaussian-process style surrogate \
over a small continuous space with very expensive trials.

DEFINING THE SEARCH SPACE — this is where a real practitioner is visible:

Choose every bound from the numbers in front of you, and justify each one with \
the measurement that set it. Depth ceilings follow from row count and feature \
count (a depth-12 tree on 2,000 rows partitions into leaves of two). \
Minimum-leaf sizes follow from the minority-class count when the target is \
imbalanced. Learning rate belongs on a log scale and must be traded against the \
estimator count. Regularisation ranges widen when the matrix is wide relative to \
its height. Subsample and column-sample ranges tighten when rows are few. Centre \
ranges on the defaults that produced the measured score, and extend in the \
direction the diagnosis argues for: overfitting -> more regularisation, more \
leaf mass, lower depth; underfitting -> more capacity, more estimators, lower \
regularisation. Prefer three or four parameters that matter over eight that \
dilute the trial budget: with 25 trials, an eight-dimensional space is sampled \
essentially nowhere.

FAILURE MODES TO AVOID, SPECIFIC TO THIS JOB:

-   Proposing a textbook grid unrelated to this dataset's size and shape.
-   Tuning against the test set. The executor searches with cross-validation on \
the training partition only; the holdout stays untouched. Do not ask for \
anything that would consult it.
-   Setting n_trials or timeout_seconds beyond the affordable figure you were \
given, which strands the run with no time to evaluate or report.
-   Claiming a specific numeric gain. You do not know it yet; describe the \
expected gain in terms of the noise floor instead.
-   Searching many parameters at low trial counts.
-   Using a grid over a log-scaled parameter, or a linear range for learning \
rate or regularisation strength.

A good rationale: "The best model (hist_gradient_boosting, ROC AUC 0.8731) is \
0.373 above the dummy baseline, so there is real signal, but its CV standard \
deviation is 0.0128 on 4,120 rows and the runner-up is only 0.0042 behind. A \
typical 1-2% relative gain here is ~0.009-0.017, straddling that spread, so the \
expected payoff is marginal but not zero. With 480s left and 1.9s per trial \
(0.38s fit x 5 folds), 25 TPE trials over four parameters is affordable and \
leaves the reserve intact; I am targeting learning_rate and max_leaf_nodes \
because the train-CV gap of 0.06 points to variance rather than bias."

A bad rationale: "Tuning is a best practice and usually improves performance, so \
we should run a grid search over the standard hyperparameters to find the \
optimal configuration." — no measured evidence, no cost, no noise floor, no \
reason for the method or the space.
"""


class TuningAgent(BaseAgent[TuningDecision]):
    """Judges whether hyperparameter search earns its compute, and designs it."""

    name = AgentName.TUNING
    title = "Tuning Agent"
    output_model = TuningDecision
    effort: Effort = "high"
    max_tokens = 16_000

    # -- prompt -------------------------------------------------------------

    def instructions(self, state: RunState) -> str:
        return _INSTRUCTIONS

    def build_prompt(self, state: RunState) -> str:
        log = state.experiments
        budget = estimate_tuning_budget(state)
        sizes = state.splits.sizes()
        families = _tunable_families(log)

        parts: list[str] = ["## TUNING DECISION", ""]
        parts.append("### Run position")
        parts.append(
            f"- task: {state.task_type.value if state.task_type else 'unknown'}; "
            f"primary metric: {state.primary_metric} "
            f"({'higher' if (log.higher_is_better if log else True) else 'lower'} is better)"
        )
        parts.append(
            f"- rows available for training: {sizes['train']:,} "
            f"(validation {sizes['validation']:,}, test {sizes['test']:,})"
        )
        parts.append(
            f"- engineered features: {len(state.feature_names) or 'unrecorded'}"
        )
        parts.append(f"- cross-validation folds: {state.config.cv_folds}")
        parts.append(
            f"- split strategy: {state.splits.strategy or 'unrecorded'}"
            + (f" - {state.splits.rationale}" if state.splits.rationale else "")
        )
        parts.append("")

        parts.append("### Clock (measured)")
        parts.append(
            f"- elapsed {state.elapsed_seconds:.0f}s of "
            f"{state.config.time_budget_seconds}s; {budget.time_remaining:.0f}s remain"
        )
        parts.append(
            f"- reserved for evaluation, explainability and reporting: "
            f"{budget.reserve_seconds:.0f}s"
        )
        parts.append(f"- spendable on tuning: {budget.budget_seconds:.0f}s")
        parts.append(
            f"- estimated cost of one trial: {budget.seconds_per_trial:.2f}s "
            f"({budget.basis})"
        )
        parts.append(
            f"- **affordable trials: about {budget.affordable_trials}**. Do not "
            "exceed this; n_trials and timeout_seconds above it will be clamped."
        )
        parts.append(
            f"- operator setting `enable_tuning`: "
            f"{str(state.config.enable_tuning).lower()}"
        )
        parts.append("")

        parts.append(render_leaderboard(log, limit=10))
        parts.append("")
        parts.append(render_margin_analysis(log))
        parts.append("")

        if families:
            parts.append("### Hyperparameters that produced those scores")
            for family in families[:5]:
                result = next(
                    (
                        r
                        for r in rank_results(log)  # type: ignore[arg-type]
                        if r.family is family
                    ),
                    None,
                )
                if result is None:
                    continue
                parts.append(f"- `{family.value}`: {_render_params(result)}")
            parts.append("")
            parts.append(
                "Tunable target families (trained successfully, not baselines): "
                + ", ".join(f"`{f.value}`" for f in families)
                + ". Any other family will be rejected."
            )
        else:
            parts.append(
                "No non-baseline candidate trained successfully, so there is "
                "nothing to tune."
            )
        parts.append("")

        if state.model_selection and state.model_selection.candidates:
            priorities = [
                f"`{c.family.value}`={c.tune_priority}"
                for c in state.model_selection.candidates
                if c.tune_priority != "none"
            ]
            if priorities:
                parts.append(
                    "### Tuning priority assigned upstream by model selection"
                )
                parts.append("- " + ", ".join(priorities))
                parts.append("")

        train_gap_note = self._train_gap_note(state)
        if train_gap_note:
            parts.append(train_gap_note)
            parts.append("")

        parts.append(
            "Decide whether tuning is worth it here. If it is not, set worthwhile "
            "to false, give the cost/benefit argument, and stop. If it is, choose "
            "the method, name the target family from the list above, set n_trials "
            "and timeout_seconds inside the affordable budget, and define a search "
            "space in which every range is justified by a number from this prompt "
            "or the dataset facts."
        )
        return "\n".join(parts)

    @staticmethod
    def _train_gap_note(state: RunState) -> str:
        """Surface an existing overfit/underfit read, which steers the space."""
        diagnostics = state.extras.get(DIAGNOSTICS_EXTRA_KEY)
        bias_variance = getattr(diagnostics, "bias_variance", None)
        if bias_variance is None:
            return ""
        gap = getattr(bias_variance, "gap", None)
        verdict = getattr(bias_variance, "verdict", None)
        if gap is None and verdict in (None, "inconclusive"):
            return ""
        return (
            "### Existing fit diagnosis (from a previous evaluation pass)\n"
            f"- verdict: {verdict}; train-vs-holdout gap: {_g(gap)}. "
            "Let this steer the direction of the ranges: variance argues for more "
            "regularisation, bias for more capacity."
        )

    # -- grounding ----------------------------------------------------------

    def postprocess(self, value: TuningDecision, state: RunState) -> TuningDecision:
        """Force the decision inside the operator's and the clock's limits."""
        decision = value.model_copy(deep=True)
        log = state.experiments
        families = _tunable_families(log)
        budget = estimate_tuning_budget(state, decision.target_family)
        clamps: list[str] = []

        # --- veto conditions -------------------------------------------------
        if not state.config.enable_tuning:
            clamps.append(
                "Tuning was disabled by the operator (enable_tuning=false), so no "
                "search will run regardless of its expected value."
            )
        elif not families:
            clamps.append(
                "No non-baseline candidate trained successfully, so there is no "
                "estimator to tune."
            )
        elif state.time_remaining < MIN_TUNING_SECONDS:
            clamps.append(
                f"Only {state.time_remaining:.0f}s of the run budget remain, below "
                f"the {MIN_TUNING_SECONDS:.0f}s floor for a meaningful search; the "
                "remaining time is reserved for evaluation and reporting."
            )
        elif decision.worthwhile and budget.affordable_trials < MIN_USEFUL_TRIALS:
            clamps.append(
                f"The remaining {budget.budget_seconds:.0f}s pays for about "
                f"{budget.affordable_trials} trial(s) at "
                f"{budget.seconds_per_trial:.2f}s each, fewer than the "
                f"{MIN_USEFUL_TRIALS} needed for a search to beat luck."
            )

        if clamps:
            decision.worthwhile = False
            decision.method = TuningMethod.NONE
            decision.n_trials = 0
            decision.timeout_seconds = 0
            decision.rationale = self._merge_rationale(clamps, value.rationale)
            state.add_warning(f"{self.title}: tuning suppressed. {clamps[0]}")
            return decision

        if not decision.worthwhile:
            # The agent declined on the merits. Keep its argument, but make the
            # numeric fields consistent with "nothing will run".
            decision.method = TuningMethod.NONE
            decision.n_trials = 0
            decision.timeout_seconds = 0
            return decision

        # --- target family ---------------------------------------------------
        if decision.target_family not in families:
            requested = (
                decision.target_family.value
                if decision.target_family
                else "unspecified"
            )
            decision.target_family = families[0]
            clamps.append(
                f"Target family '{requested}' did not train successfully; "
                f"retargeted to `{families[0].value}`, the best measured candidate."
            )
            budget = estimate_tuning_budget(state, decision.target_family)

        # --- search space ----------------------------------------------------
        decision.search_space = self._clean_search_space(
            decision.search_space, state, clamps
        )
        if not decision.search_space:
            clamps.append(
                "No usable search-space entry survived validation; the executor's "
                "default space for this family will be used instead."
            )

        # --- method ----------------------------------------------------------
        has_continuous = any(
            entry.kind in _CONTINUOUS_KINDS for entry in decision.search_space
        )
        if decision.method is TuningMethod.NONE:
            fallback = (
                TuningMethod.OPTUNA_TPE if has_continuous else TuningMethod.RANDOM_SEARCH
            )
            decision.method = fallback
            clamps.append(
                f"Tuning was marked worthwhile but no method was chosen; using "
                f"{fallback.value}, which suits the declared space."
            )
        elif decision.method is TuningMethod.GRID_SEARCH and has_continuous:
            decision.method = TuningMethod.RANDOM_SEARCH
            clamps.append(
                "A grid cannot enumerate a continuous or log-scaled range, so the "
                "method was switched to random_search over the same space."
            )

        # --- budget ----------------------------------------------------------
        max_trials = max(MIN_USEFUL_TRIALS, budget.affordable_trials)
        requested_trials = max(1, int(decision.n_trials or 0))
        if requested_trials > max_trials:
            clamps.append(
                f"n_trials reduced from {requested_trials} to {max_trials}: the "
                f"remaining {budget.budget_seconds:.0f}s at "
                f"{budget.seconds_per_trial:.2f}s per trial cannot pay for more."
            )
        decision.n_trials = min(requested_trials, max_trials, MAX_TRIALS_CAP)

        max_timeout = int(budget.budget_seconds)
        requested_timeout = int(decision.timeout_seconds or 0)
        if requested_timeout <= 0:
            decision.timeout_seconds = max_timeout
        elif requested_timeout > max_timeout:
            decision.timeout_seconds = max_timeout
            clamps.append(
                f"timeout_seconds reduced from {requested_timeout}s to "
                f"{max_timeout}s to preserve the "
                f"{DOWNSTREAM_RESERVE_SECONDS:.0f}s reserved for evaluation and "
                "reporting."
            )
        else:
            decision.timeout_seconds = requested_timeout

        if len(decision.search_space) > 6:
            state.add_warning(
                f"{self.title}: {len(decision.search_space)} hyperparameters over "
                f"{decision.n_trials} trials samples the space sparsely."
            )

        if clamps:
            decision.rationale = self._merge_rationale(clamps, value.rationale)
            for clamp in clamps:
                state.add_warning(f"{self.title}: {clamp}")
        return decision

    def _clean_search_space(
        self,
        entries: list[SearchSpaceEntry],
        state: RunState,
        clamps: list[str],
    ) -> list[SearchSpaceEntry]:
        """Drop or repair entries the executor could not sample from."""
        cleaned: list[SearchSpaceEntry] = []
        seen: set[str] = set()
        for entry in entries:
            name = (entry.name or "").strip()
            if not name or name in seen:
                continue
            item = entry.model_copy(deep=True)
            item.name = name

            if item.kind == "categorical":
                choices = [c for c in item.choices if str(c).strip() != ""]
                if len(choices) < 2:
                    clamps.append(
                        f"Dropped categorical hyperparameter '{name}': fewer than "
                        "two distinct choices means nothing to search."
                    )
                    continue
                item.choices = choices
            else:
                if item.low is None or item.high is None:
                    clamps.append(
                        f"Dropped hyperparameter '{name}': a numeric range needs "
                        "both a low and a high bound."
                    )
                    continue
                if item.high <= item.low:
                    clamps.append(
                        f"Dropped hyperparameter '{name}': the declared range "
                        f"[{_g(item.low)}, {_g(item.high)}] is empty."
                    )
                    continue
                if item.kind == "log_float" and item.low <= 0:
                    item.kind = "float"
                    clamps.append(
                        f"Hyperparameter '{name}' was declared log-scaled with a "
                        f"non-positive lower bound ({_g(item.low)}); sampled on a "
                        "linear scale instead."
                    )
            seen.add(name)
            cleaned.append(item)
        return cleaned

    @staticmethod
    def _merge_rationale(clamps: list[str], original: str) -> str:
        """Keep the agent's argument while recording what overrode it."""
        forced = " ".join(clamps)
        original = (original or "").strip()
        if not original:
            return forced
        return f"{forced} [Agent's original argument: {original}]"

    # -- state --------------------------------------------------------------

    def apply(self, state: RunState, value: TuningDecision) -> None:
        state.tuning_decision = value

    def decision_summary(self, value: TuningDecision) -> str:
        if not value.worthwhile:
            first = value.rationale.strip().split(". ")[0] if value.rationale else ""
            return f"Tuning declined: {first[:200] or 'not worthwhile'}"
        family = value.target_family.value if value.target_family else "unspecified"
        return (
            f"Tune {family} with {value.method.value}: {value.n_trials} trials, "
            f"{value.timeout_seconds}s cap, {len(value.search_space)} "
            f"hyperparameter(s)"
        )


__all__ = [
    "DOWNSTREAM_RESERVE_SECONDS",
    "MAX_TRIALS_CAP",
    "MIN_TUNING_SECONDS",
    "MIN_USEFUL_TRIALS",
    "TuningAgent",
    "TuningBudget",
    "estimate_tuning_budget",
]
