"""Evaluation diagnostics: the measurements behind the quality gate.

The Evaluation Agent decides whether a model ships. It can only do that honestly
if someone has already measured the things that distinguish "0.87 AUC" from
"0.87 AUC that collapses on the smallest customer segment": the train/holdout
gap, whether the probabilities mean anything, how wide the metric's sampling
error is, which slices underperform, what the residuals look like, whether more
data would help, and *where* the model is wrong.

This module computes all of that deterministically and hands it over as text via
:meth:`DiagnosticsBundle.to_prompt`. Every diagnostic is independently guarded —
a failure becomes a note in the bundle, never an exception, because a missing
calibration curve is not a reason to lose a completed training run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.schemas import (
    BiasVarianceDiagnosis,
    CalibrationDiagnosis,
    ConfidenceInterval,
    FairnessSlice,
)
from ..core.state import RunState
from ._metric_bridge import (
    canonical_metric,
    higher_is_better,
    metrics_source,
    resolve_scorer,
    score_predictions,
)
from .explainer import (
    PredictionContext,
    build_prediction_context,
    column_names,
    is_frame,
    row_count,
    take_rows,
)

logger = logging.getLogger(__name__)

# --- runtime budgets -------------------------------------------------------
BOOTSTRAP_RESAMPLES = 300
BOOTSTRAP_MAX_ROWS = 20_000
BOOTSTRAP_MIN_ROWS = 20
CI_LEVEL = 0.95
MAX_CI_METRICS = 3
CALIBRATION_BINS = 10
MIN_SLICE_ROWS = 25
MAX_SLICES_PER_ATTRIBUTE = 12
LEARNING_CURVE_POINTS = 5
LEARNING_CURVE_MAX_ROWS = 5_000
LEARNING_CURVE_FOLDS = 3
LEARNING_CURVE_MIN_SECONDS_LEFT = 45.0
MAX_ERROR_EXAMPLES = 5
ERROR_EXAMPLE_FEATURES = 3

# --- verdict thresholds ----------------------------------------------------
# Scores are first re-oriented so that larger is better (error metrics are
# negated), then the train-to-holdout drop is expressed relatively:
#     rel_gap = (train - holdout) / max(|train|, |holdout|, EPS)
# * rel_gap >= 0.15 with an absolute drop of at least 0.02 -> overfitting.
#   Below that, the drop is within the range ordinary sampling noise produces on
#   a holdout of a few hundred rows.
# * a train score under WEAK_*, without an overfitting-sized gap -> underfitting:
#   the model cannot even fit the data it saw, which is bias rather than variance.
#   The gap is not required to be tiny, because a weak model with a noisy holdout
#   is still a weak model.
# * anything else with both scores present -> good_fit.
# Absolute weakness is only judged for bounded metrics; "is an RMSE of 4.2 bad?"
# has no answer without a baseline, so unbounded metrics never yield an
# underfitting verdict on their own.
OVERFIT_REL_GAP = 0.15
OVERFIT_MIN_ABS_GAP = 0.02
SMALL_REL_GAP = 0.05
WEAK_BOUNDED_SCORE = 0.60
WEAK_R2_SCORE = 0.20
_BOUNDED_UNIT_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "average_precision",
    "pr_auc",
    "f1",
    "f1_macro",
    "f1_weighted",
    "precision",
    "recall",
}
_R2_METRICS = {"r2", "explained_variance"}
_EPS = 1e-9

# Calibration verdict bands: an expected calibration error under 5 points is
# routinely reported as well calibrated; over 10 points the probabilities should
# not be shown to users as risk scores without recalibration.
ECE_WELL_CALIBRATED = 0.05
ECE_MODERATE = 0.10


@dataclass
class DiagnosticsBundle:
    """Every measured fact the Evaluation Agent reasons over.

    ``notes`` records diagnostics that could not be computed and why. It exists
    because the alternative — raising — would discard a whole successful run over
    an optional measurement.
    """

    bias_variance: BiasVarianceDiagnosis = field(default_factory=BiasVarianceDiagnosis)
    calibration: CalibrationDiagnosis = field(default_factory=CalibrationDiagnosis)
    confidence_intervals: list[ConfidenceInterval] = field(default_factory=list)
    fairness: list[FairnessSlice] = field(default_factory=list)
    residual_stats: dict[str, float] = field(default_factory=dict)
    learning_curve: dict[str, list[float]] = field(default_factory=dict)
    error_examples: list[str] = field(default_factory=list)
    primary_metric: str = ""
    partition_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- rendering --------------------------------------------------------

    def to_prompt(self) -> str:
        """Render every diagnostic as compact labelled text for an agent prompt."""
        lines: list[str] = ["## MEASURED MODEL DIAGNOSTICS", ""]
        add = lines.append
        metric = self.primary_metric or "primary metric"

        add(f"### Fit quality (primary metric: {metric})")
        bv = self.bias_variance
        add(f"- train: {_fmt(bv.train_score)}")
        add(f"- validation: {_fmt(bv.validation_score)}")
        add(f"- test: {_fmt(bv.test_score)}")
        add(f"- train-minus-holdout gap: {_fmt(bv.gap)}")
        add(f"- verdict: {bv.verdict}")
        if bv.detail:
            add(f"- detail: {bv.detail}")
        add("")

        if self.partition_metrics:
            add("### Full metric panel by partition")
            for name, panel in self.partition_metrics.items():
                rendered = ", ".join(
                    f"{k}={_fmt(v)}" for k, v in sorted(panel.items()) if k != "n_rows"
                )
                add(f"- {name} (n={int(panel.get('n_rows', 0))}): {rendered or 'n/a'}")
            add("")

        add("### Probability calibration")
        cal = self.calibration
        if not cal.applicable:
            add(f"- not applicable: {cal.verdict or 'model does not emit probabilities'}")
        else:
            add(f"- Brier score: {_fmt(cal.brier_score)} (lower is better)")
            add(
                f"- expected calibration error over {CALIBRATION_BINS} bins: "
                f"{_fmt(cal.expected_calibration_error)}"
            )
            add(f"- verdict: {cal.verdict}")
        add("")

        add(f"### Bootstrap confidence intervals ({int(CI_LEVEL * 100)}%)")
        if not self.confidence_intervals:
            add("- none computed")
        for interval in self.confidence_intervals:
            add(
                f"- {interval.metric}: {_fmt(interval.point_estimate)} "
                f"[{_fmt(interval.lower)}, {_fmt(interval.upper)}] "
                f"(width {_fmt(_width(interval))}, {interval.method})"
            )
        add("")

        add("### Fairness slices")
        if not self.fairness:
            add("- no fairness attributes audited")
        for slice_ in self.fairness:
            add(
                f"- {slice_.attribute}={slice_.slice_value} (n={slice_.n_rows}): "
                f"{slice_.metric_name}={_fmt(slice_.metric_value)} "
                f"(delta vs overall {_fmt(slice_.delta_vs_overall, signed=True)})"
            )
        add("")

        add("### Residuals")
        if not self.residual_stats:
            add("- not applicable (classification) or not computed")
        for key, value in self.residual_stats.items():
            add(f"- {key}: {_fmt(value)}")
        add("")

        add("### Learning curve")
        sizes = self.learning_curve.get("train_sizes", [])
        train_scores = self.learning_curve.get("train_scores", [])
        valid_scores = self.learning_curve.get("validation_scores", [])
        if not sizes:
            add("- not computed")
        for i, size in enumerate(sizes):
            train_value = train_scores[i] if i < len(train_scores) else None
            valid_value = valid_scores[i] if i < len(valid_scores) else None
            add(
                f"- n_train={int(size)}: train={_fmt(train_value)} "
                f"cv_validation={_fmt(valid_value)}"
            )
        if len(valid_scores) >= 2:
            trend = valid_scores[-1] - valid_scores[-2]
            add(
                f"- validation score change over the last step: {_fmt(trend, signed=True)} "
                "(a near-flat tail means more rows of the same kind will not help)"
            )
        add("")

        add("### Worst predictions")
        if not self.error_examples:
            add("- none identified")
        for example in self.error_examples:
            add(f"- {example}")
        add("")

        if self.notes:
            add("### Diagnostic notes and limitations")
            for note in self.notes:
                add(f"- {note}")
            add("")

        return "\n".join(lines)


def _width(interval: ConfidenceInterval) -> float:
    return float(interval.upper - interval.lower)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:+.4g}" if signed else f"{number:.4g}"


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------


@dataclass
class _Predictions:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    n_rows: int


def _collect(ctx: PredictionContext, partition: str) -> _Predictions | None:
    X, y = ctx.partition(partition)
    if row_count(X) == 0 or y is None:
        return None
    y_pred = ctx.predict(X)
    proba = ctx.predict_proba(X)
    return _Predictions(
        y_true=np.asarray(y).ravel(),
        y_pred=np.asarray(y_pred).ravel(),
        y_proba=proba,
        n_rows=row_count(X),
    )


def _panel(ctx: PredictionContext, preds: _Predictions) -> dict[str, float]:
    panel = score_predictions(
        ctx.task, preds.y_true, preds.y_pred, y_proba=preds.y_proba, labels=ctx.labels
    )
    panel["n_rows"] = float(preds.n_rows)
    return panel


def _pick(panel: dict[str, float], metric: str) -> float | None:
    """Read ``metric`` out of a metric panel, tolerating naming variants."""
    if not panel:
        return None
    name = (metric or "").strip().lower()
    for key in (name, name.removeprefix("neg_"), canonical_metric(name)):
        if key and key in panel:
            return float(panel[key])
    return None


# ---------------------------------------------------------------------------
# Bias / variance
# ---------------------------------------------------------------------------


def _bias_variance(
    metric: str, panels: dict[str, dict[str, float]], notes: list[str]
) -> BiasVarianceDiagnosis:
    train = _pick(panels.get("train", {}), metric)
    validation = _pick(panels.get("validation", {}), metric)
    test = _pick(panels.get("test", {}), metric)
    diagnosis = BiasVarianceDiagnosis(
        train_score=train, validation_score=validation, test_score=test
    )
    holdout = validation if validation is not None else test
    holdout_name = "validation" if validation is not None else "test"
    if train is None or holdout is None:
        diagnosis.verdict = "inconclusive"
        diagnosis.detail = (
            f"Could not compare partitions: train={_fmt(train)}, holdout={_fmt(holdout)} "
            f"for metric '{metric}'."
        )
        notes.append("bias/variance inconclusive: a partition score was unavailable.")
        return diagnosis

    hib = higher_is_better(metric)
    train_o = train if hib else -train
    holdout_o = holdout if hib else -holdout
    denominator = max(abs(train_o), abs(holdout_o), _EPS)
    rel_gap = (train_o - holdout_o) / denominator
    diagnosis.gap = float(train - holdout)
    abs_gap = abs(train_o - holdout_o)

    bounded = metric.strip().lower() in _BOUNDED_UNIT_METRICS
    r2_like = metric.strip().lower() in _R2_METRICS
    weak_threshold = (
        WEAK_BOUNDED_SCORE if bounded else WEAK_R2_SCORE if r2_like else None
    )

    if rel_gap >= OVERFIT_REL_GAP and abs_gap >= OVERFIT_MIN_ABS_GAP:
        diagnosis.verdict = "overfitting"
        detail = (
            f"Train {metric} {_fmt(train)} versus {holdout_name} {_fmt(holdout)} is a "
            f"relative drop of {rel_gap * 100:.1f}%, above the {OVERFIT_REL_GAP * 100:.0f}% "
            "threshold: the model is fitting detail that does not generalise."
        )
    elif weak_threshold is not None and train_o < weak_threshold:
        diagnosis.verdict = "underfitting"
        detail = (
            f"Train {metric} is only {_fmt(train)}, below the {weak_threshold:.2f} "
            f"weak-fit threshold, with a {rel_gap * 100:.1f}% gap to {holdout_name}; "
            "the model fits even the rows it saw poorly, which is bias, not variance."
        )
    else:
        diagnosis.verdict = "good_fit"
        detail = (
            f"Train {metric} {_fmt(train)} and {holdout_name} {_fmt(holdout)} differ by "
            f"{rel_gap * 100:.1f}% relative, inside the {OVERFIT_REL_GAP * 100:.0f}% "
            "overfitting threshold."
        )
        if abs(rel_gap) <= SMALL_REL_GAP:
            detail += " That gap is within ordinary holdout sampling noise."
        if weak_threshold is None:
            detail += (
                " Absolute quality was not judged: this metric is unbounded, so a "
                "baseline comparison is required to rule out high bias."
            )
    if test is not None and validation is not None:
        detail += f" Test {metric} is {_fmt(test)}."
    diagnosis.detail = detail
    return diagnosis


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = CALIBRATION_BINS
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = confidence.shape[0]
    error = 0.0
    for i in range(bins):
        lower, upper = edges[i], edges[i + 1]
        mask = (confidence > lower) & (confidence <= upper) if i else (confidence <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error)


def _calibration(
    state: RunState,
    ctx: PredictionContext,
    preds: _Predictions | None,
    partition: str,
    notes: list[str],
) -> CalibrationDiagnosis:
    if not ctx.is_classification:
        return CalibrationDiagnosis(
            applicable=False, verdict="Calibration applies to classifiers only."
        )
    if preds is None or preds.y_proba is None:
        return CalibrationDiagnosis(
            applicable=False,
            verdict="The fitted model does not expose predict_proba, so probability "
            "calibration cannot be measured.",
        )
    try:
        from sklearn.metrics import brier_score_loss

        proba = np.asarray(preds.y_proba, dtype=float)
        labels = list(ctx.labels or np.unique(preds.y_true).tolist())
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])

        if proba.shape[1] == 2:
            index = min(ctx.positive_index, proba.shape[1] - 1)
            positive_label = labels[index] if index < len(labels) else labels[-1]
            scores = proba[:, index]
            observed = (preds.y_true.astype(object) == positive_label).astype(float)
            brier = float(brier_score_loss(observed, scores))
            ece = _expected_calibration_error(scores, observed)
            mean_predicted = float(scores.mean())
            base_rate = float(observed.mean())
            bias_text = (
                "over-confident (predicted rate exceeds the observed rate)"
                if mean_predicted - base_rate > 0.02
                else "under-confident (predicted rate below the observed rate)"
                if base_rate - mean_predicted > 0.02
                else "unbiased on average"
            )
            detail = (
                f"mean predicted P({positive_label})={mean_predicted:.3f} versus observed "
                f"{base_rate:.3f} — {bias_text}"
            )
        else:
            confidence = proba.max(axis=1)
            predicted_index = proba.argmax(axis=1)
            predicted = np.asarray([labels[i] for i in predicted_index], dtype=object)
            observed = (predicted == preds.y_true.astype(object)).astype(float)
            brier = float(brier_score_loss(preds.y_true, proba, labels=labels))
            ece = _expected_calibration_error(confidence, observed)
            detail = (
                f"mean top-class confidence {float(confidence.mean()):.3f} versus accuracy "
                f"{float(observed.mean()):.3f}"
            )

        if ece <= ECE_WELL_CALIBRATED:
            band = "well calibrated"
        elif ece <= ECE_MODERATE:
            band = "moderately calibrated"
        else:
            band = "poorly calibrated — recalibrate before using the scores as probabilities"
        return CalibrationDiagnosis(
            applicable=True,
            brier_score=brier,
            expected_calibration_error=ece,
            verdict=(
                f"{band} on the {partition} partition (ECE {ece:.3f} over "
                f"{CALIBRATION_BINS} bins, Brier {brier:.4f}); {detail}."
            ),
        )
    except Exception as exc:
        state.add_warning(f"calibration diagnostics failed: {exc}")
        notes.append(f"calibration not computed: {exc}.")
        return CalibrationDiagnosis(
            applicable=False, verdict=f"Calibration could not be measured: {exc}"
        )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _ci_metrics(state: RunState, panel: dict[str, float], primary: str) -> list[str]:
    wanted = [primary if primary in panel else canonical_metric(primary)]
    secondary = list(state.problem.secondary_metrics) if state.problem else []
    for name in secondary:
        key = (name or "").strip().lower()
        key = key if key in panel else canonical_metric(key)
        if key and key in panel and key not in wanted:
            wanted.append(key)
    if len(wanted) < MAX_CI_METRICS:
        for fallback in ("accuracy", "f1", "f1_macro", "r2", "mae", "rmse", "roc_auc"):
            if len(wanted) >= MAX_CI_METRICS:
                break
            if fallback in panel and fallback not in wanted:
                wanted.append(fallback)
    return [name for name in wanted[:MAX_CI_METRICS] if name in panel]


def _bootstrap_intervals(
    state: RunState,
    ctx: PredictionContext,
    preds: _Predictions | None,
    panel: dict[str, float],
    primary: str,
    notes: list[str],
) -> list[ConfidenceInterval]:
    if preds is None:
        notes.append("confidence intervals not computed: no evaluation predictions.")
        return []
    if preds.n_rows < BOOTSTRAP_MIN_ROWS:
        notes.append(
            f"confidence intervals not computed: only {preds.n_rows} evaluation rows "
            f"(minimum {BOOTSTRAP_MIN_ROWS})."
        )
        return []
    wanted = _ci_metrics(state, panel, primary)
    if not wanted:
        notes.append("confidence intervals not computed: no metric available to resample.")
        return []

    try:
        rng = np.random.default_rng(ctx.random_state)
        n = preds.n_rows
        base = np.arange(n)
        if n > BOOTSTRAP_MAX_ROWS:
            base = np.sort(rng.choice(n, size=BOOTSTRAP_MAX_ROWS, replace=False))
            notes.append(
                f"bootstrap sampled {BOOTSTRAP_MAX_ROWS:,} of {n:,} evaluation rows for runtime."
            )
        collected: dict[str, list[float]] = {name: [] for name in wanted}
        for _ in range(BOOTSTRAP_RESAMPLES):
            draw = rng.choice(base, size=base.shape[0], replace=True)
            resampled = score_predictions(
                ctx.task,
                preds.y_true[draw],
                preds.y_pred[draw],
                y_proba=None if preds.y_proba is None else np.asarray(preds.y_proba)[draw],
                labels=ctx.labels,
            )
            for name in wanted:
                value = resampled.get(name)
                if value is not None and np.isfinite(value):
                    collected[name].append(float(value))

        alpha = (1.0 - CI_LEVEL) / 2.0
        intervals: list[ConfidenceInterval] = []
        for name in wanted:
            draws = collected[name]
            if len(draws) < 10:
                notes.append(f"confidence interval for '{name}' skipped: too few valid resamples.")
                continue
            array = np.asarray(draws, dtype=float)
            intervals.append(
                ConfidenceInterval(
                    metric=name,
                    point_estimate=float(panel[name]),
                    lower=float(np.quantile(array, alpha)),
                    upper=float(np.quantile(array, 1.0 - alpha)),
                    level=CI_LEVEL,
                    method=f"percentile bootstrap, {len(draws)} resamples",
                )
            )
        return intervals
    except Exception as exc:
        state.add_warning(f"bootstrap confidence intervals failed: {exc}")
        notes.append(f"confidence intervals not computed: {exc}.")
        return []


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------


def _attribute_series(state: RunState, X: Any, attribute: str) -> np.ndarray | None:
    """Values of ``attribute`` aligned to the evaluation rows, or ``None``."""
    if is_frame(X) and attribute in set(column_names(X)):
        return np.asarray(X[attribute].to_numpy(), dtype=object)
    if not is_frame(X):
        return None
    for frame in (state.feature_frame, state.working_df, state.raw_df):
        if frame is None or not hasattr(frame, "columns"):
            continue
        if attribute not in {str(c) for c in frame.columns}:
            continue
        try:
            return np.asarray(frame.loc[X.index, attribute].to_numpy(), dtype=object)
        except Exception as exc:
            logger.debug("fairness alignment failed for %s: %s", attribute, exc)
    return None


def _fairness(
    state: RunState,
    ctx: PredictionContext,
    preds: _Predictions | None,
    partition: str,
    primary: str,
    overall: float | None,
    notes: list[str],
) -> list[FairnessSlice]:
    """Primary metric per slice of each configured fairness attribute."""
    attributes = list(state.config.fairness_attributes or [])
    if not attributes:
        return []
    if preds is None or overall is None:
        notes.append("fairness audit skipped: no evaluation predictions or overall score.")
        return []

    X, _ = ctx.partition(partition)
    out: list[FairnessSlice] = []
    for attribute in attributes:
        values = _attribute_series(state, X, attribute)
        if values is None:
            state.add_warning(
                f"fairness attribute '{attribute}' is not present in the evaluated data; "
                "skipping that audit."
            )
            notes.append(f"fairness attribute '{attribute}' not found; skipped.")
            continue
        if values.shape[0] != preds.n_rows:
            notes.append(
                f"fairness attribute '{attribute}' has {values.shape[0]} values for "
                f"{preds.n_rows} rows; skipped."
            )
            continue
        try:
            unique, counts = np.unique(values.astype(str), return_counts=True)
            order = np.argsort(counts)[::-1][:MAX_SLICES_PER_ATTRIBUTE]
            skipped_small = 0
            for position in order:
                slice_value = str(unique[position])
                mask = values.astype(str) == slice_value
                n_rows = int(mask.sum())
                if n_rows < MIN_SLICE_ROWS:
                    # A metric on a handful of rows is sampling noise, and
                    # reporting it as a disparity would be actively misleading.
                    skipped_small += 1
                    continue
                panel = score_predictions(
                    ctx.task,
                    preds.y_true[mask],
                    preds.y_pred[mask],
                    y_proba=None
                    if preds.y_proba is None
                    else np.asarray(preds.y_proba)[mask],
                    labels=ctx.labels,
                )
                value = _pick(panel, primary)
                if value is None:
                    notes.append(
                        f"fairness slice {attribute}={slice_value}: '{primary}' could not be "
                        "computed (likely a single class in the slice)."
                    )
                    continue
                out.append(
                    FairnessSlice(
                        attribute=attribute,
                        slice_value=slice_value,
                        n_rows=n_rows,
                        metric_name=primary,
                        metric_value=float(value),
                        delta_vs_overall=float(value - overall),
                    )
                )
            if skipped_small:
                notes.append(
                    f"fairness: {skipped_small} slice(s) of '{attribute}' had fewer than "
                    f"{MIN_SLICE_ROWS} rows and were skipped as noise."
                )
        except Exception as exc:
            state.add_warning(f"fairness audit for '{attribute}' failed: {exc}")
            notes.append(f"fairness audit for '{attribute}' failed: {exc}.")
    return out


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


def _residuals(
    state: RunState,
    ctx: PredictionContext,
    preds: _Predictions | None,
    notes: list[str],
) -> dict[str, float]:
    """Residual shape, normality, and a heteroscedasticity signal."""
    if ctx.is_classification:
        return {}
    if preds is None:
        notes.append("residual statistics not computed: no evaluation predictions.")
        return {}
    try:
        from scipy import stats

        y_true = np.asarray(preds.y_true, dtype=float)
        y_pred = np.asarray(preds.y_pred, dtype=float)
        residuals = y_true - y_pred
        finite = np.isfinite(residuals)
        residuals = residuals[finite]
        y_pred = y_pred[finite]
        if residuals.size < 3:
            notes.append("residual statistics not computed: fewer than 3 finite residuals.")
            return {}

        out: dict[str, float] = {
            "n": float(residuals.size),
            "mean": float(residuals.mean()),
            "std": float(residuals.std(ddof=1)) if residuals.size > 1 else 0.0,
            "mean_absolute": float(np.abs(residuals).mean()),
            "max_absolute": float(np.abs(residuals).max()),
            "skew": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals)),
        }
        # A non-zero mean residual is bias: the model is systematically high or
        # low, which no amount of variance reduction fixes.
        out["bias_as_fraction_of_std"] = (
            float(out["mean"] / out["std"]) if out["std"] > _EPS else 0.0
        )
        if residuals.size >= 8:
            try:
                statistic, p_value = stats.normaltest(residuals)
                out["normality_statistic"] = float(statistic)
                out["normality_p_value"] = float(p_value)
            except Exception as exc:
                notes.append(f"residual normality test failed: {exc}.")
        if residuals.size >= 5 and np.std(y_pred) > _EPS:
            try:
                corr, p_value = stats.pearsonr(np.abs(residuals), y_pred)
                out["heteroscedasticity_corr_abs_resid_vs_pred"] = float(corr)
                out["heteroscedasticity_p_value"] = float(p_value)
            except Exception as exc:
                notes.append(f"heteroscedasticity check failed: {exc}.")
        return out
    except Exception as exc:
        state.add_warning(f"residual diagnostics failed: {exc}")
        notes.append(f"residual statistics not computed: {exc}.")
        return {}


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------


def _stack(a: Any, b: Any) -> Any:
    """Concatenate two partitions of the same kind, or return the first."""
    if row_count(b) == 0:
        return a
    if row_count(a) == 0:
        return b
    if is_frame(a) and is_frame(b):
        import pandas as pd

        return pd.concat([a, b], axis=0)
    if hasattr(a, "iloc") and hasattr(b, "iloc"):
        import pandas as pd

        return pd.concat([a, b], axis=0)
    try:
        first, second = np.asarray(a), np.asarray(b)
        return np.vstack([first, second]) if first.ndim > 1 else np.concatenate([first, second])
    except Exception:
        return a


def _row_nanmean(scores: Any) -> np.ndarray:
    """Per-row mean ignoring NaN, returning NaN for an all-NaN row without warning."""
    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    counts = np.sum(np.isfinite(arr), axis=1)
    totals = np.nansum(np.where(np.isfinite(arr), arr, np.nan), axis=1)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    nonempty = counts > 0
    out[nonempty] = totals[nonempty] / counts[nonempty]
    return out


def _learning_curve(
    state: RunState, ctx: PredictionContext, primary: str, notes: list[str]
) -> dict[str, list[float]]:
    """Cross-validated score against training-set size over ~5 sizes."""
    if state.time_remaining and state.time_remaining < LEARNING_CURVE_MIN_SECONDS_LEFT:
        notes.append(
            "learning curve skipped: less than "
            f"{LEARNING_CURVE_MIN_SECONDS_LEFT:.0f}s of run budget remaining."
        )
        return {}
    try:
        from sklearn.base import clone
        from sklearn.model_selection import KFold, StratifiedKFold, learning_curve

        X = _stack(ctx.X_train, ctx.X_valid)
        y = _stack(ctx.y_train, ctx.y_valid)
        n = row_count(X)
        if n < LEARNING_CURVE_FOLDS * 4 or row_count(y) != n:
            notes.append(f"learning curve skipped: only {n} usable training rows.")
            return {}

        rng = np.random.default_rng(ctx.random_state)
        if n > LEARNING_CURVE_MAX_ROWS:
            positions = np.sort(
                rng.choice(n, size=LEARNING_CURVE_MAX_ROWS, replace=False)
            )
            X, y = take_rows(X, positions), take_rows(y, positions)
            notes.append(
                f"learning curve sampled {LEARNING_CURVE_MAX_ROWS:,} of {n:,} training rows."
            )

        y_array = np.asarray(y).ravel()
        if ctx.is_classification:
            _, class_counts = np.unique(y_array, return_counts=True)
            folds = int(max(2, min(LEARNING_CURVE_FOLDS, class_counts.min())))
            splitter: Any = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=ctx.random_state
            )
        else:
            splitter = KFold(
                n_splits=LEARNING_CURVE_FOLDS, shuffle=True, random_state=ctx.random_state
            )

        scorer, scorer_name, negated = resolve_scorer(
            primary, ctx.task, pos_label=ctx.scorer_pos_label
        )
        sizes = np.linspace(0.2, 1.0, LEARNING_CURVE_POINTS)
        # n_jobs is left sequential on purpose: this refits roughly folds x sizes
        # models, and on the row budget above, process start-up costs more than
        # the parallelism saves.
        train_sizes, train_scores, valid_scores = learning_curve(
            clone(ctx.estimator),
            X,
            y_array,
            cv=splitter,
            train_sizes=sizes,
            scoring=scorer if scorer is not None else None,
            n_jobs=None,
            error_score=np.nan,
        )
        sign = -1.0 if negated else 1.0
        # error_score=np.nan means a fold that raised leaves NaN behind. Averaging
        # an all-NaN row yields NaN, and reporting "train=nan" as a learning curve
        # is worse than reporting no curve at all, so keep only finite points.
        with np.errstate(invalid="ignore"):
            train_mean = sign * _row_nanmean(train_scores)
            valid_mean = sign * _row_nanmean(valid_scores)
        keep = np.isfinite(train_mean) & np.isfinite(valid_mean)
        if not bool(keep.any()):
            notes.append(
                "learning curve not computed: every fold failed to score with "
                f"{scorer_name or 'estimator.score'}."
            )
            return {}
        if not bool(keep.all()):
            notes.append(
                f"learning curve: {int((~keep).sum())} of {keep.size} training sizes "
                "failed to score and were dropped."
            )
        notes.append(
            f"learning curve: {LEARNING_CURVE_POINTS} sizes x {splitter.get_n_splits()}-fold CV, "
            f"scoring={scorer_name or 'estimator.score'}."
        )
        return {
            "train_sizes": [float(v) for v in np.asarray(train_sizes)[keep]],
            "train_scores": [float(v) for v in train_mean[keep]],
            "validation_scores": [float(v) for v in valid_mean[keep]],
        }
    except Exception as exc:
        state.add_warning(f"learning curve failed: {exc}")
        notes.append(f"learning curve not computed: {exc}.")
        return {}


# ---------------------------------------------------------------------------
# Error examples
# ---------------------------------------------------------------------------


def _context_features(state: RunState, X: Any) -> list[str]:
    """A few columns worth printing beside a bad prediction."""
    available = column_names(X)
    if not is_frame(X) or not available:
        return []
    ordered: list[str] = []
    report = state.explainability
    if report is not None:
        for attribution in report.global_attributions:
            if attribution.feature in available and attribution.feature not in ordered:
                ordered.append(attribution.feature)
    for name in available:
        if len(ordered) >= ERROR_EXAMPLE_FEATURES:
            break
        if name not in ordered:
            ordered.append(name)
    return ordered[:ERROR_EXAMPLE_FEATURES]


def _row_label(X: Any, position: int) -> str:
    if is_frame(X):
        try:
            return str(X.index[position])
        except Exception:
            return str(position)
    return str(position)


def _row_context(X: Any, position: int, features: list[str]) -> str:
    if not features or not is_frame(X):
        return ""
    parts: list[str] = []
    try:
        row = X.iloc[position]
        for feature in features:
            value = row[feature]
            parts.append(f"{feature}={_fmt(value) if isinstance(value, (int, float, np.floating)) else value}")
    except Exception:
        return ""
    return "; ".join(parts)


def _error_examples(
    state: RunState,
    ctx: PredictionContext,
    preds: _Predictions | None,
    partition: str,
    notes: list[str],
) -> list[str]:
    """The worst-predicted rows, rendered as short readable strings."""
    if preds is None:
        notes.append("error examples not computed: no evaluation predictions.")
        return []
    X, _ = ctx.partition(partition)
    try:
        features = _context_features(state, X)
        out: list[str] = []
        if ctx.is_classification:
            wrong = np.flatnonzero(
                preds.y_true.astype(object) != preds.y_pred.astype(object)
            )
            if wrong.size == 0:
                return [
                    f"No misclassified rows in the {partition} partition "
                    f"({preds.n_rows} rows) — inspect for leakage if this is unexpected."
                ]
            if preds.y_proba is not None:
                proba = np.asarray(preds.y_proba, dtype=float)
                confidence = proba.max(axis=1)
                ranked = wrong[np.argsort(confidence[wrong])[::-1]]
            else:
                confidence = None
                ranked = wrong
            for position in ranked[:MAX_ERROR_EXAMPLES]:
                pieces = [
                    f"row {_row_label(X, int(position))}: actual="
                    f"{preds.y_true[position]}, predicted={preds.y_pred[position]}"
                ]
                if confidence is not None:
                    pieces[0] += f" with confidence {confidence[position]:.2f}"
                context = _row_context(X, int(position), features)
                if context:
                    pieces.append(context)
                out.append(" | ".join(pieces))
            out.append(
                f"{wrong.size} of {preds.n_rows} {partition} rows misclassified "
                f"({wrong.size / max(preds.n_rows, 1) * 100:.1f}%)."
            )
            return out

        residuals = np.asarray(preds.y_true, dtype=float) - np.asarray(
            preds.y_pred, dtype=float
        )
        ranked = np.argsort(np.abs(residuals))[::-1]
        for position in ranked[:MAX_ERROR_EXAMPLES]:
            pieces = [
                f"row {_row_label(X, int(position))}: actual={_fmt(preds.y_true[position])}, "
                f"predicted={_fmt(preds.y_pred[position])}, residual="
                f"{_fmt(residuals[position], signed=True)}"
            ]
            context = _row_context(X, int(position), features)
            if context:
                pieces.append(context)
            out.append(" | ".join(pieces))
        return out
    except Exception as exc:
        state.add_warning(f"error-example extraction failed: {exc}")
        notes.append(f"error examples not computed: {exc}.")
        return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compute_diagnostics(state: RunState) -> DiagnosticsBundle:
    """Measure everything the Evaluation Agent needs to grade the model.

    Scores each partition, diagnoses bias/variance and calibration, bootstraps
    confidence intervals, audits configured fairness attributes, describes the
    residuals, traces a learning curve, and extracts the worst predictions. Each
    diagnostic is guarded: a failure is recorded in ``bundle.notes`` (and as a run
    warning) instead of propagating.

    Args:
        state: The run blackboard, read for the fitted model, splits, primary
            metric, and ``config.fairness_attributes``.

    Returns:
        A :class:`DiagnosticsBundle`. Also stored at
        ``state.extras['diagnostics']`` for downstream agents.
    """
    started = time.perf_counter()
    notes: list[str] = [f"metric functions: {metrics_source()}"]
    primary = canonical_metric(state.primary_metric or "")
    bundle = DiagnosticsBundle(primary_metric=primary or "unknown", notes=notes)

    ctx = build_prediction_context(state)
    if ctx is None:
        notes.append(
            "no diagnostics were computed: no fitted model or usable data splits were "
            "available on the run state."
        )
        state.extras["diagnostics"] = bundle
        return bundle
    notes.extend(ctx.notes)

    if not primary:
        from ._metric_bridge import primary_metric_for

        primary = primary_metric_for(ctx.task)
        bundle.primary_metric = primary
        notes.append(f"no primary metric on the run state; defaulted to '{primary}'.")

    panels: dict[str, dict[str, float]] = {}
    collected: dict[str, _Predictions | None] = {}
    for partition in ("train", "validation", "test"):
        try:
            preds = _collect(ctx, partition)
        except Exception as exc:
            state.add_warning(f"prediction on the {partition} partition failed: {exc}")
            notes.append(f"{partition} partition could not be scored: {exc}.")
            preds = None
        collected[partition] = preds
        if preds is not None:
            panels[partition] = _panel(ctx, preds)
    bundle.partition_metrics = panels

    bundle.bias_variance = _bias_variance(primary, panels, notes)

    # The evaluation partition is the untouched test set when it exists, since
    # every number below is quoted as the model's expected field performance.
    eval_partition = "test" if collected.get("test") is not None else ctx.eval_partition
    eval_preds = collected.get(eval_partition)
    eval_panel = panels.get(eval_partition, {})
    notes.append(f"holdout diagnostics measured on the {eval_partition} partition.")

    bundle.calibration = _calibration(state, ctx, eval_preds, eval_partition, notes)
    bundle.confidence_intervals = _bootstrap_intervals(
        state, ctx, eval_preds, eval_panel, primary, notes
    )
    bundle.fairness = _fairness(
        state,
        ctx,
        eval_preds,
        eval_partition,
        primary,
        _pick(eval_panel, primary),
        notes,
    )
    bundle.residual_stats = _residuals(state, ctx, eval_preds, notes)
    bundle.learning_curve = _learning_curve(state, ctx, primary, notes)
    bundle.error_examples = _error_examples(state, ctx, eval_preds, eval_partition, notes)

    notes.append(f"diagnostics pass took {time.perf_counter() - started:.1f}s.")
    state.extras["diagnostics"] = bundle
    return bundle


__all__ = ["DiagnosticsBundle", "compute_diagnostics"]
