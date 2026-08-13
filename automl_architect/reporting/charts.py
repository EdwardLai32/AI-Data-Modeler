"""Deterministic chart rendering for every :class:`ChartKind`.

The Visualization Agent chooses *what* to plot and says why; this module does the
plotting. Nothing here asks a model anything: every mark comes from the profile,
the splits, the experiment log, the explainability report, or the fitted pipeline
already sitting on :class:`~automl_architect.core.state.RunState`.

Two design rules matter more than the plotting details:

* **Every chart is guarded individually.** A chart that cannot be built records
  its reason on its own :class:`ChartArtifact` and the rest still render. A run
  never loses its whole visual layer because one model lacked
  ``predict_proba``.
* **Agent-supplied column names are never trusted.** Anything that reaches
  pandas has been filtered against the real frame's columns first, with a
  profile-driven fallback when the agent named nothing usable.

Each chart is written twice: a standalone interactive HTML file (plotly.js from
the CDN) and, when kaleido is importable, a PNG for the PDF and PowerPoint
renderers. The figure JSON is written too, which is what lets the dashboard be
rebuilt without re-running the pipeline.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.schemas import (
    ChartArtifact,
    ChartKind,
    ChartSpec,
    TaskType,
    VisualizationBundle,
    VisualizationPlan,
    params_to_dict,
)
from . import theme
from .common import (
    enum_value,
    experiment_label,
    fmt_number,
    metric_direction,
    successful_experiments,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import RunState

logger = logging.getLogger(__name__)

MAX_IMPORTANCE_FEATURES = 20
MAX_HEATMAP_COLUMNS = 16
MAX_CATEGORY_BARS = 15
MAX_MISSINGNESS_BARS = 25
MAX_CURVE_CLASSES = 8
MAX_SCATTER_POINTS = 6000
MAX_PREDICT_ROWS = 50_000
LEARNING_CURVE_ROW_LIMIT = 8000


class ChartUnavailable(RuntimeError):
    """This chart cannot be built from what the run actually measured.

    Distinct from an unexpected exception: it is the *expected* negative case
    (no probabilities, no temporal column, regression-only chart on a classifier)
    and is reported as a note on the artifact rather than as a defect.
    """


@dataclass
class _Built:
    """A rendered figure plus the extras the dashboard and reports want."""

    figure: Any
    note: str = ""
    table: tuple[list[str], list[list[str]]] | None = None
    png_override: str | None = None


# ---------------------------------------------------------------------------
# data access helpers
# ---------------------------------------------------------------------------


def _frame(state: RunState) -> pd.DataFrame:
    frame = state.df
    if frame is None or not hasattr(frame, "columns"):
        raise ChartUnavailable("no dataframe is loaded on the run state")
    if len(frame) == 0:
        raise ChartUnavailable("the dataframe is empty")
    return frame


def _known_columns(frame: pd.DataFrame, names: list[str] | None) -> list[str]:
    """Filter agent-supplied column names against the real frame.

    Hard rule: nothing an agent named reaches pandas without passing through
    here first.
    """
    if not names:
        return []
    actual = {str(c) for c in frame.columns}
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = str(name)
        if key in actual and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _numeric_columns(frame: pd.DataFrame, candidates: list[str] | None = None) -> list[str]:
    pool = candidates if candidates else [str(c) for c in frame.columns]
    return [c for c in pool if pd.api.types.is_numeric_dtype(frame[c])]


def _categorical_columns(
    frame: pd.DataFrame, candidates: list[str] | None = None
) -> list[str]:
    pool = candidates if candidates else [str(c) for c in frame.columns]
    out = []
    for column in pool:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        out.append(column)
    return out


def _target_name(state: RunState) -> str | None:
    target = state.target
    frame = state.df
    if target and frame is not None and target in set(map(str, frame.columns)):
        return target
    return None


def _feature_pool(state: RunState, frame: pd.DataFrame) -> list[str]:
    """Model-input columns, with the target and known identifiers removed."""
    target = _target_name(state)
    drop = {target} if target else set()
    profile = state.profile
    if profile is not None:
        drop |= set(profile.identifier_columns) | set(profile.constant_columns)
    return [str(c) for c in frame.columns if str(c) not in drop]


def _informative_numeric(state: RunState, frame: pd.DataFrame, limit: int) -> list[str]:
    """Pick numeric columns worth plotting when the agent named none.

    Preference order: strongest measured target correlations, then widest
    dispersion. Both come from the deterministic profile where available.
    """
    numeric = _numeric_columns(frame, _feature_pool(state, frame))
    if not numeric:
        return []
    profile = state.profile
    ranked: list[str] = []
    if profile is not None:
        for pair in profile.target_correlations:
            for candidate in (pair.left, pair.right):
                if candidate in numeric and candidate not in ranked:
                    ranked.append(candidate)
    if len(ranked) < limit:
        spread = (
            frame[numeric]
            .std(numeric_only=True)
            .abs()
            .sort_values(ascending=False)
        )
        for name in spread.index:
            key = str(name)
            if key not in ranked:
                ranked.append(key)
    return ranked[:limit]


def _sample(frame: Any, limit: int, seed: int) -> Any:
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=seed)


def _stable_labels(*arrays: Any) -> list[Any]:
    """Union of the values in ``arrays``, in a deterministic order.

    Confusion matrices are unreadable if the label order shifts between runs, so
    sorting is attempted first and a string sort is the fallback for mixed types.
    """
    values: list[Any] = []
    seen: set[str] = set()
    for array in arrays:
        if array is None:
            continue
        for value in pd.unique(pd.Series(np.asarray(array).ravel())):
            key = repr(value)
            if key not in seen:
                seen.add(key)
                values.append(value)
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def _display_label(state: RunState, value: Any) -> str:
    """Human-facing name for an encoded class value."""
    encoder = state.label_encoder
    classes = getattr(encoder, "classes_", None)
    if classes is not None:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return str(value)
        if 0 <= index < len(classes):
            return str(classes[index])
    return str(value)


class _ModelView:
    """Lazily computed, cached predictions for the held-out split.

    Built once per :func:`render_charts` call so that ROC, PR, confusion matrix,
    residual, and calibration charts share one ``predict``/``predict_proba``
    pass instead of paying for five.
    """

    def __init__(self, state: RunState) -> None:
        self._state = state
        self._loaded = False
        self.error: str | None = None
        self.split_name = ""
        self.X: Any = None
        self.y_true: Any = None
        self.y_pred: Any = None
        self.scores: Any = None
        self.score_kind = ""
        self.classes: list[Any] = []

    @property
    def estimator(self) -> Any:
        return self._state.best_pipeline or self._state.best_model

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        state = self._state
        estimator = self.estimator
        if estimator is None:
            self.error = "no fitted model is available on the run state"
            return
        splits = state.splits
        for name, features, labels in (
            ("test", splits.X_test, splits.y_test),
            ("validation", splits.X_valid, splits.y_valid),
            ("train", splits.X_train, splits.y_train),
        ):
            if features is not None and labels is not None and len(features) > 0:
                self.split_name = name
                self.X = features
                self.y_true = np.asarray(labels)
                break
        if self.X is None:
            self.error = "no materialised splits to score"
            return
        if len(self.X) > MAX_PREDICT_ROWS:
            self.X = self.X[:MAX_PREDICT_ROWS]
            self.y_true = self.y_true[:MAX_PREDICT_ROWS]
        try:
            self.y_pred = np.asarray(estimator.predict(self.X))
        except Exception as exc:  # a broken estimator must not kill every chart
            self.error = f"predict() failed: {type(exc).__name__}: {exc}"
            return
        raw_classes = getattr(estimator, "classes_", None)
        self.classes = (
            list(raw_classes) if raw_classes is not None else _stable_labels(self.y_true)
        )
        self._load_scores(estimator)

    def _load_scores(self, estimator: Any) -> None:
        if hasattr(estimator, "predict_proba"):
            try:
                self.scores = np.asarray(estimator.predict_proba(self.X))
                self.score_kind = "proba"
                return
            except Exception as exc:
                logger.debug("predict_proba failed: %s", exc)
        if hasattr(estimator, "decision_function"):
            try:
                self.scores = np.asarray(estimator.decision_function(self.X))
                self.score_kind = "decision"
            except Exception as exc:
                logger.debug("decision_function failed: %s", exc)

    def require(self) -> None:
        """Raise :class:`ChartUnavailable` unless predictions exist."""
        self._load()
        if self.error:
            raise ChartUnavailable(self.error)

    def require_scores(self, *, need_proba: bool = False) -> Any:
        """Return the score matrix, or explain why the chart must be skipped."""
        self.require()
        if self.scores is None:
            raise ChartUnavailable(
                "the fitted model exposes neither predict_proba nor "
                "decision_function, so no score-based curve can be drawn"
            )
        if need_proba and self.score_kind != "proba":
            raise ChartUnavailable(
                "calibration needs calibrated probabilities and this model has no "
                "predict_proba; decision-function scores are not on a 0-1 scale"
            )
        return self.scores

    def positive_index(self, positive: str | None) -> int:
        """Index of the positive class in ``classes``, defaulting to the last."""
        if positive:
            for index, value in enumerate(self.classes):
                if str(value) == str(positive) or _display_label(
                    self._state, value
                ) == str(positive):
                    return index
        return len(self.classes) - 1 if self.classes else 0

    def column_for(self, index: int) -> np.ndarray:
        """Score column for one class, handling 1-D binary score vectors."""
        scores = np.asarray(self.scores)
        if scores.ndim == 1:
            return scores if index >= 1 else -scores
        if scores.shape[1] == 1:
            flat = scores[:, 0]
            return flat if index >= 1 else 1.0 - flat
        return scores[:, min(index, scores.shape[1] - 1)]


def _is_classification(state: RunState) -> bool:
    task = state.task_type
    return bool(task and task.is_classification)


def _is_regression(state: RunState) -> bool:
    task = state.task_type
    return task in (TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING)


def _importance_pairs(state: RunState) -> list[tuple[str, float]]:
    """Feature importances from the explainability report, then the model."""
    report = state.explainability
    if report is not None:
        for attributions in (report.global_attributions, report.permutation_importance):
            if attributions:
                return [(a.feature, float(a.importance)) for a in attributions]
    estimator = state.best_pipeline or state.best_model
    model = estimator
    steps = getattr(estimator, "steps", None)
    if steps:
        model = steps[-1][1]
    names = list(state.feature_names or [])
    if not names:
        names = [str(c) for c in getattr(state.splits.X_train, "columns", [])]
    values: Any = getattr(model, "feature_importances_", None)
    if values is None:
        coef = getattr(model, "coef_", None)
        if coef is not None:
            array = np.asarray(coef)
            values = np.abs(array).mean(axis=0) if array.ndim > 1 else np.abs(array)
    if values is None:
        return []
    values = np.asarray(values, dtype=float).ravel()
    if names and len(names) != len(values):
        names = [f"feature_{i}" for i in range(len(values))]
    if not names:
        names = [f"feature_{i}" for i in range(len(values))]
    total = float(np.nansum(np.abs(values))) or 1.0
    return [(names[i], float(abs(values[i])) / total) for i in range(len(values))]


# ---------------------------------------------------------------------------
# mark helpers
# ---------------------------------------------------------------------------


def _marker(color: str, *, size: int = 8, opacity: float = 0.85) -> dict[str, Any]:
    """A dot with a 2px surface ring, so overlapping points stay countable."""
    return {
        "size": size,
        "color": color,
        "opacity": opacity,
        "line": {"width": 2, "color": theme.SURFACE},
    }


def _bar_marker(color: str | list[str], pattern: list[str] | None = None) -> dict[str, Any]:
    """A bar fill with a 2px surface gap instead of a border."""
    marker: dict[str, Any] = {
        "color": color,
        "line": {"width": 2, "color": theme.SURFACE},
    }
    if pattern:
        marker["pattern"] = {
            "shape": pattern,
            "solidity": 0.32,
            "size": 7,
            "fgcolor": theme.SURFACE,
        }
    return marker


def _line(color: str, width: int = 2, dash: str | None = None) -> dict[str, Any]:
    line: dict[str, Any] = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    return line


def _value_text(values: Any, digits: int = 3) -> list[str]:
    return [fmt_number(v, digits) for v in values]


def _bargap(n_categories: int) -> float:
    """Wider gaps for few categories: three full-width blocks read as a poster."""
    if n_categories <= 3:
        return 0.5
    if n_categories <= 6:
        return 0.36
    return 0.28


def _bar_thickness(n_bars: int, figure_height: int, target_px: int = 44) -> float:
    """Trace width (in category units) that keeps a bar around ``target_px`` thick.

    ``bargap`` is a fraction of the slot, so a chart with one or two categories
    gets absurdly thick bars however the gap is set. Expressing the target in
    pixels and converting back is what keeps a one-column missingness chart
    looking like the twenty-column version.
    """
    plot_height = max(120, figure_height - 164)
    return min(0.75, max(0.08, target_px * max(n_bars, 1) / plot_height))


# ---------------------------------------------------------------------------
# builders — data & quality
# ---------------------------------------------------------------------------


def _build_correlation_heatmap(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    frame = _frame(state)
    params = params_to_dict(spec.parameters)
    limit = int(params.get("max_columns", MAX_HEATMAP_COLUMNS))
    chosen = _numeric_columns(frame, _known_columns(frame, spec.columns))
    if len(chosen) < 2:
        chosen = _informative_numeric(state, frame, limit)
        target = _target_name(state)
        if target and pd.api.types.is_numeric_dtype(frame[target]) and target not in chosen:
            chosen = [target] + chosen
    chosen = chosen[:limit]
    if len(chosen) < 2:
        raise ChartUnavailable("fewer than two numeric columns are available")
    matrix = frame[chosen].corr(numeric_only=True)
    labels = [str(c) for c in matrix.columns]
    values = matrix.to_numpy(dtype=float)
    annotate = len(labels) <= 10
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=theme.diverging_scale(),
            colorbar=theme.colorbar("Pearson r"),
            text=np.round(values, 2),
            texttemplate="%{text}" if annotate else None,
            textfont={"size": 11, "color": theme.INK_PRIMARY},
            hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    width, height = theme.square_matrix_size(len(labels))
    theme.style_figure(
        figure,
        title=spec.title or "Correlation between numeric features",
        x_title="Feature",
        y_title="Feature",
        subtitle=f"Pearson correlation, {len(labels)} columns, {len(frame):,} rows",
        height=height,
    )
    figure.update_layout(
        yaxis={"autorange": "reversed"},
        margin={"r": 150},
        width=width,
        xaxis={"tickangle": -30 if len(labels) > 8 else 0},
    )
    return _Built(figure=figure)


def _build_histogram(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    frame = _frame(state)
    params = params_to_dict(spec.parameters)
    bins = int(params.get("bins", params.get("nbins", 40)) or 40)
    chosen = _numeric_columns(frame, _known_columns(frame, spec.columns))
    if not chosen:
        chosen = _informative_numeric(state, frame, 1)
    if not chosen:
        raise ChartUnavailable("no numeric column to bin")
    chosen = chosen[:3]
    figure = go.Figure()
    for index, column in enumerate(chosen):
        figure.add_trace(
            go.Histogram(
                x=frame[column].to_numpy(),
                nbinsx=bins,
                name=column,
                marker=_bar_marker(theme.series_color(index)),
                opacity=0.78 if len(chosen) > 1 else 1.0,
                hovertemplate=f"{column}: %{{x}}<br>count %{{y}}<extra></extra>",
            )
        )
    figure.update_layout(barmode="overlay", showlegend=len(chosen) > 1)
    theme.style_figure(
        figure,
        title=spec.title or f"Distribution of {chosen[0]}",
        x_title=chosen[0] if len(chosen) == 1 else "Value",
        y_title="Row count",
        subtitle=f"{len(frame):,} rows, {bins} bins",
    )
    return _Built(figure=figure)


def _build_box(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    frame = _frame(state)
    chosen = _numeric_columns(frame, _known_columns(frame, spec.columns))
    if not chosen:
        chosen = _informative_numeric(state, frame, 5)
    if not chosen:
        raise ChartUnavailable("no numeric column to summarise")
    chosen = chosen[:6]
    figure = go.Figure()
    for index, column in enumerate(chosen):
        figure.add_trace(
            go.Box(
                y=frame[column].to_numpy(),
                name=column,
                marker=_marker(theme.series_color(index), size=6, opacity=0.6),
                line=_line(theme.series_color(index)),
                fillcolor=theme.alpha(theme.series_color(index), 0.12),
                boxpoints="outliers",
                hovertemplate=f"{column}<br>%{{y}}<extra></extra>",
            )
        )
    theme.style_figure(
        figure,
        title=spec.title or "Spread and outliers by feature",
        x_title="Feature",
        y_title="Value",
        subtitle="Box shows the interquartile range; whiskers 1.5x IQR, points beyond are outliers",
        show_legend=False,
    )
    figure.update_layout(margin={"r": 60})
    return _Built(figure=figure)


def _build_scatter(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    frame = _frame(state)
    chosen = _numeric_columns(frame, _known_columns(frame, spec.columns))
    if len(chosen) < 2:
        fallback = _informative_numeric(state, frame, 2)
        chosen = (chosen + [c for c in fallback if c not in chosen])[:2]
    if len(chosen) < 2:
        raise ChartUnavailable("a scatter plot needs two numeric columns")
    x_col, y_col = chosen[0], chosen[1]
    target = _target_name(state)
    sampled = _sample(frame, MAX_SCATTER_POINTS, state.config.random_state)
    note = (
        f"sampled {len(sampled):,} of {len(frame):,} rows"
        if len(sampled) < len(frame)
        else ""
    )
    figure = go.Figure()
    groups: list[Any] = []
    if target and _is_classification(state):
        values = sampled[target].dropna().unique().tolist()
        # Scatter is an all-pairs form: past three series the palette can no
        # longer keep every pair separable under simulated CVD, so colour is
        # dropped rather than faked.
        if 2 <= len(values) <= theme.ALL_PAIRS_SERIES_CAP:
            groups = sorted(values, key=str)
    if groups:
        for index, value in enumerate(groups):
            subset = sampled[sampled[target] == value]
            figure.add_trace(
                go.Scatter(
                    x=subset[x_col].to_numpy(),
                    y=subset[y_col].to_numpy(),
                    mode="markers",
                    name=f"{target} = {_display_label(state, value)}",
                    marker=_marker(theme.series_color(index)),
                    hovertemplate=f"{x_col} %{{x}}<br>{y_col} %{{y}}<extra></extra>",
                )
            )
    else:
        figure.add_trace(
            go.Scatter(
                x=sampled[x_col].to_numpy(),
                y=sampled[y_col].to_numpy(),
                mode="markers",
                name=y_col,
                marker=_marker(theme.series_color(0)),
                hovertemplate=f"{x_col} %{{x}}<br>{y_col} %{{y}}<extra></extra>",
            )
        )
    figure.update_layout(showlegend=bool(groups))
    theme.style_figure(
        figure,
        title=spec.title or f"{y_col} against {x_col}",
        x_title=x_col,
        y_title=y_col,
        subtitle=note or f"{len(sampled):,} rows",
    )
    return _Built(figure=figure, note=note)


def _build_bar(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    frame = _frame(state)
    params = params_to_dict(spec.parameters)
    top_n = int(params.get("top_n", MAX_CATEGORY_BARS) or MAX_CATEGORY_BARS)
    named = _known_columns(frame, spec.columns)
    chosen = _categorical_columns(frame, named)
    if not chosen:
        chosen = _categorical_columns(frame, _feature_pool(state, frame))
        chosen = [c for c in chosen if frame[c].nunique(dropna=True) <= 60]
    if not chosen:
        raise ChartUnavailable("no low-cardinality categorical column to count")
    column = chosen[0]
    counts = frame[column].astype("string").fillna("(missing)").value_counts().head(top_n)
    labels = [str(i) for i in counts.index]
    values = counts.to_numpy()
    # One nominal series, one hue: bar length already encodes the value, so a
    # colour ramp here would spend the identity channel on nothing.
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=_bar_marker(theme.series_color(0)),
            text=_value_text(values, 0),
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate="%{x}<br>%{y:,} rows<extra></extra>",
        )
    )
    theme.style_figure(
        figure,
        title=spec.title or f"Most frequent values of {column}",
        x_title=column,
        y_title="Row count",
        subtitle=f"top {len(labels)} of {frame[column].nunique(dropna=True):,} distinct values",
        show_legend=False,
    )
    figure.update_layout(bargap=_bargap(len(labels)))
    return _Built(
        figure=figure,
        table=([column, "rows"], [[labels[i], f"{int(values[i]):,}"] for i in range(len(labels))]),
    )


def _build_line(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    frame = _frame(state)
    named = _known_columns(frame, spec.columns)
    profile = state.profile
    temporal_candidates = [
        c
        for c in (
            [state.problem.temporal_column] if state.problem and state.problem.temporal_column else []
        )
        + (list(profile.temporal_columns) if profile else [])
        + named
        if c
    ]
    x_col = next((c for c in _known_columns(frame, temporal_candidates) if c), None)
    y_cols = [c for c in _numeric_columns(frame, named) if c != x_col]
    if not y_cols:
        y_cols = _informative_numeric(state, frame, 3)
    if not y_cols:
        raise ChartUnavailable("no numeric series to plot over time")
    y_cols = y_cols[:4]
    if x_col:
        ordered = frame[[x_col, *y_cols]].sort_values(x_col)
        x_values = ordered[x_col].to_numpy()
        x_title = x_col
    else:
        ordered = frame[y_cols]
        x_values = np.arange(len(ordered))
        x_title = "Row order"
    figure = go.Figure()
    for index, column in enumerate(y_cols):
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=ordered[column].to_numpy(),
                mode="lines",
                name=column,
                line=_line(theme.series_color(index)),
                hovertemplate=f"{column}: %{{y}}<br>%{{x}}<extra></extra>",
            )
        )
    figure.update_layout(showlegend=len(y_cols) > 1, hovermode="x unified")
    theme.style_figure(
        figure,
        title=spec.title or f"{y_cols[0]} over {x_title.lower()}",
        x_title=x_title,
        y_title=y_cols[0] if len(y_cols) == 1 else "Value",
        subtitle=f"{len(ordered):,} points",
    )
    return _Built(figure=figure)


def _build_missingness(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    profile = state.profile
    if profile is not None and profile.columns:
        pairs = [(c.name, float(c.missing_fraction)) for c in profile.columns]
        total_rows = profile.n_rows
    else:
        frame = _frame(state)
        pairs = [(str(c), float(frame[c].isna().mean())) for c in frame.columns]
        total_rows = len(frame)
    pairs.sort(key=lambda item: item[1], reverse=True)
    shown = [p for p in pairs if p[1] > 0][:MAX_MISSINGNESS_BARS]
    note = ""
    if not shown:
        shown = pairs[:MAX_MISSINGNESS_BARS]
        note = "no column has missing values"
    shown = list(reversed(shown))  # largest at the top of a horizontal bar
    names = [p[0] for p in shown]
    values = [p[1] * 100 for p in shown]
    height = max(420, 26 * len(names) + 180)
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            width=_bar_thickness(len(names), height),
            marker=_bar_marker(theme.series_color(0)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate="%{y}<br>%{x:.2f}% missing<extra></extra>",
        )
    )
    theme.style_figure(
        figure,
        title=spec.title or "Missing values by column",
        x_title="Missing (% of rows)",
        y_title="Column",
        subtitle=note or f"{total_rows:,} rows; columns with any missing values, worst first",
        show_legend=False,
        height=height,
    )
    # Scale to the data but keep a 25% floor: on a full 0-100 axis a 4% bar is
    # invisible, and on a tight axis it looks like a crisis.
    ceiling = min(100.0, max(25.0, max(values or [0.0]) * 1.3))
    figure.update_layout(
        margin={"l": 200, "r": 90}, xaxis={"range": [0, ceiling]}
    )
    return _Built(
        figure=figure,
        note=note,
        table=(
            ["column", "missing %"],
            [[names[i], f"{values[i]:.2f}"] for i in reversed(range(len(names)))],
        ),
    )


def _build_class_balance(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    profile = state.profile
    target_summary = profile.target if profile else None
    if target_summary is not None and target_summary.class_counts:
        labels = [c.value for c in target_summary.class_counts]
        counts = [c.count for c in target_summary.class_counts]
        subtitle = f"target '{target_summary.name}'"
        if target_summary.imbalance_ratio:
            subtitle += (
                f", majority/minority = {fmt_number(target_summary.imbalance_ratio, 2)}"
                f"{' (imbalanced)' if target_summary.is_imbalanced else ''}"
            )
    else:
        frame = _frame(state)
        target = _target_name(state)
        if not target:
            raise ChartUnavailable("no target column is resolved for this run")
        if not _is_classification(state):
            values = frame[target].to_numpy()
            figure = go.Figure(
                go.Histogram(
                    x=values,
                    nbinsx=40,
                    marker=_bar_marker(theme.series_color(0)),
                    hovertemplate="%{x}<br>count %{y}<extra></extra>",
                )
            )
            theme.style_figure(
                figure,
                title=spec.title or f"Distribution of {target}",
                x_title=target,
                y_title="Row count",
                subtitle="continuous target: a class balance chart does not apply, "
                "so the target distribution is shown instead",
                show_legend=False,
            )
            return _Built(figure=figure, note="continuous target: showing its distribution")
        counted = frame[target].value_counts()
        labels = [_display_label(state, i) for i in counted.index]
        counts = counted.to_numpy().tolist()
        subtitle = f"target '{target}'"
    total = float(sum(counts)) or 1.0
    figure = go.Figure(
        go.Bar(
            x=[str(label) for label in labels],
            y=counts,
            marker=_bar_marker(theme.series_color(0)),
            text=[f"{c:,} ({c / total * 100:.1f}%)" for c in counts],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate="%{x}<br>%{y:,} rows<extra></extra>",
        )
    )
    theme.style_figure(
        figure,
        title=spec.title or "Target class balance",
        x_title="Class",
        y_title="Row count",
        subtitle=subtitle,
        show_legend=False,
    )
    figure.update_layout(bargap=_bargap(len(labels)))
    return _Built(
        figure=figure,
        table=(
            ["class", "rows", "share"],
            [
                [str(labels[i]), f"{counts[i]:,}", f"{counts[i] / total * 100:.1f}%"]
                for i in range(len(labels))
            ],
        ),
    )


# ---------------------------------------------------------------------------
# builders — model performance
# ---------------------------------------------------------------------------


def _build_leaderboard(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    results = successful_experiments(state)
    if not results:
        raise ChartUnavailable("no experiment produced a score")
    higher = metric_direction(state)
    metric = state.primary_metric
    best_id = state.experiments.best_experiment_id if state.experiments else None
    if best_id is None:
        best_id = results[0].experiment_id
    # Horizontal bars place the first point at the bottom, so order worst -> best.
    ordered = sorted(results, key=lambda r: float(r.primary_score or 0.0), reverse=not higher)
    labels = [experiment_label(r) for r in ordered]
    values = [float(r.primary_score or 0.0) for r in ordered]
    colors: list[str] = []
    patterns: list[str] = []
    for result in ordered:
        if result.is_baseline:
            colors.append(theme.REFERENCE)
            patterns.append("/")
        elif result.experiment_id == best_id:
            colors.append(theme.series_color(0))
            patterns.append("")
        else:
            colors.append(theme.mix(theme.series_color(0), theme.SURFACE, 0.55))
            patterns.append("")
    height = max(400, 40 * len(labels) + 190)
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            width=_bar_thickness(len(labels), height, target_px=52),
            marker=_bar_marker(colors, patterns),
            text=_value_text(values, 4),
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate=f"%{{y}}<br>{metric} = %{{x:.4f}}<extra></extra>",
        )
    )
    direction = "higher is better" if higher else "lower is better"
    theme.style_figure(
        figure,
        title=spec.title or f"Model leaderboard on {metric}",
        x_title=f"{metric} ({direction})",
        y_title="Model",
        subtitle="winning model in solid blue, baseline hatched in grey",
        show_legend=False,
        height=height,
    )
    # Bars encode magnitude from zero. A truncated axis would turn the 0.013 gap
    # between the top two models into a visual chasm.
    lower = min(0.0, min(values) * 1.15)
    figure.update_layout(
        margin={"l": 230, "r": 110},
        xaxis={"range": [lower, max(values) * 1.18 if max(values) > 0 else 1.0]},
    )
    header = ["model", metric, "baseline"]
    rows = [
        [experiment_label(r), fmt_number(r.primary_score), "yes" if r.is_baseline else "no"]
        for r in results
    ]
    return _Built(figure=figure, table=(header, rows))


def _build_roc_curve(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    from sklearn.metrics import auc, roc_curve

    if not _is_classification(state):
        raise ChartUnavailable("ROC curves apply to classification tasks only")
    scores = view.require_scores()
    figure = go.Figure()
    y_true = np.asarray(view.y_true)
    classes = view.classes or _stable_labels(y_true)
    if len(classes) <= 2:
        positive_index = view.positive_index(
            state.problem.positive_class if state.problem else None
        )
        positive = classes[positive_index] if classes else 1
        binary = (y_true == positive).astype(int)
        if binary.sum() in (0, len(binary)):
            raise ChartUnavailable("the evaluation split contains a single class")
        fpr, tpr, _ = roc_curve(binary, view.column_for(positive_index))
        figure.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{_display_label(state, positive)} (AUC {auc(fpr, tpr):.3f})",
                line=_line(theme.series_color(0)),
                hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
            )
        )
    else:
        for index, value in enumerate(classes[:MAX_CURVE_CLASSES]):
            binary = (y_true == value).astype(int)
            if binary.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary, view.column_for(index))
            figure.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{_display_label(state, value)} (AUC {auc(fpr, tpr):.3f})",
                    line=_line(theme.series_color(index)),
                    hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
                )
            )
    if not figure.data:
        raise ChartUnavailable("no class had positive examples in the evaluation split")
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="random model",
            line=_line(theme.REFERENCE, 1, dash="dash"),
            hoverinfo="skip",
        )
    )
    note = "" if view.score_kind == "proba" else "built from decision_function scores"
    theme.style_figure(
        figure,
        title=spec.title or "ROC curve",
        x_title="False positive rate",
        y_title="True positive rate",
        subtitle=f"{view.split_name} split, {len(y_true):,} rows"
        + (f" — {note}" if note else ""),
        show_legend=True,
    )
    figure.update_yaxes(range=[-0.02, 1.02])
    figure.update_xaxes(range=[-0.02, 1.02])
    return _Built(figure=figure, note=note)


def _build_pr_curve(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    if not _is_classification(state):
        raise ChartUnavailable("precision-recall curves apply to classification only")
    view.require_scores()
    y_true = np.asarray(view.y_true)
    classes = view.classes or _stable_labels(y_true)
    figure = go.Figure()
    targets = (
        [(view.positive_index(state.problem.positive_class if state.problem else None), None)]
        if len(classes) <= 2
        else [(i, None) for i in range(min(len(classes), MAX_CURVE_CLASSES))]
    )
    baseline_rate = None
    for index, _ in targets:
        value = classes[index] if classes else 1
        binary = (y_true == value).astype(int)
        if binary.sum() == 0:
            continue
        column = view.column_for(index)
        precision, recall, _ = precision_recall_curve(binary, column)
        score = average_precision_score(binary, column)
        if baseline_rate is None:
            baseline_rate = float(binary.mean())
        figure.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{_display_label(state, value)} (AP {score:.3f})",
                line=_line(theme.series_color(index)),
                hovertemplate="recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>",
            )
        )
    if not figure.data:
        raise ChartUnavailable("no class had positive examples in the evaluation split")
    if baseline_rate is not None:
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[baseline_rate, baseline_rate],
                mode="lines",
                name=f"base rate ({baseline_rate:.3f})",
                line=_line(theme.REFERENCE, 1, dash="dash"),
                hoverinfo="skip",
            )
        )
    note = "" if view.score_kind == "proba" else "built from decision_function scores"
    theme.style_figure(
        figure,
        title=spec.title or "Precision-recall curve",
        x_title="Recall",
        y_title="Precision",
        subtitle=f"{view.split_name} split, {len(y_true):,} rows"
        + (f" — {note}" if note else ""),
        show_legend=True,
    )
    figure.update_yaxes(range=[-0.02, 1.02])
    figure.update_xaxes(range=[-0.02, 1.02])
    return _Built(figure=figure, note=note)


def _build_confusion_matrix(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    from sklearn.metrics import confusion_matrix

    if not _is_classification(state):
        raise ChartUnavailable("confusion matrices apply to classification only")
    view.require()
    y_true = np.asarray(view.y_true)
    y_pred = np.asarray(view.y_pred)
    labels = view.classes if view.classes else _stable_labels(y_true, y_pred)
    labels = [label for label in labels]  # stable order, model classes first
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = [_display_label(state, label) for label in labels]
    row_totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(
        matrix, np.where(row_totals == 0, 1, row_totals), dtype=float
    )
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=display,
            y=display,
            colorscale=theme.sequential_scale(),
            colorbar=theme.colorbar("Rows"),
            text=matrix,
            texttemplate="%{text:,}",
            textfont={"size": 12},
            customdata=np.round(shares * 100, 1),
            hovertemplate=(
                "true %{y} → predicted %{x}"
                "<br>%{z:,} rows (%{customdata:.1f}% of true class)<extra></extra>"
            ),
        )
    )
    correct = int(np.trace(matrix))
    total = int(matrix.sum())
    width, height = theme.square_matrix_size(len(display))
    theme.style_figure(
        figure,
        title=spec.title or "Confusion matrix",
        x_title="Predicted label",
        y_title="True label",
        subtitle=f"{view.split_name} split — {correct:,}/{total:,} correct "
        f"({correct / max(total, 1) * 100:.1f}%)",
        show_legend=False,
        height=height,
    )
    figure.update_layout(
        yaxis={"autorange": "reversed"}, margin={"r": 150}, width=width
    )
    header = ["true \\ predicted", *display]
    rows = [
        [display[i], *[f"{int(matrix[i][j]):,}" for j in range(len(display))]]
        for i in range(len(display))
    ]
    return _Built(figure=figure, table=(header, rows))


def _build_calibration_curve(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    from sklearn.calibration import calibration_curve

    if not _is_classification(state):
        raise ChartUnavailable("calibration applies to classification only")
    params = params_to_dict(spec.parameters)
    bins = int(params.get("n_bins", 10) or 10)
    view.require_scores(need_proba=True)
    y_true = np.asarray(view.y_true)
    classes = view.classes or _stable_labels(y_true)
    if len(classes) > 2:
        raise ChartUnavailable(
            "calibration is drawn for binary targets; this run is multiclass"
        )
    positive_index = view.positive_index(
        state.problem.positive_class if state.problem else None
    )
    positive = classes[positive_index] if classes else 1
    binary = (y_true == positive).astype(int)
    if binary.sum() in (0, len(binary)):
        raise ChartUnavailable("the evaluation split contains a single class")
    probability = view.column_for(positive_index)
    observed, predicted = calibration_curve(binary, probability, n_bins=bins, strategy="uniform")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="perfectly calibrated",
            line=_line(theme.REFERENCE, 1, dash="dash"),
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=predicted,
            y=observed,
            mode="lines+markers",
            name="model",
            line=_line(theme.series_color(0)),
            marker=_marker(theme.series_color(0)),
            hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<extra></extra>",
        )
    )
    theme.style_figure(
        figure,
        title=spec.title or "Calibration curve",
        x_title="Mean predicted probability",
        y_title="Observed frequency",
        subtitle=f"{view.split_name} split, {bins} equal-width bins, "
        f"positive class '{_display_label(state, positive)}'",
        show_legend=True,
    )
    figure.update_yaxes(range=[-0.02, 1.02])
    figure.update_xaxes(range=[-0.02, 1.02])
    return _Built(figure=figure)


def _build_prediction_distribution(
    state: RunState, spec: ChartSpec, view: _ModelView
) -> _Built:
    view.require()
    y_true = np.asarray(view.y_true)
    figure = go.Figure()
    if _is_classification(state) and view.scores is not None:
        classes = view.classes or _stable_labels(y_true)
        positive_index = view.positive_index(
            state.problem.positive_class if state.problem else None
        )
        positive = classes[positive_index] if classes else 1
        probability = view.column_for(positive_index)
        for index, value in enumerate(classes[:MAX_CURVE_CLASSES]):
            mask = y_true == value
            if not mask.any():
                continue
            figure.add_trace(
                go.Histogram(
                    x=probability[mask],
                    nbinsx=30,
                    name=f"actual {_display_label(state, value)}",
                    marker=_bar_marker(theme.series_color(index)),
                    opacity=0.75,
                    hovertemplate="score %{x:.2f}<br>count %{y}<extra></extra>",
                )
            )
        figure.update_layout(barmode="overlay", showlegend=True)
        theme.style_figure(
            figure,
            title=spec.title or "Predicted score by actual class",
            x_title=f"Predicted score for '{_display_label(state, positive)}'",
            y_title="Row count",
            subtitle=f"{view.split_name} split — separation between the "
            "distributions is the model's discriminative power",
        )
        return _Built(figure=figure)
    if _is_regression(state):
        figure.add_trace(
            go.Histogram(
                x=np.asarray(y_true, dtype=float),
                nbinsx=40,
                name="actual",
                marker=_bar_marker(theme.series_color(0)),
                opacity=0.75,
            )
        )
        figure.add_trace(
            go.Histogram(
                x=np.asarray(view.y_pred, dtype=float),
                nbinsx=40,
                name="predicted",
                marker=_bar_marker(theme.series_color(1)),
                opacity=0.75,
            )
        )
        figure.update_layout(barmode="overlay", showlegend=True)
        theme.style_figure(
            figure,
            title=spec.title or "Predicted against actual distribution",
            x_title=state.target or "Target",
            y_title="Row count",
            subtitle=f"{view.split_name} split — a narrower predicted "
            "distribution means the model is regressing toward the mean",
        )
        return _Built(figure=figure)
    counted = pd.Series(view.y_pred).value_counts()
    labels = [_display_label(state, i) for i in counted.index]
    figure.add_trace(
        go.Bar(
            x=labels,
            y=counted.to_numpy(),
            marker=_bar_marker(theme.series_color(0)),
            text=_value_text(counted.to_numpy(), 0),
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
        )
    )
    theme.style_figure(
        figure,
        title=spec.title or "Predicted class counts",
        x_title="Predicted label",
        y_title="Row count",
        subtitle=f"{view.split_name} split; no probabilities available",
        show_legend=False,
    )
    return _Built(figure=figure, note="no probability scores available")


def _build_learning_curve(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    curve = _stored_learning_curve(state)
    note = ""
    y_title = state.primary_metric
    if curve is None:
        curve = _compute_learning_curve(state)
        note = "computed here; the diagnostics step recorded no learning curve"
        y_title = "Score (estimator default scorer)"
    sizes, train_scores, valid_scores = curve
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=sizes,
            y=train_scores,
            mode="lines+markers",
            name="training score",
            line=_line(theme.series_color(0)),
            marker=_marker(theme.series_color(0)),
            hovertemplate="%{x:,} rows<br>train %{y:.4f}<extra></extra>",
        )
    )
    if valid_scores:
        figure.add_trace(
            go.Scatter(
                x=sizes,
                y=valid_scores,
                mode="lines+markers",
                name="validation score",
                line=_line(theme.series_color(1)),
                marker=_marker(theme.series_color(1)),
                hovertemplate="%{x:,} rows<br>validation %{y:.4f}<extra></extra>",
            )
        )
    theme.style_figure(
        figure,
        title=spec.title or "Learning curve",
        x_title="Training rows",
        y_title=y_title,
        subtitle="a wide, persistent gap is overfitting; two low flat lines are underfitting"
        + (f" ({note})" if note else ""),
        show_legend=True,
    )
    return _Built(figure=figure, note=note)


def _stored_learning_curve(
    state: RunState,
) -> tuple[list[float], list[float], list[float]] | None:
    """Read a learning curve out of whatever the diagnostics step left behind.

    The diagnostics bundle is not a field on ``RunState``, so it arrives through
    ``extras``. Key names are matched loosely on purpose — this module must not
    dictate another module's dict keys.
    """
    candidates: list[Any] = []
    extras = getattr(state, "extras", {}) or {}
    for key in ("learning_curve", "diagnostics", "diagnostics_bundle"):
        value = extras.get(key)
        if value is None:
            continue
        candidates.append(value if isinstance(value, dict) else getattr(value, "learning_curve", None))
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate:
            continue
        sizes: list[float] = []
        train: list[float] = []
        valid: list[float] = []
        for key, values in candidate.items():
            if not isinstance(values, (list, tuple)) or not values:
                continue
            name = str(key).lower()
            numbers = [float(v) for v in values]
            if "size" in name or name in ("n", "x", "rows"):
                sizes = numbers
            elif "train" in name:
                train = numbers
            elif any(token in name for token in ("valid", "test", "cv", "score")):
                valid = valid or numbers
        if train and not sizes:
            sizes = list(range(1, len(train) + 1))
        if train:
            length = min(len(sizes), len(train)) or len(train)
            return sizes[:length], train[:length], valid[:length] if valid else []
    return None


def _compute_learning_curve(state: RunState) -> tuple[list[float], list[float], list[float]]:
    from sklearn.model_selection import learning_curve

    estimator = state.best_pipeline or state.best_model
    features, labels = state.splits.X_train, state.splits.y_train
    if estimator is None or features is None or labels is None:
        raise ChartUnavailable(
            "no learning curve was recorded and there is no fitted model to compute one"
        )
    if len(features) > LEARNING_CURVE_ROW_LIMIT:
        raise ChartUnavailable(
            f"no learning curve was recorded and the training split "
            f"({len(features):,} rows) is too large to compute one here"
        )
    from sklearn.base import clone

    sizes, train_scores, valid_scores = learning_curve(
        clone(estimator),
        features,
        labels,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=3,
        n_jobs=1,
        shuffle=True,
        random_state=state.config.random_state,
        error_score=np.nan,
    )
    return (
        [float(v) for v in sizes],
        [float(v) for v in np.nanmean(train_scores, axis=1)],
        [float(v) for v in np.nanmean(valid_scores, axis=1)],
    )


def _build_time_series_forecast(
    state: RunState, spec: ChartSpec, view: _ModelView
) -> _Built:
    view.require()
    features = view.X
    temporal = None
    candidates = [state.problem.temporal_column] if state.problem else []
    if state.profile:
        candidates += list(state.profile.temporal_columns)
    candidates += list(spec.columns)
    if hasattr(features, "columns"):
        available = {str(c) for c in features.columns}
        temporal = next((c for c in candidates if c and str(c) in available), None)
    actual = np.asarray(view.y_true, dtype=float)
    predicted = np.asarray(view.y_pred, dtype=float)
    if temporal is not None:
        x_values = pd.Series(features[temporal]).to_numpy()
        order = np.argsort(x_values)
        x_values, actual, predicted = x_values[order], actual[order], predicted[order]
        x_title = str(temporal)
    else:
        x_values = np.arange(len(actual))
        x_title = "Step in the evaluation window"
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=actual,
            mode="lines",
            name="actual",
            line=_line(theme.series_color(0)),
            hovertemplate="%{x}<br>actual %{y:.4g}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=predicted,
            mode="lines",
            name="predicted",
            line=_line(theme.series_color(1)),
            hovertemplate="%{x}<br>predicted %{y:.4g}<extra></extra>",
        )
    )
    figure.update_layout(hovermode="x unified")
    theme.style_figure(
        figure,
        title=spec.title or "Actual against predicted over time",
        x_title=x_title,
        y_title=state.target or "Target",
        subtitle=f"{view.split_name} split, {len(actual):,} points",
        show_legend=True,
    )
    return _Built(figure=figure)


# ---------------------------------------------------------------------------
# builders — drivers & diagnostics
# ---------------------------------------------------------------------------


def _importance_figure(
    state: RunState, spec: ChartSpec, pairs: list[tuple[str, float]], *, method: str
) -> _Built:
    params = params_to_dict(spec.parameters)
    top_n = int(params.get("top_n", MAX_IMPORTANCE_FEATURES) or MAX_IMPORTANCE_FEATURES)
    ranked = sorted(pairs, key=lambda item: abs(item[1]), reverse=True)[:top_n]
    ranked = list(reversed(ranked))  # largest at the top
    names = [item[0] for item in ranked]
    values = [float(item[1]) for item in ranked]
    height = max(420, 26 * len(names) + 190)
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            width=_bar_thickness(len(names), height),
            marker=_bar_marker(theme.series_color(0)),
            text=[fmt_number(v, 4) for v in values],
            textposition="outside",
            textfont={"color": theme.INK_SECONDARY, "size": 11},
            hovertemplate="%{y}<br>importance %{x:.4f}<extra></extra>",
        )
    )
    theme.style_figure(
        figure,
        title=spec.title or f"Top {len(names)} features by importance",
        x_title=f"Importance ({method})",
        y_title="Feature",
        subtitle=f"{len(pairs)} features scored; showing the {len(names)} largest",
        show_legend=False,
        height=height,
    )
    top = max(values) if values else 1.0
    figure.update_layout(margin={"l": 240, "r": 110}, xaxis={"range": [0, top * 1.2]})
    header = ["feature", f"importance ({method})"]
    rows = [[names[i], fmt_number(values[i], 5)] for i in reversed(range(len(names)))]
    return _Built(figure=figure, table=(header, rows))


def _build_feature_importance(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    pairs = _importance_pairs(state)
    if not pairs:
        raise ChartUnavailable(
            "no feature attributions were recorded and the model exposes neither "
            "feature_importances_ nor coef_"
        )
    report = state.explainability
    method = "shap"
    if report is not None and report.global_attributions:
        method = report.global_attributions[0].method or "shap"
    elif report is not None and report.permutation_importance:
        method = report.permutation_importance[0].method or "permutation"
    else:
        method = "model-native"
    return _importance_figure(state, spec, pairs, method=method)


def _build_shap_summary(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    report = state.explainability
    if report is None:
        raise ChartUnavailable("no explainability report is available")
    attributions = [
        a for a in report.global_attributions if "shap" in (a.method or "").lower()
    ]
    method_label = "mean |SHAP value|"

    if not attributions:
        # SHAP values were not available as numbers. Rather than drop the panel,
        # fall back to whatever attribution the explainer did produce — but label
        # it with its real method, because a permutation ranking presented as a
        # SHAP chart would misstate what was measured.
        fallback = list(report.global_attributions) or list(report.permutation_importance)
        if not fallback:
            raise ChartUnavailable(
                "SHAP attributions are unavailable (shap not installed or "
                "explanation skipped), and no other attribution was computed"
            )
        attributions = fallback
        methods = sorted({(a.method or "unknown") for a in fallback})
        method_label = " / ".join(methods) if methods else "unknown"

    built = _importance_figure(
        state,
        spec,
        [(a.feature, float(a.importance)) for a in attributions],
        method=method_label,
    )
    if method_label == "mean |SHAP value|":
        subtitle = (
            "mean absolute SHAP value per feature — the share of the prediction "
            "each feature moves"
        )
    else:
        subtitle = (
            f"SHAP values were not available numerically; showing {method_label} "
            "instead. Importance ranks influence, not causation."
        )
    built.figure.update_layout(
        title={
            "text": spec.title or "Feature attribution",
            "subtitle": {"text": subtitle},
        }
    )
    # Only substitute the SHAP image when the numbers on screen are SHAP numbers.
    if report.shap_summary_path and method_label == "mean |SHAP value|":
        built.png_override = report.shap_summary_path
    return built


def _build_residuals(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    if not _is_regression(state):
        raise ChartUnavailable("residual plots apply to regression tasks only")
    view.require()
    actual = np.asarray(view.y_true, dtype=float)
    predicted = np.asarray(view.y_pred, dtype=float)
    residuals = actual - predicted
    if len(predicted) > MAX_SCATTER_POINTS:
        rng = np.random.default_rng(state.config.random_state)
        index = rng.choice(len(predicted), MAX_SCATTER_POINTS, replace=False)
        predicted, residuals = predicted[index], residuals[index]
        note = f"sampled {MAX_SCATTER_POINTS:,} of {len(actual):,} rows"
    else:
        note = ""
    figure = go.Figure(
        go.Scatter(
            x=predicted,
            y=residuals,
            mode="markers",
            name="residual",
            marker=_marker(theme.series_color(0), opacity=0.7),
            hovertemplate="predicted %{x:.4g}<br>residual %{y:.4g}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_width=1, line_color=theme.BASELINE)
    theme.style_figure(
        figure,
        title=spec.title or "Residuals against predicted value",
        x_title="Predicted value",
        y_title="Residual (actual - predicted)",
        subtitle=f"{view.split_name} split — structure here means the model is "
        "missing a pattern" + (f"; {note}" if note else ""),
        show_legend=False,
    )
    figure.update_layout(margin={"r": 60})
    return _Built(figure=figure, note=note)


def _build_residual_histogram(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    if not _is_regression(state):
        raise ChartUnavailable("residual histograms apply to regression tasks only")
    view.require()
    residuals = np.asarray(view.y_true, dtype=float) - np.asarray(view.y_pred, dtype=float)
    mean = float(np.nanmean(residuals))
    figure = go.Figure(
        go.Histogram(
            x=residuals,
            nbinsx=40,
            marker=_bar_marker(theme.series_color(0)),
            hovertemplate="residual %{x:.4g}<br>count %{y}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=theme.BASELINE)
    theme.style_figure(
        figure,
        title=spec.title or "Residual distribution",
        x_title="Residual (actual - predicted)",
        y_title="Row count",
        subtitle=f"{view.split_name} split — mean residual {fmt_number(mean)}, "
        f"std {fmt_number(float(np.nanstd(residuals)))}; a centred, symmetric "
        "shape is what an unbiased model looks like",
        show_legend=False,
    )
    figure.update_layout(margin={"r": 60})
    return _Built(figure=figure)


def _build_partial_dependence(state: RunState, spec: ChartSpec, view: _ModelView) -> _Built:
    from plotly.subplots import make_subplots
    from sklearn.inspection import partial_dependence

    estimator = state.best_pipeline or state.best_model
    features = state.splits.X_train
    if estimator is None or features is None or not hasattr(features, "columns"):
        raise ChartUnavailable("partial dependence needs a fitted model and a training frame")
    named = _known_columns(features, list(spec.columns))
    if not named:
        importance = [name for name, _ in _importance_pairs(state)]
        available = {str(c) for c in features.columns}
        named = [name for name in importance if name in available][:3]
    if not named:
        named = _numeric_columns(features)[:2]
    if not named:
        raise ChartUnavailable("no usable feature for a partial dependence plot")
    named = named[:3]
    sampled = _sample(features, 2000, state.config.random_state)
    figure = make_subplots(
        rows=1, cols=len(named), shared_yaxes=True, horizontal_spacing=0.08
    )
    drawn = 0
    for position, column in enumerate(named, start=1):
        try:
            result = partial_dependence(
                estimator, sampled, features=[column], kind="average", grid_resolution=25
            )
        except Exception as exc:
            logger.debug("partial dependence failed for %s: %s", column, exc)
            continue
        grid = np.asarray(result["grid_values"][0], dtype=float)
        average = np.asarray(result["average"][0], dtype=float)
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=average,
                mode="lines",
                name=column,
                line=_line(theme.series_color(0)),
                hovertemplate=f"{column} %{{x:.4g}}<br>prediction %{{y:.4g}}<extra></extra>",
            ),
            row=1,
            col=position,
        )
        figure.update_xaxes(title_text=column, row=1, col=position)
        drawn += 1
    if not drawn:
        raise ChartUnavailable("scikit-learn could not compute partial dependence for these features")
    figure.update_yaxes(title_text="Average prediction", row=1, col=1)
    theme.style_figure(
        figure,
        title=spec.title or "Partial dependence",
        subtitle="each panel keeps its own x scale — model response as one feature "
        f"varies, averaged over {len(sampled):,} rows",
        show_legend=False,
    )
    return _Built(figure=figure)


BUILDERS: dict[ChartKind, Callable[..., _Built]] = {
    ChartKind.CORRELATION_HEATMAP: _build_correlation_heatmap,
    ChartKind.HISTOGRAM: _build_histogram,
    ChartKind.BOX: _build_box,
    ChartKind.SCATTER: _build_scatter,
    ChartKind.BAR: _build_bar,
    ChartKind.LINE: _build_line,
    ChartKind.ROC_CURVE: _build_roc_curve,
    ChartKind.PR_CURVE: _build_pr_curve,
    ChartKind.CONFUSION_MATRIX: _build_confusion_matrix,
    ChartKind.FEATURE_IMPORTANCE: _build_feature_importance,
    ChartKind.SHAP_SUMMARY: _build_shap_summary,
    ChartKind.RESIDUALS: _build_residuals,
    ChartKind.RESIDUAL_HISTOGRAM: _build_residual_histogram,
    ChartKind.LEARNING_CURVE: _build_learning_curve,
    ChartKind.PREDICTION_DISTRIBUTION: _build_prediction_distribution,
    ChartKind.CALIBRATION_CURVE: _build_calibration_curve,
    ChartKind.MISSINGNESS: _build_missingness,
    ChartKind.CLASS_BALANCE: _build_class_balance,
    ChartKind.LEADERBOARD: _build_leaderboard,
    ChartKind.TIME_SERIES_FORECAST: _build_time_series_forecast,
    ChartKind.PARTIAL_DEPENDENCE: _build_partial_dependence,
}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def _kaleido_available() -> bool:
    return importlib.util.find_spec("kaleido") is not None


def _shutdown_image_backend() -> None:
    """Stop kaleido's headless-browser server, if it was started.

    Kaleido keeps a Chrome subprocess alive between exports. Tearing it down
    here — inside a guard — means a wedged browser costs a logged debug line at a
    controlled point instead of hanging the interpreter at exit, where nothing
    in this package can catch it.
    """
    if not _kaleido_available():
        return
    try:
        import kaleido

        stop = getattr(kaleido, "stop_sync_server", None)
        if callable(stop):
            stop(silence_warnings=True)
    except Exception as exc:  # teardown must never affect the run's outcome
        logger.debug("kaleido shutdown raised: %s", exc)


def _write_figure(
    state: RunState,
    artifact: ChartArtifact,
    built: _Built,
    slug: str,
    *,
    png: dict[str, bool],
) -> None:
    """Write the HTML, JSON, and (best-effort) PNG for one figure.

    PNG failures are swallowed deliberately: an image backend that has stopped
    working must cost the run one warning and some pictures, not its charts.
    """
    figure = built.figure
    html_path = state.artifact_path("charts", f"{slug}.html")
    figure.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
        config=theme.PLOTLY_CONFIG,
        div_id=slug,
    )
    artifact.html_path = str(html_path)

    json_path = state.artifact_path("charts", f"{slug}.json")
    json_path.write_text(figure.to_json(), encoding="utf-8")
    artifact.json_path = str(json_path)

    if built.png_override:
        artifact.png_path = built.png_override
        return
    if not png["enabled"]:
        return
    png_path = state.artifact_path("charts", f"{slug}.png")
    height = int(getattr(figure.layout, "height", None) or theme.DEFAULT_HEIGHT)
    # A figure that pinned its own width did so for a reason (square matrices).
    width = int(getattr(figure.layout, "width", None) or theme.EXPORT_WIDTH)
    try:
        figure.write_image(
            str(png_path), width=width, height=height, scale=theme.EXPORT_SCALE
        )
    except Exception as exc:
        png["enabled"] = False
        state.add_warning(
            f"PNG export disabled after kaleido failed on '{slug}' "
            f"({type(exc).__name__}: {exc}); later charts are HTML only"
        )
        return
    artifact.png_path = str(png_path)


def render_charts(state: RunState, plan: VisualizationPlan) -> VisualizationBundle:
    """Render every chart in ``plan`` and compose them into a dashboard.

    Each chart is built, styled, and written independently: a chart that cannot
    be produced records its reason on its own artifact and the remaining charts
    are unaffected. PNG export is attempted only while kaleido keeps working —
    the first export failure disables it for the rest of the run so a broken
    image backend costs one warning, not one per chart.

    Args:
        state: The run blackboard. Charts read the profile, splits, experiment
            log, explainability report, and fitted pipeline from it.
        plan: The Visualization Agent's chart specifications and dashboard
            narrative.

    Returns:
        A :class:`VisualizationBundle` with one artifact per requested chart, the
        dashboard path when the dashboard could be composed, and the narrative.
    """
    theme.register_template()
    bundle = VisualizationBundle(narrative=plan.dashboard_narrative)
    view = _ModelView(state)
    png = {"enabled": _kaleido_available()}
    if not png["enabled"]:
        state.add_warning(
            "kaleido is not installed: charts render as interactive HTML only, and "
            "the PDF/PowerPoint reports will omit chart images"
        )
    figures: dict[str, Any] = {}
    tables: dict[str, tuple[list[str], list[list[str]]]] = {}

    for index, spec in enumerate(plan.charts, start=1):
        kind = spec.kind if isinstance(spec.kind, ChartKind) else ChartKind(spec.kind)
        slug = f"{index:02d}_{enum_value(kind)}"
        artifact = ChartArtifact(spec=spec, caption=spec.rationale)
        builder = BUILDERS.get(kind)
        try:
            if builder is None:
                raise ChartUnavailable(f"no renderer is registered for chart kind '{kind}'")
            built = builder(state, spec, view)
            _write_figure(state, artifact, built, slug, png=png)
            artifact.rendered = True
            if built.note:
                artifact.caption = (
                    f"{artifact.caption} (Note: {built.note}.)"
                    if artifact.caption
                    else f"Note: {built.note}."
                )
            figures[slug] = built.figure
            if built.table:
                tables[slug] = built.table
        except ChartUnavailable as exc:
            artifact.error = str(exc)
            state.bus.log(f"chart '{spec.title}' skipped: {exc}")
        except Exception as exc:  # never let one chart end the run
            artifact.error = f"{type(exc).__name__}: {exc}"
            logger.exception("chart %s failed", slug)
            state.add_warning(f"chart '{spec.title}' failed: {artifact.error}")
        else:
            state.bus.artifact(artifact.html_path or slug, kind="chart")
        bundle.artifacts.append(artifact)

    _shutdown_image_backend()

    try:
        from .dashboard import build_dashboard

        bundle.dashboard_path = build_dashboard(
            state,
            bundle.artifacts,
            narrative=plan.dashboard_narrative,
            figures=figures,
            tables=tables,
        )
    except Exception as exc:  # the charts themselves are still usable
        logger.exception("dashboard composition failed")
        state.add_warning(f"dashboard could not be composed: {type(exc).__name__}: {exc}")

    state.visualizations = bundle
    return bundle


__all__ = [
    "BUILDERS",
    "ChartUnavailable",
    "MAX_IMPORTANCE_FEATURES",
    "render_charts",
]
