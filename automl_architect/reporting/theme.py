"""The visual language shared by every rendered artefact.

One palette, one plotly template, one set of ink tokens: charts, the dashboard,
and the HTML report all read from here so that a report looks like a single
system rather than a pile of independently styled plots.

The eight categorical hues are a *fixed order*. Slot assignment never cycles and
never re-colours when a series disappears, because a reader who learns "the
baseline is grey" must not be re-taught it on the next chart. That order is also
the colourblind-safety mechanism: it clears the adjacent-pair gates under
protanopia and deuteranopia (Machado-Oliveira-Fernandes 2009, severity 1.0) at
OKLab dE 9.1 with a normal-vision worst pair of 19.6 on this light surface.

Two consequences are load-bearing for the chart code in this package:

* Three light-mode slots (aqua, yellow, magenta) sit below 3:1 contrast against
  the surface, so any chart using them must also carry a readable label, a
  legend entry, or a table twin — never colour alone.
* Charts where *any* two marks can end up adjacent (scatter, bubble) cap out at
  three categorical series; past that they fold to a single hue. Lines, bars,
  and stacks only ever place neighbours side by side and may use all eight.
"""

from __future__ import annotations

from typing import Any

# --- categorical identity -------------------------------------------------

PALETTE: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

#: Series count beyond which an all-pairs chart form (scatter, bubble) collapses
#: to a single hue rather than inventing separations the palette cannot hold.
ALL_PAIRS_SERIES_CAP = 3

# --- surfaces and ink -----------------------------------------------------

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"
NEUTRAL_MID = "#f0efec"

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

#: A recessive fill for reference series (baselines, "perfect calibration").
#: Grey means "not a competitor", which is exactly what a baseline is.
REFERENCE = "#898781"

# --- magnitude and polarity ----------------------------------------------

#: Single-hue blue ramp, light -> dark, for continuous magnitude.
SEQUENTIAL_HEXES: tuple[str, ...] = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

TEMPLATE_NAME = "automl_light"

DEFAULT_HEIGHT = 480
EXPORT_WIDTH = 1180
EXPORT_SCALE = 2


def series_color(index: int) -> str:
    """Return the categorical hue for slot ``index`` (0-based).

    Args:
        index: Slot number. Indices at or beyond the palette length return the
            recessive reference grey rather than a generated hue — a ninth
            colour would be indistinguishable from an existing slot under
            simulated colour-vision deficiency.

    Returns:
        A hex colour string.
    """
    if index < 0:
        raise ValueError("series index must be non-negative")
    if index >= len(PALETTE):
        return REFERENCE
    return PALETTE[index]


def mix(first: str, second: str, weight: float) -> str:
    """Blend two hex colours in sRGB.

    Used only to derive ramp steps from documented anchors, never to invent a
    categorical hue.

    Args:
        first: Hex colour used at ``weight`` 0.
        second: Hex colour used at ``weight`` 1.
        weight: Blend position in ``[0, 1]``.

    Returns:
        The blended colour as a hex string.
    """
    ratio = min(1.0, max(0.0, float(weight)))
    left = _to_rgb(first)
    right = _to_rgb(second)
    blended = tuple(
        round(left[i] + (right[i] - left[i]) * ratio) for i in range(3)
    )
    return "#{:02x}{:02x}{:02x}".format(*blended)


def _to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a 6-digit hex colour, got {value!r}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def sequential_scale() -> list[list[Any]]:
    """Plotly colorscale for magnitude: one hue, light to dark."""
    last = len(SEQUENTIAL_HEXES) - 1
    return [[i / last, hexcode] for i, hexcode in enumerate(SEQUENTIAL_HEXES)]


def diverging_scale() -> list[list[Any]]:
    """Plotly colorscale for polarity: blue to a neutral grey to red.

    The warm arm is derived from the documented critical-status red by blending
    toward the neutral midpoint and the primary ink, so the arms have matching
    step counts and monotone lightness without eyeballed hex values.
    """
    warm = STATUS["critical"]
    return [
        [0.00, SEQUENTIAL_HEXES[6]],
        [0.16, SEQUENTIAL_HEXES[4]],
        [0.33, SEQUENTIAL_HEXES[1]],
        [0.50, NEUTRAL_MID],
        [0.67, mix(NEUTRAL_MID, warm, 0.55)],
        [0.84, warm],
        [1.00, mix(warm, INK_PRIMARY, 0.35)],
    ]


def colorbar(title: str) -> dict[str, Any]:
    """Styling for a trace colour bar.

    Colour bars are a per-trace property in plotly, so they cannot live in the
    layout template — this keeps them consistent instead.

    Args:
        title: Label for the scale, including units.

    Returns:
        A ``colorbar`` property dict.
    """
    return {
        "title": {
            "text": title,
            "side": "right",
            "font": {"color": INK_SECONDARY, "size": 12},
        },
        "outlinewidth": 0,
        "thickness": 12,
        "tickfont": {"color": INK_MUTED, "size": 11},
    }


def square_matrix_size(n_cells: int) -> tuple[int, int]:
    """Figure width and height that render an ``n_cells`` matrix as squares.

    Confusion matrices and correlation matrices are read as grids; stretching the
    cells to the width of a wide figure makes the diagonal impossible to scan.

    Args:
        n_cells: Number of rows (and columns) in the matrix.

    Returns:
        A ``(width, height)`` pixel pair including the standard margins.
    """
    cells = max(1, n_cells)
    cell = max(74, min(150, int(760 / cells)))
    plot = cell * cells
    return plot + 78 + 150, plot + 96 + 78


def alpha(hexcode: str, opacity: float) -> str:
    """Return ``hexcode`` as an ``rgba()`` string at ``opacity``."""
    red, green, blue = _to_rgb(hexcode)
    return f"rgba({red},{green},{blue},{max(0.0, min(1.0, opacity)):.3f})"


# --- plotly template ------------------------------------------------------


def _axis() -> dict[str, Any]:
    return {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "griddash": "solid",
        "zeroline": True,
        "zerolinecolor": BASELINE,
        "zerolinewidth": 1,
        "showline": True,
        "linecolor": BASELINE,
        "linewidth": 1,
        "ticks": "outside",
        "ticklen": 4,
        "tickcolor": BASELINE,
        "tickfont": {"color": INK_MUTED, "size": 12},
        "title": {"font": {"color": INK_SECONDARY, "size": 13}, "standoff": 10},
        "automargin": True,
    }


def build_template() -> Any:
    """Construct the light plotly template used by every chart."""
    import plotly.graph_objects as go

    return go.layout.Template(
        layout={
            "colorway": list(PALETTE),
            "font": {"family": FONT_STACK, "size": 13, "color": INK_SECONDARY},
            "title": {
                "font": {"family": FONT_STACK, "size": 18, "color": INK_PRIMARY},
                "x": 0.0,
                "xanchor": "left",
                "y": 0.97,
                "yanchor": "top",
                # A gutter measured from the figure edge, not the plot area: it
                # keeps titles aligned across charts whose left margins differ
                # wildly (a horizontal bar chart needs 240px for its labels).
                "pad": {"l": 26, "t": 6},
                "subtitle": {"font": {"size": 12.5, "color": INK_MUTED}},
            },
            "paper_bgcolor": PAGE,
            "plot_bgcolor": SURFACE,
            "colorscale": {
                "sequential": sequential_scale(),
                "sequentialminus": sequential_scale(),
                "diverging": diverging_scale(),
            },
            "xaxis": _axis(),
            "yaxis": _axis(),
            "legend": {
                "orientation": "v",
                "x": 1.02,
                "xanchor": "left",
                "y": 1.0,
                "yanchor": "top",
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "font": {"color": INK_SECONDARY, "size": 12},
                "itemsizing": "constant",
                "tracegroupgap": 8,
            },
            "margin": {"l": 78, "r": 176, "t": 96, "b": 68},
            "hoverlabel": {
                "bgcolor": SURFACE,
                "bordercolor": BASELINE,
                "font": {"family": FONT_STACK, "size": 12, "color": INK_PRIMARY},
                "align": "left",
            },
            "hovermode": "closest",
            "bargap": 0.28,
            "boxgap": 0.4,
            "showlegend": False,
        }
    )


def register_template() -> str:
    """Register the template with plotly.io (idempotent).

    Returns:
        The registered template name, suitable for ``fig.update_layout``.
    """
    import plotly.io as pio

    if TEMPLATE_NAME not in pio.templates:
        pio.templates[TEMPLATE_NAME] = build_template()
    return TEMPLATE_NAME


def style_figure(
    figure: Any,
    *,
    title: str,
    x_title: str = "",
    y_title: str = "",
    subtitle: str = "",
    height: int | None = None,
    show_legend: bool | None = None,
) -> Any:
    """Apply the shared template and the mandatory labelling to a figure.

    Every chart in this package goes through here, which is what guarantees the
    non-negotiables: a real title, named axes, a light surface, and the shared
    colourway.

    Args:
        figure: A plotly ``Figure``.
        title: Chart title. Required — an untitled chart is not shippable.
        x_title: X axis title. Omit only for axis-free forms (heatmaps of a
            square matrix still get one).
        y_title: Y axis title.
        subtitle: Optional second line, used for units, sample size, or a
            "computed on the test split" note.
        height: Pixel height; defaults to :data:`DEFAULT_HEIGHT`.
        show_legend: Force the legend on or off. ``None`` leaves plotly's
            per-trace decision alone.

    Returns:
        The same figure, mutated, for chaining.
    """
    layout: dict[str, Any] = {
        "template": register_template(),
        "title": {"text": title},
        "height": height or DEFAULT_HEIGHT,
    }
    if subtitle:
        layout["title"]["subtitle"] = {"text": subtitle}
    if x_title:
        layout["xaxis"] = {"title": {"text": x_title}}
    if y_title:
        layout["yaxis"] = {"title": {"text": y_title}}
    if show_legend is not None:
        layout["showlegend"] = show_legend
        if not show_legend:
            # Reclaim the space the template reserves for a right-hand legend.
            layout["margin"] = {"r": 64}
    figure.update_layout(**layout)
    return figure


PLOTLY_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def css_variables() -> str:
    """Return the palette as a CSS custom-property block.

    The dashboard and the HTML report both declare these once and then style
    against roles, so a palette change is a one-line edit here.
    """
    lines = [
        f"--surface-1: {SURFACE};",
        f"--surface-page: {PAGE};",
        f"--text-primary: {INK_PRIMARY};",
        f"--text-secondary: {INK_SECONDARY};",
        f"--text-muted: {INK_MUTED};",
        f"--grid: {GRID};",
        f"--baseline: {BASELINE};",
        f"--border: {BORDER};",
        f"--good: {STATUS['good']};",
        f"--warning: {STATUS['warning']};",
        f"--critical: {STATUS['critical']};",
        f"--font-stack: {FONT_STACK};",
    ]
    lines += [f"--series-{i + 1}: {hexcode};" for i, hexcode in enumerate(PALETTE)]
    return "\n  ".join(lines)


__all__ = [
    "ALL_PAIRS_SERIES_CAP",
    "BASELINE",
    "BORDER",
    "DEFAULT_HEIGHT",
    "EXPORT_SCALE",
    "EXPORT_WIDTH",
    "FONT_STACK",
    "GRID",
    "INK_MUTED",
    "INK_PRIMARY",
    "INK_SECONDARY",
    "NEUTRAL_MID",
    "PAGE",
    "PALETTE",
    "PLOTLY_CONFIG",
    "REFERENCE",
    "SEQUENTIAL_HEXES",
    "STATUS",
    "SURFACE",
    "TEMPLATE_NAME",
    "alpha",
    "build_template",
    "colorbar",
    "css_variables",
    "diverging_scale",
    "mix",
    "register_template",
    "sequential_scale",
    "series_color",
    "square_matrix_size",
    "style_figure",
]
