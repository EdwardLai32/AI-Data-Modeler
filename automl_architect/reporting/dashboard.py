"""One self-contained interactive dashboard.

The constraint that shapes this module: the file has to open by double-clicking
it, on a laptop with no network, with no build step and no server. That rules out
a CDN script tag and any relative asset, so plotly.js is inlined once and every
figure is embedded as JSON next to it. The result is a single large HTML file that
keeps full hover, zoom, and legend interaction offline.

Layout is deliberately one column. A two-column grid halves the width available
to axis labels, and the first thing that breaks in a narrow chart is the tick
labels — which is the one thing a reader needs.
"""

from __future__ import annotations

import html as html_module
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.schemas import ChartArtifact
from . import theme
from .common import (
    Metric,
    enum_value,
    group_artifacts,
    headline_metrics,
    markdown_to_html,
    run_metadata,
    slugify,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import RunState

logger = logging.getLogger(__name__)


def _esc(value: Any) -> str:
    return html_module.escape("" if value is None else str(value), quote=True)


def _slug_for(artifact: ChartArtifact) -> str:
    if artifact.html_path:
        return Path(artifact.html_path).stem
    return slugify(artifact.spec.title, fallback=enum_value(artifact.spec.kind))


def _load_figure(artifact: ChartArtifact, figures: dict[str, Any]) -> Any | None:
    """Return the figure for an artifact, reloading from JSON when needed."""
    slug = _slug_for(artifact)
    figure = figures.get(slug)
    if figure is not None:
        return figure
    if not artifact.json_path:
        return None
    try:
        import plotly.io as pio

        return pio.from_json(Path(artifact.json_path).read_text(encoding="utf-8"))
    except Exception as exc:  # a missing figure costs one card, not the page
        logger.warning("could not reload figure %s: %s", artifact.json_path, exc)
        return None


def _metric_tiles(metrics: list[Metric]) -> str:
    if not metrics:
        return ""
    tiles = []
    for metric in metrics:
        note = f'<div class="tile-note">{_esc(metric.note)}</div>' if metric.note else ""
        tiles.append(
            '<div class="tile">'
            f'<div class="tile-label">{_esc(metric.label)}</div>'
            f'<div class="tile-value">{_esc(metric.value)}</div>'
            f"{note}</div>"
        )
    return f'<div class="tiles">{"".join(tiles)}</div>'


def _table_html(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{_esc(cell)}</th>" for cell in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        '<table class="data"><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _embedded_figure(figure: Any) -> tuple[Any, str]:
    """Strip a figure's own title for embedding, returning its subtitle text.

    The card already carries the chart title as its heading; printing it again
    inside the plot is the kind of duplication that makes a dashboard look
    assembled rather than designed. The standalone chart file and the PNG keep
    their titles, so this only affects the embedded copy.
    """
    import copy

    clone = copy.deepcopy(figure)
    subtitle = ""
    try:
        subtitle = clone.layout.title.subtitle.text or ""
    except AttributeError:
        subtitle = ""
    height = int(getattr(clone.layout, "height", None) or theme.DEFAULT_HEIGHT)
    clone.update_layout(
        title=None, margin={"t": 30}, height=max(300, height - 52)
    )
    return clone, subtitle


def _chart_card(
    artifact: ChartArtifact,
    figure: Any,
    table: tuple[list[str], list[list[str]]] | None,
) -> str:
    import plotly.io as pio

    slug = _slug_for(artifact)
    embedded, subtitle = _embedded_figure(figure)
    caption_parts = []
    if artifact.caption:
        caption_parts.append(_esc(artifact.caption))
    if subtitle:
        caption_parts.append(f'<span class="note">{_esc(subtitle)}</span>')
    caption = (
        f'<p class="caption">{" ".join(caption_parts)}</p>' if caption_parts else ""
    )
    plot = pio.to_html(
        embedded,
        full_html=False,
        include_plotlyjs=False,
        div_id=f"plot-{slug}",
        config=theme.PLOTLY_CONFIG,
        default_width="100%",
    )
    table_html = ""
    if table:
        table_html = (
            "<details class='twin'><summary>Data table</summary>"
            f"{_table_html(table[0], table[1])}</details>"
        )
    return (
        f'<section class="card" id="{_esc(slug)}">'
        f'<h3>{_esc(artifact.spec.title)}</h3>'
        f"{caption}{plot}{table_html}</section>"
    )


def _skipped_block(artifacts: list[ChartArtifact]) -> str:
    skipped = [a for a in artifacts if not a.rendered]
    if not skipped:
        return ""
    items = "".join(
        f"<li><strong>{_esc(a.spec.title)}</strong> "
        f'<span class="kind">({_esc(enum_value(a.spec.kind))})</span> — '
        f"{_esc(a.error or 'not rendered')}</li>"
        for a in skipped
    )
    return (
        '<section class="card" id="not-rendered">'
        "<h3>Charts that could not be rendered</h3>"
        '<p class="caption">Each entry records why the chart was skipped, so a '
        "missing plot is never silent.</p>"
        f'<ul class="skipped">{items}</ul></section>'
    )


_CSS = """
*, *::before, *::after { box-sizing: border-box; }
:root {
  color-scheme: light;
  %(vars)s
}
html { -webkit-text-size-adjust: 100%%; }
body {
  margin: 0;
  background: var(--surface-page);
  color: var(--text-primary);
  font-family: var(--font-stack);
  font-size: 15px;
  line-height: 1.55;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 40px 24px 72px; }
header.hero { border-bottom: 1px solid var(--grid); padding-bottom: 28px; margin-bottom: 8px; }
.eyebrow {
  text-transform: uppercase; letter-spacing: 0.08em; font-size: 11.5px;
  color: var(--text-muted); margin: 0 0 6px;
}
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; font-weight: 650; }
h2 { font-size: 20px; margin: 44px 0 4px; font-weight: 620; }
h3 { font-size: 16px; margin: 0 0 4px; font-weight: 620; }
.sub { color: var(--text-secondary); margin: 0 0 20px; max-width: 78ch; }
.tiles {
  display: grid; gap: 12px; margin: 22px 0 4px;
  grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
}
.tile {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px;
}
.tile-label {
  font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--text-muted);
}
.tile-value { font-size: 25px; font-weight: 640; color: var(--text-primary); margin-top: 2px; }
.tile-note { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
nav.jump { margin: 26px 0 0; font-size: 13.5px; }
nav.jump a { color: var(--series-1); text-decoration: none; margin-right: 18px; }
nav.jump a:hover { text-decoration: underline; }
.narrative { background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px 22px; margin: 26px 0 0; }
.narrative p { margin: 0 0 10px; color: var(--text-secondary); max-width: 82ch; }
.narrative p:last-child { margin-bottom: 0; }
.card {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px 22px 14px; margin: 18px 0 0;
}
.caption { color: var(--text-secondary); font-size: 13.5px; margin: 0 0 12px; max-width: 88ch; }
.caption .note { color: var(--text-muted); }
.group-note { color: var(--text-muted); font-size: 13px; margin: 0 0 6px; }
details.twin { margin-top: 6px; border-top: 1px solid var(--grid); padding-top: 8px; }
details.twin summary {
  cursor: pointer; font-size: 13px; color: var(--text-secondary); outline-offset: 3px;
}
table.data { border-collapse: collapse; margin-top: 12px; font-size: 13px; width: 100%%; }
table.data th, table.data td {
  text-align: left; padding: 6px 12px 6px 0; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums;
}
table.data th { color: var(--text-muted); font-weight: 600; white-space: nowrap; }
ul.skipped { margin: 0; padding-left: 20px; color: var(--text-secondary); font-size: 13.5px; }
ul.skipped .kind { color: var(--text-muted); }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 4px 24px; }
.meta div { font-size: 13px; color: var(--text-secondary); padding: 4px 0;
  border-bottom: 1px solid var(--grid); }
.meta span { color: var(--text-muted); display: inline-block; min-width: 132px; }
footer { margin-top: 40px; color: var(--text-muted); font-size: 12.5px;
  border-top: 1px solid var(--grid); padding-top: 16px; }
@media print {
  body { background: #fff; }
  .card { break-inside: avoid; border-color: #ccc; }
}
"""


def build_dashboard(
    state: RunState,
    artifacts: list[ChartArtifact],
    *,
    narrative: str = "",
    figures: dict[str, Any] | None = None,
    tables: dict[str, tuple[list[str], list[list[str]]]] | None = None,
    path: Path | str | None = None,
) -> str | None:
    """Compose rendered charts into one offline-capable HTML dashboard.

    Args:
        state: The run blackboard, read for headline metrics and metadata.
        artifacts: Chart artifacts from :func:`~automl_architect.reporting.charts.render_charts`.
            Unrendered ones are listed with their reason instead of being hidden.
        narrative: The Visualization Agent's dashboard narrative (Markdown).
        figures: Optional in-memory figures keyed by chart slug (the stem of the
            chart's HTML file). Missing entries are reloaded from ``json_path``.
        tables: Optional per-chart table twins keyed the same way.
        path: Output path. Defaults to ``<run>/report/dashboard.html``.

    Returns:
        The dashboard path as a string, or ``None`` when no chart rendered.
    """
    figures = figures or {}
    tables = tables or {}
    rendered = [a for a in artifacts if a.rendered]
    if not rendered:
        state.add_warning("no chart rendered, so no dashboard was written")
        return None

    import plotly.offline as offline

    grouped = group_artifacts(rendered)
    body: list[str] = []
    for group_title, group in grouped:
        anchor = slugify(group_title)
        body.append(f'<h2 id="{anchor}">{_esc(group_title)}</h2>')
        cards = []
        for artifact in group:
            figure = _load_figure(artifact, figures)
            if figure is None:
                continue
            cards.append(_chart_card(artifact, figure, tables.get(_slug_for(artifact))))
        if not cards:
            body.append(
                '<p class="group-note">Figures for this group could not be reloaded.</p>'
            )
        body.extend(cards)
    body.append(_skipped_block(artifacts))

    nav_items = "".join(
        f'<a href="#{slugify(title)}">{_esc(title)}</a>' for title, _ in grouped
    )
    if any(not a.rendered for a in artifacts):
        nav_items += '<a href="#not-rendered">Skipped</a>'

    narrative_html = (
        f'<div class="narrative">{markdown_to_html(narrative)}</div>' if narrative else ""
    )
    meta = "".join(
        f"<div><span>{_esc(key)}</span>{_esc(value)}</div>" for key, value in run_metadata(state)
    )
    title = f"{state.config.project} — run {state.run_id}"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    document = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} — AI Data Modeler dashboard</title>
<style>{_CSS % {"vars": theme.css_variables()}}</style>
<script type="text/javascript">{offline.get_plotlyjs()}</script>
</head>
<body>
<div class="wrap">
<header class="hero">
  <p class="eyebrow">AI Data Modeler · run dashboard</p>
  <h1>{_esc(state.config.project)}</h1>
  <p class="sub">Run <code>{_esc(state.run_id)}</code> · {len(rendered)} of
  {len(artifacts)} charts rendered · generated {_esc(generated)}</p>
  {_metric_tiles(headline_metrics(state))}
  <nav class="jump">{nav_items}<a href="#run-metadata">Run metadata</a></nav>
  {narrative_html}
</header>
{"".join(body)}
<h2 id="run-metadata">Run metadata</h2>
<section class="card"><div class="meta">{meta}</div></section>
<footer>Charts are computed from the run's own measurements. This file embeds
plotly.js, so it works offline with no server and no build step.</footer>
</div>
</body>
</html>
"""
    target = Path(path) if path else state.artifact_path("report", "dashboard.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    state.bus.artifact(str(target), kind="dashboard")
    logger.info("dashboard written to %s (%d charts)", target, len(rendered))
    return str(target)


__all__ = ["build_dashboard"]
