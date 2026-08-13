"""PowerPoint rendering with python-pptx.

A deck is not a report: slides get the *claims*, not the prose. Bullets are
therefore extracted from the authored Markdown — real list items where the author
wrote them, leading sentences otherwise — and never invented. Long sections spill
onto continuation slides rather than shrinking to unreadable type.

Charts become their own full-bleed slides, because a chart squeezed beside a
bullet list is legible in neither role.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.schemas import ChartArtifact, FinalReport
from . import theme
from .common import (
    Metric,
    bullets_from_markdown,
    chart_artifacts,
    decision_log,
    enum_value,
    headline_metrics,
    leaderboard_table,
    strip_inline,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import RunState

logger = logging.getLogger(__name__)

SLIDE_WIDTH_IN = 13.333
SLIDE_HEIGHT_IN = 7.5
BULLETS_PER_SLIDE = 6
MAX_CHART_SLIDES = 12
MAX_LEADERBOARD_ROWS = 6


def _rgb(hexcode: str) -> Any:
    from pptx.dml.color import RGBColor

    return RGBColor.from_string(hexcode.lstrip("#").upper())


def _style_runs(paragraph: Any, *, size: float, color: str, bold: bool = False) -> None:
    from pptx.util import Pt

    if not paragraph.runs:
        return
    for run in paragraph.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = _rgb(color)


def _set_title(slide: Any, text: str, *, size: float = 30) -> None:
    from pptx.util import Pt

    title = slide.shapes.title
    if title is None:
        return
    title.text_frame.text = strip_inline(text)
    paragraph = title.text_frame.paragraphs[0]
    _style_runs(paragraph, size=size, color=theme.INK_PRIMARY, bold=True)
    title.text_frame.word_wrap = True
    for run in paragraph.runs:
        run.font.size = Pt(size)


def _fill_bullets(frame: Any, bullets: list[str], *, size: float = 16) -> None:
    from pptx.util import Pt

    frame.word_wrap = True
    if not bullets:
        bullets = ["No content was authored for this section."]
    for index, raw in enumerate(bullets):
        level = (len(raw) - len(raw.lstrip(" "))) // 4
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = raw.strip()
        paragraph.level = min(level, 2)
        paragraph.space_after = Pt(8)
        _style_runs(
            paragraph,
            size=size if level == 0 else size - 2,
            color=theme.INK_SECONDARY if level else theme.INK_PRIMARY,
        )


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)] or [[]]


def _content_slide(presentation: Any, title: str, bullets: list[str]) -> None:
    for index, chunk in enumerate(_chunk(bullets, BULLETS_PER_SLIDE)):
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        _set_title(slide, title if index == 0 else f"{title} (cont.)", size=26)
        body = None
        for placeholder in slide.placeholders:
            if placeholder.placeholder_format.idx != 0:
                body = placeholder
                break
        if body is None:  # layout without a body placeholder
            continue
        _fill_bullets(body.text_frame, chunk)


def _metric_strip(slide: Any, metrics: list[Metric]) -> None:
    """Lay headline numbers across the slide as a row of stat tiles."""
    from pptx.util import Emu, Inches, Pt

    if not metrics:
        return
    shown = metrics[:4]
    gap = Inches(0.25)
    margin = Inches(0.6)
    total = Inches(SLIDE_WIDTH_IN) - margin * 2
    tile_width = Emu(int((total - gap * (len(shown) - 1)) / len(shown)))
    top = Inches(5.55)
    height = Inches(1.15)
    for index, metric in enumerate(shown):
        left = Emu(int(margin + (tile_width + gap) * index))
        box = slide.shapes.add_textbox(left, top, tile_width, height)
        frame = box.text_frame
        frame.word_wrap = True
        label = frame.paragraphs[0]
        label.text = metric.label.upper()
        _style_runs(label, size=10, color=theme.INK_MUTED)
        value = frame.add_paragraph()
        value.text = metric.value
        _style_runs(value, size=22, color=theme.INK_PRIMARY, bold=True)
        if metric.note:
            note = frame.add_paragraph()
            note.text = metric.note
            _style_runs(note, size=9.5, color=theme.INK_SECONDARY)
        for paragraph in frame.paragraphs:
            paragraph.space_after = Pt(0)


def _add_table(
    slide: Any,
    header: list[str],
    rows: list[list[str]],
    *,
    top_in: float = 1.7,
    height_in: float = 4.4,
) -> None:
    from pptx.util import Inches, Pt

    left = Inches(0.6)
    width = Inches(SLIDE_WIDTH_IN - 1.2)
    shape = slide.shapes.add_table(
        len(rows) + 1, len(header), left, Inches(top_in), width, Inches(height_in)
    )
    table = shape.table
    for column, name in enumerate(header):
        cell = table.cell(0, column)
        cell.text = str(name)
        _style_runs(cell.text_frame.paragraphs[0], size=12, color=theme.INK_PRIMARY, bold=True)
    for row_index, row in enumerate(rows, start=1):
        for column in range(len(header)):
            value = row[column] if column < len(row) else ""
            cell = table.cell(row_index, column)
            cell.text = str(value)
            paragraph = cell.text_frame.paragraphs[0]
            _style_runs(paragraph, size=11, color=theme.INK_SECONDARY)
            paragraph.space_after = Pt(0)


def _chart_slide(presentation: Any, artifact: ChartArtifact) -> bool:
    from pptx.util import Emu, Inches, Pt

    if not artifact.png_path or not Path(artifact.png_path).is_file():
        return False
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    _set_title(slide, artifact.spec.title, size=24)
    max_width = Inches(SLIDE_WIDTH_IN - 1.6)
    try:
        picture = slide.shapes.add_picture(
            artifact.png_path, Inches(0.8), Inches(1.45), width=max_width
        )
    except Exception as exc:  # a corrupt PNG costs one slide, not the deck
        logger.warning("could not place %s: %s", artifact.png_path, exc)
        return False
    max_height = Inches(4.55)
    if picture.height > max_height:
        ratio = max_height / picture.height
        picture.height = Emu(int(picture.height * ratio))
        picture.width = Emu(int(picture.width * ratio))
        picture.left = Emu(int((Inches(SLIDE_WIDTH_IN) - picture.width) / 2))
    if artifact.caption:
        box = slide.shapes.add_textbox(
            Inches(0.8), Inches(6.25), Inches(SLIDE_WIDTH_IN - 1.6), Inches(0.9)
        )
        box.text_frame.word_wrap = True
        paragraph = box.text_frame.paragraphs[0]
        paragraph.text = strip_inline(artifact.caption)[:280]
        paragraph.space_after = Pt(0)
        _style_runs(paragraph, size=12, color=theme.INK_SECONDARY)
    return True


def _recommendation_bullets(state: RunState, report: FinalReport) -> list[str]:
    bullets: list[str] = []
    evaluation = state.evaluation
    if evaluation is not None:
        bullets.append(
            f"Verdict: grade {enum_value(evaluation.overall_grade)} — "
            f"{'recommended for deployment' if evaluation.acceptable else 'not yet deployable'}; "
            f"action: {enum_value(evaluation.recommended_action)}"
        )
        bullets += [f"    {item}" for item in evaluation.specific_improvements[:4]]
    deployment = report.deployment
    bullets.append(
        f"Deploy as {enum_value(deployment.pattern)}"
        + (f", retraining {deployment.retraining_cadence}" if deployment.retraining_cadence else "")
    )
    bullets += [f"    {item}" for item in deployment.monitoring_plan[:3]]
    insights = state.insights
    if insights is not None:
        bullets += [
            f"Next: {item}" for item in insights.suggested_next_experiments[:3]
        ]
    if not bullets:
        bullets = [strip_inline(note) for note in report.appendix_notes[:5]]
    return bullets


def render_pptx(
    state: RunState, report: FinalReport, path: Path | str | None = None
) -> Path:
    """Render the report as a PowerPoint deck.

    Args:
        state: The run blackboard.
        report: The authored report. Bullets are extracted from its Markdown.
        path: Output path. Defaults to ``<run>/report/report.pptx``.

    Returns:
        The path written.

    Raises:
        MissingDependencyError: If python-pptx is not installed.
    """
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError as exc:  # pragma: no cover - python-pptx ships with the venv
        from ..core.errors import MissingDependencyError

        raise MissingDependencyError("python-pptx", "PowerPoint report export") from exc

    target = Path(path) if path else state.artifact_path("report", "report.pptx")
    target.parent.mkdir(parents=True, exist_ok=True)

    presentation = Presentation()
    presentation.slide_width = Inches(SLIDE_WIDTH_IN)
    presentation.slide_height = Inches(SLIDE_HEIGHT_IN)

    # --- title slide -----------------------------------------------------
    slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    _set_title(slide, report.title, size=36)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subtitle_placeholder = next(
        (p for p in slide.placeholders if p.placeholder_format.idx != 0), None
    )
    if subtitle_placeholder is not None:
        frame = subtitle_placeholder.text_frame
        frame.word_wrap = True
        first = frame.paragraphs[0]
        first.text = strip_inline(report.subtitle) or state.config.project
        _style_runs(first, size=17, color=theme.INK_SECONDARY)
        detail = frame.add_paragraph()
        detail.text = (
            f"{state.config.project} · run {state.run_id} · generated {generated}"
        )
        detail.space_after = Pt(0)
        _style_runs(detail, size=12, color=theme.INK_MUTED)

    # --- executive summary ----------------------------------------------
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    _set_title(slide, "Executive summary", size=28)
    box = slide.shapes.add_textbox(
        Inches(0.6), Inches(1.5), Inches(SLIDE_WIDTH_IN - 1.2), Inches(3.8)
    )
    _fill_bullets(
        box.text_frame, bullets_from_markdown(report.executive_summary, limit=5), size=17
    )
    _metric_strip(slide, headline_metrics(state))

    # --- leaderboard ------------------------------------------------------
    header, rows = leaderboard_table(state)
    if rows:
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        _set_title(slide, f"Model leaderboard — {state.primary_metric}", size=26)
        _add_table(
            slide,
            header,
            rows[:MAX_LEADERBOARD_ROWS],
            height_in=min(4.6, 0.5 * (min(len(rows), MAX_LEADERBOARD_ROWS) + 1)),
        )

    # --- one slide per authored section ----------------------------------
    for section in report.ordered_sections():
        _content_slide(
            presentation,
            section.heading,
            bullets_from_markdown(section.body_markdown, limit=BULLETS_PER_SLIDE * 2),
        )

    # --- charts -----------------------------------------------------------
    placed = 0
    for artifact in chart_artifacts(state):
        if placed >= MAX_CHART_SLIDES:
            break
        if _chart_slide(presentation, artifact):
            placed += 1

    # --- recommendations --------------------------------------------------
    _content_slide(presentation, "Recommendations", _recommendation_bullets(state, report))

    # --- decision trail ---------------------------------------------------
    decisions = decision_log(state)
    if decisions:
        _content_slide(
            presentation,
            "Why the pipeline did what it did",
            [f"{d.agent}: {d.decision}" for d in decisions[:BULLETS_PER_SLIDE * 2]],
        )

    presentation.save(str(target))
    return target


__all__ = ["render_pptx"]
