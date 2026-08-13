"""PDF rendering with reportlab.

The authored Markdown is converted to platypus flowables rather than dumped as
raw markup: headings become heading paragraphs, bullets become bulleted
paragraphs, and pipe tables become real tables with repeating header rows. A PDF
full of ``**bold**`` asterisks is the failure mode this module exists to avoid.

Chart PNGs are embedded where the chart renderer managed to export them. When
kaleido was unavailable the PDF simply has no pictures — it never fails for want
of an image.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.schemas import ChartArtifact, FinalReport
from . import theme
from .common import (
    chart_artifacts,
    charts_for_refs,
    decision_log,
    enum_value,
    fmt_number,
    headline_metrics,
    inline_to_reportlab,
    leaderboard_table,
    parse_markdown,
    run_metadata,
    usage_rows,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import RunState

logger = logging.getLogger(__name__)

MAX_IMAGE_HEIGHT_RATIO = 0.38


def _styles() -> dict[str, Any]:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    sheet = getSampleStyleSheet()
    ink = HexColor(theme.INK_PRIMARY)
    secondary = HexColor(theme.INK_SECONDARY)
    muted = HexColor(theme.INK_MUTED)
    body = ParagraphStyle(
        "ReportBody",
        parent=sheet["BodyText"],
        fontName="Helvetica",
        fontSize=9.6,
        leading=14.4,
        textColor=secondary,
        spaceAfter=8,
        alignment=TA_LEFT,
    )
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sheet["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            textColor=ink,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=body,
            fontSize=13,
            leading=18,
            textColor=secondary,
            spaceAfter=18,
        ),
        "meta": ParagraphStyle(
            "ReportMeta", parent=body, fontSize=8.6, leading=12, textColor=muted
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=ink,
            spaceBefore=18,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=ink,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "ReportH3",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=secondary,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": body,
        "bullet": ParagraphStyle(
            "ReportBullet", parent=body, leftIndent=16, bulletIndent=4, spaceAfter=4
        ),
        "quote": ParagraphStyle(
            "ReportQuote",
            parent=body,
            leftIndent=14,
            textColor=muted,
            borderPadding=(0, 0, 0, 6),
        ),
        "code": ParagraphStyle(
            "ReportCode",
            parent=body,
            fontName="Courier",
            fontSize=8.2,
            leading=11,
            textColor=ink,
            leftIndent=8,
            spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "ReportCell", parent=body, fontSize=8.4, leading=11.4, spaceAfter=0
        ),
        "cell_head": ParagraphStyle(
            "ReportCellHead",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=8.4,
            leading=11.4,
            textColor=ink,
            spaceAfter=0,
        ),
        "caption": ParagraphStyle(
            "ReportCaption",
            parent=body,
            fontSize=8.4,
            leading=11.4,
            textColor=muted,
            spaceBefore=4,
            spaceAfter=14,
        ),
        "tile": ParagraphStyle(
            "ReportTile", parent=body, fontSize=9, leading=12, spaceAfter=0
        ),
    }


def _table(
    header: list[str],
    rows: list[list[str]],
    styles: dict[str, Any],
    available_width: float,
    ratios: list[float] | None = None,
) -> Any:
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Paragraph, Table, TableStyle

    columns = max(len(header), max((len(r) for r in rows), default=0)) or 1
    if not ratios or len(ratios) != columns:
        ratios = [1.0 / columns] * columns
    widths = [available_width * ratio for ratio in ratios]

    data = [[Paragraph(inline_to_reportlab(str(c)), styles["cell_head"]) for c in header]]
    for row in rows:
        padded = list(row) + [""] * (columns - len(row))
        data.append(
            [Paragraph(inline_to_reportlab(str(c)), styles["cell"]) for c in padded]
        )
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor(theme.NEUTRAL_MID)),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, HexColor(theme.GRID)),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, HexColor(theme.BASELINE)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _markdown_flowables(
    text: str, styles: dict[str, Any], available_width: float
) -> list[Any]:
    """Convert authored Markdown into platypus flowables."""
    from reportlab.platypus import HRFlowable, Paragraph, Spacer

    from reportlab.lib.colors import HexColor

    flowables: list[Any] = []
    for block in parse_markdown(text):
        if block.kind == "heading":
            key = {1: "h1", 2: "h2"}.get(block.level, "h3")
            flowables.append(Paragraph(inline_to_reportlab(block.text), styles[key]))
        elif block.kind == "paragraph":
            flowables.append(Paragraph(inline_to_reportlab(block.text), styles["body"]))
        elif block.kind == "quote":
            flowables.append(Paragraph(inline_to_reportlab(block.text), styles["quote"]))
        elif block.kind in ("bullets", "numbered"):
            counter = 0
            for indent, item in block.items:
                counter += 1
                bullet = "•" if block.kind == "bullets" else f"{counter}."
                style = styles["bullet"].clone(
                    f"bullet{indent}", leftIndent=16 + 14 * indent, bulletIndent=4 + 14 * indent
                )
                flowables.append(
                    Paragraph(inline_to_reportlab(item), style, bulletText=bullet)
                )
            flowables.append(Spacer(1, 5))
        elif block.kind == "table" and block.rows:
            head, *rest = block.rows
            flowables.append(_table(head, rest, styles, available_width))
            flowables.append(Spacer(1, 6))
        elif block.kind == "code":
            for line in block.text.split("\n"):
                flowables.append(
                    Paragraph(
                        inline_to_reportlab(line).replace(" ", "&nbsp;") or "&nbsp;",
                        styles["code"],
                    )
                )
            flowables.append(Spacer(1, 6))
        elif block.kind == "rule":
            flowables.append(
                HRFlowable(
                    width="100%", thickness=0.5, color=HexColor(theme.GRID), spaceAfter=10
                )
            )
    return flowables


def _image_flowables(
    artifacts: list[ChartArtifact],
    styles: dict[str, Any],
    available_width: float,
    page_height: float,
) -> list[Any]:
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image, KeepTogether, Paragraph

    flowables: list[Any] = []
    for artifact in artifacts:
        if not artifact.png_path:
            continue
        path = Path(artifact.png_path)
        if not path.is_file():
            continue
        try:
            native_width, native_height = ImageReader(str(path)).getSize()
        except Exception as exc:  # a bad PNG costs one figure, not the document
            logger.warning("skipping chart image %s: %s", path, exc)
            continue
        scale = available_width / float(native_width)
        height = float(native_height) * scale
        max_height = page_height * MAX_IMAGE_HEIGHT_RATIO
        width = available_width
        if height > max_height:
            width = width * (max_height / height)
            height = max_height
        caption = f"<b>{artifact.spec.title}</b>"
        if artifact.caption:
            caption += f" — {inline_to_reportlab(artifact.caption)}"
        flowables.append(
            KeepTogether(
                [
                    Image(str(path), width=width, height=height),
                    Paragraph(caption, styles["caption"]),
                ]
            )
        )
    return flowables


def _page_decoration(canvas: Any, doc: Any, *, run_id: str, project: str) -> None:
    from reportlab.lib.colors import HexColor

    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(HexColor(theme.INK_MUTED))
    canvas.drawString(
        doc.leftMargin, 0.55 * doc.bottomMargin, f"{project} · run {run_id}"
    )
    canvas.drawRightString(
        doc.pagesize[0] - doc.rightMargin,
        0.55 * doc.bottomMargin,
        f"Page {canvas.getPageNumber()}",
    )
    canvas.setStrokeColor(HexColor(theme.GRID))
    canvas.setLineWidth(0.4)
    canvas.line(
        doc.leftMargin,
        0.55 * doc.bottomMargin + 10,
        doc.pagesize[0] - doc.rightMargin,
        0.55 * doc.bottomMargin + 10,
    )
    canvas.restoreState()


def render_pdf(
    state: RunState, report: FinalReport, path: Path | str | None = None
) -> Path:
    """Render the report as a PDF.

    Args:
        state: The run blackboard.
        report: The authored report. Section Markdown is converted to flowables,
            never emitted as raw markup.
        path: Output path. Defaults to ``<run>/report/report.pdf``.

    Returns:
        The path written.

    Raises:
        MissingDependencyError: If reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError as exc:  # pragma: no cover - reportlab ships with the venv
        from ..core.errors import MissingDependencyError

        raise MissingDependencyError("reportlab", "PDF report export") from exc

    target = Path(path) if path else state.artifact_path("report", "report.pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    document = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=report.title,
        author="AutoML Architect",
        subject=f"{state.config.project} — run {state.run_id}",
    )
    width = document.width
    page_height = A4[1]
    artifacts = chart_artifacts(state)
    story: list[Any] = []

    # --- title page ------------------------------------------------------
    story.append(Spacer(1, 42 * mm))
    story.append(Paragraph(inline_to_reportlab(report.title), styles["title"]))
    if report.subtitle:
        story.append(Paragraph(inline_to_reportlab(report.subtitle), styles["subtitle"]))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(
        Paragraph(
            f"{state.config.project} &nbsp;·&nbsp; run {state.run_id}"
            f"<br/>Generated {generated}",
            styles["meta"],
        )
    )
    story.append(Spacer(1, 14 * mm))
    metrics = headline_metrics(state)
    if metrics:
        story.append(
            _table(
                ["Measure", "Value", "Note"],
                [[m.label, m.value, m.note] for m in metrics],
                styles,
                width,
                ratios=[0.28, 0.28, 0.44],
            )
        )
    story.append(PageBreak())

    # --- contents --------------------------------------------------------
    story.append(Paragraph("Contents", styles["h1"]))
    contents = ["Executive summary"]
    contents += [s.heading for s in report.ordered_sections()]
    contents += [
        "Deployment recommendation",
        "Appendix A — Decision log",
        "Appendix B — Model leaderboard",
        "Appendix C — Run metadata",
        "Appendix D — Usage and cost",
    ]
    for index, entry in enumerate(contents, start=1):
        story.append(
            Paragraph(f"{index}. {inline_to_reportlab(entry)}", styles["bullet"])
        )
    story.append(PageBreak())

    # --- body ------------------------------------------------------------
    story.append(Paragraph("Executive summary", styles["h1"]))
    story.extend(_markdown_flowables(report.executive_summary, styles, width))

    referenced: set[int] = set()
    for section in report.ordered_sections():
        story.append(Paragraph(inline_to_reportlab(section.heading), styles["h1"]))
        story.extend(_markdown_flowables(section.body_markdown, styles, width))
        matched = charts_for_refs(section.chart_refs, artifacts)
        referenced.update(id(a) for a in matched)
        story.extend(_image_flowables(matched, styles, width, page_height))

    deployment = report.deployment
    story.append(Paragraph("Deployment recommendation", styles["h1"]))
    story.append(
        _table(
            ["Field", "Value"],
            [
                ["Pattern", enum_value(deployment.pattern)],
                ["Rationale", deployment.rationale],
                ["Latency (ms)", fmt_number(deployment.estimated_latency_ms)],
                ["Throughput (rps)", fmt_number(deployment.estimated_throughput_rps)],
                ["Model size (MB)", fmt_number(deployment.model_size_mb)],
                ["Retraining", deployment.retraining_cadence or "not specified"],
                ["Rollout", deployment.rollout_strategy or "not specified"],
                ["Infrastructure", deployment.infrastructure_notes or "not specified"],
            ],
            styles,
            width,
            ratios=[0.24, 0.76],
        )
    )
    if deployment.monitoring_plan:
        story.append(Paragraph("Monitoring plan", styles["h3"]))
        for item in deployment.monitoring_plan:
            story.append(
                Paragraph(inline_to_reportlab(item), styles["bullet"], bulletText="•")
            )
    if deployment.risks:
        story.append(Paragraph("Risks", styles["h3"]))
        for item in deployment.risks:
            story.append(
                Paragraph(inline_to_reportlab(item), styles["bullet"], bulletText="•")
            )

    unreferenced = [a for a in artifacts if id(a) not in referenced]
    if unreferenced:
        story.append(PageBreak())
        story.append(Paragraph("Charts", styles["h1"]))
        story.extend(_image_flowables(unreferenced, styles, width, page_height))

    story.append(PageBreak())
    story.append(Paragraph("Appendix A — Decision log", styles["h1"]))
    story.append(
        Paragraph(
            "Every choice an agent made, with the reason it gave, reproduced verbatim.",
            styles["body"],
        )
    )
    decisions = decision_log(state)
    story.append(
        _table(
            ["Agent", "Decision", "Rationale"],
            [[d.agent, d.decision, d.rationale] for d in decisions]
            or [["—", "no decisions were recorded", "—"]],
            styles,
            width,
            ratios=[0.13, 0.29, 0.58],
        )
    )

    story.append(Paragraph("Appendix B — Model leaderboard", styles["h1"]))
    header, rows = leaderboard_table(state)
    story.append(
        _table(
            header,
            rows or [["—", "no experiment produced a score", "—", "—", "—", "—"]],
            styles,
            width,
            ratios=[0.06, 0.34, 0.16, 0.16, 0.14, 0.14],
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("Appendix C — Run metadata", styles["h1"]),
                _table(
                    ["Field", "Value"],
                    [[k, v] for k, v in run_metadata(state)],
                    styles,
                    width,
                    ratios=[0.28, 0.72],
                ),
            ]
        )
    )
    story.append(
        KeepTogether(
            [
                Paragraph("Appendix D — Usage and cost", styles["h1"]),
                _table(
                    ["Measure", "Value"],
                    [[k, v] for k, v in usage_rows(state)],
                    styles,
                    width,
                    ratios=[0.4, 0.6],
                ),
            ]
        )
    )

    notes = list(report.appendix_notes) + list(state.warnings)
    if notes:
        story.append(Paragraph("Notes and warnings", styles["h1"]))
        for note in notes:
            story.append(
                Paragraph(inline_to_reportlab(note), styles["bullet"], bulletText="•")
            )

    def decorate(canvas: Any, doc: Any) -> None:
        _page_decoration(
            canvas, doc, run_id=state.run_id, project=state.config.project
        )

    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return target


__all__ = ["render_pdf"]
