"""Styled, standalone HTML rendering of the final report.

The template lives next to this module (``templates/report.html.j2``) with its CSS
embedded, and chart images are inlined as base64 data URIs. That combination is
what makes the output survive being emailed as a single file: no stylesheet to
lose, no image directory to forget, no network fetch to fail. The interactive
chart files are still linked for readers who have the whole run directory.

The section bodies are authored Markdown and are converted, not rewritten.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markupsafe import Markup

from ..core.schemas import ChartArtifact, FinalReport
from . import theme
from .common import (
    chart_artifacts,
    charts_for_refs,
    decision_log,
    enum_value,
    fmt_number,
    headline_metrics,
    leaderboard_table,
    markdown_to_html,
    run_metadata,
    slugify,
    usage_rows,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import RunState

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.html.j2"

#: Above this size an image is linked rather than inlined — a 40 MB data URI
#: makes a browser unhappy and the interactive file is right next to it anyway.
MAX_INLINE_IMAGE_BYTES = 4_000_000


def _relative(target: str | None, base: Path) -> str | None:
    if not target:
        return None
    try:
        return Path(os.path.relpath(Path(target), base)).as_posix()
    except (OSError, ValueError):
        return Path(target).as_posix()


def _data_uri(path: str | None) -> str | None:
    """Base64 data URI for a chart image, or ``None`` if it cannot be inlined."""
    if not path:
        return None
    file_path = Path(path)
    try:
        if not file_path.is_file() or file_path.stat().st_size > MAX_INLINE_IMAGE_BYTES:
            return None
        payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    except OSError as exc:
        logger.warning("could not inline %s: %s", path, exc)
        return None
    mime = mimetypes.guess_type(file_path.name)[0] or "image/png"
    return f"data:{mime};base64,{payload}"


def _chart_context(artifact: ChartArtifact, base: Path) -> dict[str, Any]:
    href = _relative(artifact.html_path, base)
    src = _data_uri(artifact.png_path)
    if src is None and artifact.png_path:
        src = _relative(artifact.png_path, base)
    return {
        "title": artifact.spec.title,
        "caption": artifact.caption,
        "kind": enum_value(artifact.spec.kind),
        "src": src,
        "href": href,
    }


def build_context(
    state: RunState, report: FinalReport, base: Path | None = None
) -> dict[str, Any]:
    """Assemble the template context for the HTML report.

    Exposed separately so a caller (or a test) can inspect exactly what the
    template will see.

    Args:
        state: The run blackboard.
        report: The authored report.
        base: Directory the HTML will be written to, for relative links.

    Returns:
        The jinja2 context dictionary.
    """
    base_dir = base or state.artifact_path("report", "report.html").parent
    artifacts = chart_artifacts(state)
    used: set[int] = set()
    sections = []
    for section in report.ordered_sections():
        matched = charts_for_refs(section.chart_refs, artifacts)
        used.update(id(a) for a in matched)
        sections.append(
            {
                "heading": section.heading,
                "anchor": slugify(section.heading, f"section-{section.order}"),
                "body": Markup(markdown_to_html(section.body_markdown)),
                "charts": [_chart_context(a, base_dir) for a in matched],
            }
        )

    orphans = [a for a in artifacts if id(a) not in used]
    bundle = state.visualizations
    skipped = [
        {
            "title": a.spec.title,
            "kind": enum_value(a.spec.kind),
            "error": a.error or "not rendered",
        }
        for a in (bundle.artifacts if bundle else [])
        if not a.rendered
    ]
    dashboard_href = _relative(bundle.dashboard_path, base_dir) if bundle else None

    toc = [{"title": "Executive summary", "anchor": "executive-summary"}]
    toc += [{"title": s["heading"], "anchor": s["anchor"]} for s in sections]
    toc += [
        {"title": "Charts", "anchor": "charts"},
        {"title": "Deployment recommendation", "anchor": "deployment"},
        {"title": "Model leaderboard", "anchor": "leaderboard"},
        {"title": "Decision log", "anchor": "decision-log"},
        {"title": "Run metadata", "anchor": "run-metadata"},
        {"title": "Usage and cost", "anchor": "usage"},
    ]
    if report.appendix_notes or state.warnings:
        toc.append({"title": "Notes and warnings", "anchor": "notes"})

    deployment = report.deployment
    leaderboard_header, leaderboard_rows = leaderboard_table(state)
    evaluation = state.evaluation

    return {
        "title": f"{report.title} — {state.config.project}",
        "report_title": report.title,
        "subtitle": report.subtitle,
        "project": state.config.project,
        "run_id": state.run_id,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "grade": enum_value(evaluation.overall_grade) if evaluation else "",
        "acceptable": bool(evaluation.acceptable) if evaluation else False,
        "css_vars": Markup(theme.css_variables()),
        "metrics": [m.__dict__ for m in headline_metrics(state)],
        "executive_summary": Markup(markdown_to_html(report.executive_summary)),
        "toc": toc,
        "sections": sections,
        "orphan_charts": [_chart_context(a, base_dir) for a in orphans],
        "skipped_charts": skipped,
        "dashboard_href": dashboard_href,
        "deployment_rows": [
            ("Pattern", enum_value(deployment.pattern)),
            ("Rationale", deployment.rationale),
            ("Estimated latency (ms)", fmt_number(deployment.estimated_latency_ms)),
            ("Estimated throughput (rps)", fmt_number(deployment.estimated_throughput_rps)),
            ("Model size (MB)", fmt_number(deployment.model_size_mb)),
            ("Retraining cadence", deployment.retraining_cadence or "not specified"),
            ("Rollout strategy", deployment.rollout_strategy or "not specified"),
            ("Infrastructure", deployment.infrastructure_notes or "not specified"),
        ],
        "monitoring_plan": list(deployment.monitoring_plan),
        "deployment_risks": list(deployment.risks),
        "leaderboard_header": leaderboard_header,
        "leaderboard_rows": leaderboard_rows,
        "decisions": [d.__dict__ for d in decision_log(state)],
        "metadata_rows": run_metadata(state),
        "usage_rows": usage_rows(state),
        "appendix_notes": list(report.appendix_notes),
        "warnings": list(state.warnings),
    }


def render_html(state: RunState, report: FinalReport, base: Path | None = None) -> str:
    """Render the report to a single self-contained HTML document.

    Args:
        state: The run blackboard.
        report: The authored report.
        base: Directory the document will be written to, for relative links.

    Returns:
        The complete HTML document.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
        trim_blocks=True,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )
    template = environment.get_template(TEMPLATE_NAME)
    return template.render(**build_context(state, report, base=base))


def write_html(
    state: RunState, report: FinalReport, path: Path | str | None = None
) -> Path:
    """Render and write the HTML report.

    Args:
        state: The run blackboard.
        report: The authored report.
        path: Output path. Defaults to ``<run>/report/report.html``.

    Returns:
        The path written.
    """
    target = Path(path) if path else state.artifact_path("report", "report.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(state, report, base=target.parent), encoding="utf-8")
    return target


__all__ = ["MAX_INLINE_IMAGE_BYTES", "build_context", "render_html", "write_html"]
