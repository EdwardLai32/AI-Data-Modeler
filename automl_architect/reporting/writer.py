"""Report export dispatch.

The contract this module keeps: **a missing PDF library must never lose the
Markdown report.** Every renderer runs inside its own guard, records a warning on
failure, and the bundle comes back with whatever succeeded. ``write_report`` has
no raising path — the caller is an orchestrator finishing a run, and a broken
optional dependency at that point should cost a format, not the run.

The JSON dump of the :class:`~automl_architect.core.schemas.RunSummary` is always
written, regardless of the requested formats, because it is the machine-readable
record everything else can be rebuilt from.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..core.schemas import FinalReport, ReportBundle

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import RunState

logger = logging.getLogger(__name__)

#: Accepted spellings for each output format.
FORMAT_ALIASES: dict[str, str] = {
    "md": "markdown",
    "markdown": "markdown",
    "html": "html",
    "htm": "html",
    "pdf": "pdf",
    "ppt": "pptx",
    "pptx": "pptx",
    "powerpoint": "pptx",
    "slides": "pptx",
    "deck": "pptx",
    "json": "json",
}

SUPPORTED_FORMATS: tuple[str, ...] = ("markdown", "html", "pdf", "pptx", "json")


def normalise_formats(formats: list[str] | None) -> list[str]:
    """Resolve caller-supplied format names to canonical ones.

    Args:
        formats: Requested formats in any accepted spelling. ``None`` or empty
            means Markdown plus HTML.

    Returns:
        Canonical format names, de-duplicated, in a stable render order. Unknown
        names are dropped here and reported by :func:`write_report`.
    """
    requested = [str(f).strip().lower() for f in (formats or []) if str(f).strip()]
    if not requested:
        requested = ["markdown", "html"]
    resolved = {FORMAT_ALIASES[name] for name in requested if name in FORMAT_ALIASES}
    return [name for name in SUPPORTED_FORMATS if name in resolved]


def unknown_formats(formats: list[str] | None) -> list[str]:
    """Requested names that no renderer claims."""
    return [
        str(f)
        for f in (formats or [])
        if str(f).strip().lower() not in FORMAT_ALIASES and str(f).strip()
    ]


def _write_summary_json(state: RunState, path: Path) -> None:
    payload = state.to_summary().model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_report(
    state: RunState, report: FinalReport, formats: list[str]
) -> ReportBundle:
    """Render the report in every requested format.

    Args:
        state: The run blackboard. ``state.report`` is populated from ``report``
            when it is not already set, so the JSON dump contains the report.
        report: The Report Agent's authored report.
        formats: Requested output formats (``markdown``, ``html``, ``pdf``,
            ``pptx``, ``json``; several aliases are accepted).

    Returns:
        A :class:`ReportBundle` with a path per format that succeeded and a
        warning per format that did not. Never raises.
    """
    bundle = ReportBundle()
    if state.report is None:
        state.report = report

    for name in unknown_formats(formats):
        message = f"unknown report format '{name}' was requested and ignored"
        bundle.warnings.append(message)
        state.add_warning(message)

    wanted = normalise_formats(formats)

    def render_markdown_format() -> Path:
        from .markdown import write_markdown

        return write_markdown(state, report)

    def render_html_format() -> Path:
        from .html import write_html

        return write_html(state, report)

    def render_pdf_format() -> Path:
        from .pdf import render_pdf

        return render_pdf(state, report)

    def render_pptx_format() -> Path:
        from .pptx import render_pptx

        return render_pptx(state, report)

    renderers: dict[str, Callable[[], Path]] = {
        "markdown": render_markdown_format,
        "html": render_html_format,
        "pdf": render_pdf_format,
        "pptx": render_pptx_format,
    }
    attribute = {
        "markdown": "markdown_path",
        "html": "html_path",
        "pdf": "pdf_path",
        "pptx": "pptx_path",
    }

    for name in wanted:
        if name == "json":
            continue
        try:
            path = renderers[name]()
        except Exception as exc:  # one format failing must not lose the others
            message = f"{name} report export failed: {type(exc).__name__}: {exc}"
            logger.exception("%s export failed", name)
            bundle.warnings.append(message)
            state.add_warning(message)
            continue
        setattr(bundle, attribute[name], str(path))
        state.bus.artifact(str(path), kind=f"report:{name}")

    # The JSON dump is unconditional: it is the record the rest can be rebuilt
    # from, and it is the only format with no optional dependency.
    json_path = state.artifact_path("report", "run_summary.json")
    bundle.json_path = str(json_path)
    state.report_bundle = bundle
    try:
        _write_summary_json(state, json_path)
        state.bus.artifact(str(json_path), kind="report:json")
    except Exception as exc:
        bundle.json_path = None
        message = f"run summary JSON export failed: {type(exc).__name__}: {exc}"
        logger.exception("json export failed")
        bundle.warnings.append(message)
        state.add_warning(message)

    written = [
        name
        for name, field in attribute.items()
        if getattr(bundle, field, None)
    ]
    if bundle.json_path:
        written.append("json")
    state.bus.log(f"report written in: {', '.join(written) or 'no format'}")
    return bundle


__all__ = [
    "FORMAT_ALIASES",
    "SUPPORTED_FORMATS",
    "normalise_formats",
    "unknown_formats",
    "write_report",
]
