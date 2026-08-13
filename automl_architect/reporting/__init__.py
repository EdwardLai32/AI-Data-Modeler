"""Chart rendering and multi-format report export.

The layering here mirrors the rest of the system: agents decide *what* to show
and write the prose, and this package presents it. Nothing in here interprets
results — :func:`render_charts` plots measurements, and the report renderers lay
out already-authored Markdown.

Entry points:

``render_charts(state, plan)``
    Every :class:`~automl_architect.core.schemas.ChartKind` as standalone HTML
    plus PNG, composed into one offline dashboard.
``write_report(state, report, formats)``
    Markdown, HTML, PDF, PowerPoint, and an always-written JSON dump of the run
    summary.

Imports are lazy: plotly, reportlab, and python-pptx are only loaded by the
renderer that needs them, so importing this package stays cheap.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_dashboard",
    "render_charts",
    "render_html",
    "render_markdown",
    "render_pdf",
    "render_pptx",
    "theme",
    "write_report",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "render_charts": (".charts", "render_charts"),
    "build_dashboard": (".dashboard", "build_dashboard"),
    "render_markdown": (".markdown", "render_markdown"),
    "render_html": (".html", "render_html"),
    "render_pdf": (".pdf", "render_pdf"),
    "render_pptx": (".pptx", "render_pptx"),
    "write_report": (".writer", "write_report"),
}


def __getattr__(name: str) -> Any:
    """Resolve a public symbol on first use.

    ``import_module`` is used rather than ``from . import x``: the latter looks
    the name up on this package again, which re-enters this hook and recurses
    while the submodule is still initialising.
    """
    from importlib import import_module

    if name == "theme":
        return import_module(".theme", __name__)
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        return getattr(import_module(module_name, __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
