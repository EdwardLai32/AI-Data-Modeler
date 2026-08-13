"""Markdown rendering of the final report.

The section bodies are already authored by the Report Agent, so this module adds
only the scaffolding a reader needs to navigate them: a title block, a run
metadata table, a table of contents, the chart links, and the appendices.

The appendices are the part that makes the document auditable rather than
merely readable. Appendix A reproduces every recorded rationale verbatim — if an
agent dropped a column or picked a model, the reason it gave is in the document.
Appendix B is the token and cost ledger for the run.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.schemas import ChartArtifact, FinalReport
from .common import (
    chart_artifacts,
    charts_for_refs,
    decision_log,
    enum_value,
    fmt_number,
    leaderboard_table,
    run_metadata,
    slugify,
    usage_rows,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import RunState


def _cell(text: str) -> str:
    """Make a string safe for a Markdown table cell."""
    return (
        str(text or "")
        .replace("|", "\\|")
        .replace("\r\n", " ")
        .replace("\n", "<br>")
        .strip()
    )


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_No rows._", ""]
    lines = [
        "| " + " | ".join(_cell(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    lines.append("")
    return lines


def _relative(target: str | None, base: Path) -> str | None:
    """Path from the report directory to an artifact, POSIX-style for Markdown."""
    if not target:
        return None
    try:
        return Path(os.path.relpath(Path(target), base)).as_posix()
    except (OSError, ValueError):
        return Path(target).as_posix()


def _chart_links(artifacts: list[ChartArtifact], base: Path) -> list[str]:
    lines: list[str] = []
    for artifact in artifacts:
        png = _relative(artifact.png_path, base)
        html = _relative(artifact.html_path, base)
        title = artifact.spec.title
        if png:
            lines.append(f"![{title}]({png})")
        if html:
            suffix = " (interactive)" if png else ""
            lines.append(f"[{title}{suffix}]({html})")
        if artifact.caption:
            lines.append(f"_{artifact.caption}_")
        lines.append("")
    return lines


def render_markdown(state: RunState, report: FinalReport, base: Path | None = None) -> str:
    """Render the full report as Markdown.

    Args:
        state: The run blackboard, read for metadata, the leaderboard, the
            decision log, and usage totals.
        report: The Report Agent's authored report. Section bodies are emitted
            verbatim.
        base: Directory the document will be written to, used to make chart
            links relative. Defaults to ``<run>/report``.

    Returns:
        The complete Markdown document.
    """
    base_dir = base or state.artifact_path("report", "report.md").parent
    artifacts = chart_artifacts(state)
    sections = report.ordered_sections()
    lines: list[str] = [f"# {report.title}", ""]
    if report.subtitle:
        lines += [f"_{report.subtitle}_", ""]
    lines += [
        f"**Project:** {state.config.project} &nbsp;·&nbsp; "
        f"**Run:** `{state.run_id}` &nbsp;·&nbsp; "
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Run metadata",
        "",
    ]
    lines += _table(["Field", "Value"], [[k, v] for k, v in run_metadata(state)])

    # The Report Agent is asked for an "Executive Summary" section and also fills
    # `report.executive_summary`, so emitting both duplicates the heading in the
    # document and the table of contents. The agent's section wins when present.
    section_headings = {s.heading.strip().lower() for s in sections}
    own_summary = "executive summary" not in section_headings

    lines += ["## Contents", ""]
    toc: list[tuple[str, str]] = []
    if own_summary:
        toc.append(("Executive summary", "executive-summary"))
    toc += [(s.heading, slugify(s.heading, f"section-{s.order}")) for s in sections]
    toc += [
        ("Deployment parameters", "deployment-parameters"),
        ("Appendix A — Decision log", "appendix-a--decision-log"),
        ("Appendix B — Model leaderboard", "appendix-b--model-leaderboard"),
        ("Appendix C — Charts", "appendix-c--charts"),
        ("Appendix D — Usage and cost", "appendix-d--usage-and-cost"),
        ("Appendix E — Notes and warnings", "appendix-e--notes-and-warnings"),
    ]
    lines += [f"{i}. [{title}](#{anchor})" for i, (title, anchor) in enumerate(toc, 1)]
    if own_summary:
        lines += ["", "## Executive summary", "", report.executive_summary.strip(), ""]
    else:
        lines += [""]

    for section in sections:
        lines += [f"## {section.heading}", "", section.body_markdown.strip(), ""]
        matched = charts_for_refs(section.chart_refs, artifacts)
        if matched:
            lines += _chart_links(matched, base_dir)

    # Titled "parameters" rather than "recommendation": the agent's own
    # Deployment Suggestions section carries the argument, and this block is the
    # measured fields behind it (latency, model size, throughput). Two sections
    # both called "recommendation" read as a duplication bug.
    deployment = report.deployment
    lines += ["## Deployment parameters", ""]
    lines += _table(
        ["Field", "Value"],
        [
            ["Pattern", enum_value(deployment.pattern)],
            ["Rationale", deployment.rationale],
            ["Estimated latency (ms)", fmt_number(deployment.estimated_latency_ms)],
            ["Estimated throughput (rps)", fmt_number(deployment.estimated_throughput_rps)],
            ["Model size (MB)", fmt_number(deployment.model_size_mb)],
            ["Retraining cadence", deployment.retraining_cadence or "not specified"],
            ["Rollout strategy", deployment.rollout_strategy or "not specified"],
            ["Infrastructure", deployment.infrastructure_notes or "not specified"],
        ],
    )
    if deployment.monitoring_plan:
        lines += ["**Monitoring plan**", ""]
        lines += [f"- {item}" for item in deployment.monitoring_plan] + [""]
    if deployment.risks:
        lines += ["**Risks**", ""]
        lines += [f"- {item}" for item in deployment.risks] + [""]

    lines += ["## Appendix A — Decision log", ""]
    lines += [
        "Every choice an agent made, with the reason it gave. Rationales are "
        "reproduced verbatim.",
        "",
    ]
    lines += _table(
        ["Agent", "Decision", "Rationale"],
        [[d.agent, d.decision, d.rationale] for d in decision_log(state)],
    )

    lines += ["## Appendix B — Model leaderboard", ""]
    header, rows = leaderboard_table(state)
    lines += _table(header, rows)

    lines += ["## Appendix C — Charts", ""]
    if artifacts:
        lines += _table(
            ["Chart", "Kind", "Interactive", "Image"],
            [
                [
                    a.spec.title,
                    enum_value(a.spec.kind),
                    _relative(a.html_path, base_dir) or "—",
                    _relative(a.png_path, base_dir) or "—",
                ]
                for a in artifacts
            ],
        )
    else:
        lines += ["_No charts were rendered._", ""]
    skipped = [
        a
        for a in (state.visualizations.artifacts if state.visualizations else [])
        if not a.rendered
    ]
    if skipped:
        lines += ["Charts that could not be rendered:", ""]
        lines += [
            f"- **{a.spec.title}** ({enum_value(a.spec.kind)}): {a.error or 'not rendered'}"
            for a in skipped
        ]
        lines += [""]

    lines += ["## Appendix D — Usage and cost", ""]
    lines += _table(["Measure", "Value"], [[k, v] for k, v in usage_rows(state)])

    lines += ["## Appendix E — Notes and warnings", ""]
    notes = list(report.appendix_notes)
    if notes:
        lines += [f"- {note}" for note in notes] + [""]
    if state.warnings:
        lines += ["**Run warnings**", ""]
        lines += [f"- {warning}" for warning in state.warnings] + [""]
    if not notes and not state.warnings:
        lines += ["_No warnings were recorded._", ""]

    return "\n".join(lines).rstrip() + "\n"


def write_markdown(
    state: RunState, report: FinalReport, path: Path | str | None = None
) -> Path:
    """Render and write the Markdown report.

    Args:
        state: The run blackboard.
        report: The authored report.
        path: Output path. Defaults to ``<run>/report/report.md``.

    Returns:
        The path written.
    """
    target = Path(path) if path else state.artifact_path("report", "report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(state, report, base=target.parent), encoding="utf-8")
    return target


__all__ = ["render_markdown", "write_markdown"]
