"""Shared helpers for every report renderer.

Two jobs live here. The first is *reading the run*: pulling headline metrics, the
leaderboard, and the decision log off :class:`~automl_architect.core.state.RunState`
so that Markdown, HTML, PDF, and PowerPoint all report the same numbers instead of
each re-deriving them slightly differently.

The second is *parsing authored Markdown*. The Report Agent writes the section
bodies; the renderers must present that text, never rewrite it. Markdown and HTML
can pass it through, but PDF and PowerPoint need structured blocks (headings,
bullets, tables) to build flowables and slide bullets from. :func:`parse_markdown`
is that bridge — a deliberately small block-level parser, because the alternative
is dumping raw markup into a PDF.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from ..core.schemas import ChartArtifact, ExperimentResult, RunStatus, StepStatus

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import RunState
    from ..core.schemas import FinalReport

#: Metric-name fragments whose scores improve as they get smaller.
_LOWER_IS_BETTER = (
    "loss",
    "error",
    "mse",
    "rmse",
    "mae",
    "mape",
    "smape",
    "medae",
    "mdae",
    "median_ae",
    "msle",
    "rmsle",
    "brier",
    "entropy",
    "deviance",
    "perplexity",
    "aic",
    "bic",
    "distance",
    "inertia",
    "davies_bouldin",
    "noise_fraction",
    "outlier_fraction",
)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def enum_value(value: Any) -> str:
    """Return the wire value of an enum, or the string form of anything else."""
    if isinstance(value, Enum):
        return str(value.value)
    return "" if value is None else str(value)


def fmt_number(value: Any, digits: int = 4) -> str:
    """Format a number for display, keeping small values legible.

    Args:
        value: Any number, or ``None``.
        digits: Significant digits for the general case.

    Returns:
        A trimmed string, or ``"n/a"`` when the value is missing or not numeric.
    """
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    if number in (float("inf"), float("-inf")):
        return "inf" if number > 0 else "-inf"
    if number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"
    if abs(number) >= 1000 or abs(number) < 1e-4:
        return f"{number:,.{max(1, digits - 1)}g}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def fmt_percent(fraction: Any, digits: int = 1) -> str:
    """Format a 0-1 fraction as a percentage."""
    if fraction is None:
        return "n/a"
    try:
        return f"{float(fraction) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(fraction)


def fmt_duration(seconds: Any) -> str:
    """Format a second count as ``1h 02m 03s`` / ``2m 03s`` / ``4.1s``."""
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return "n/a"
    if total < 60:
        return f"{total:.1f}s"
    minutes, secs = divmod(int(round(total)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def fmt_timestamp(value: datetime | None) -> str:
    """Format a datetime as an ISO-ish minute-precision UTC string."""
    if value is None:
        return "n/a"
    stamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def slugify(text: str, fallback: str = "section") -> str:
    """Turn a heading into a stable HTML anchor / file-name fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or fallback


def higher_is_better(metric: str) -> bool:
    """Whether a larger value of ``metric`` is a better model.

    The execution layer owns the authoritative version of this
    (:func:`automl_architect.execution.metrics.higher_is_better`); this copy
    exists so the reporting layer can render a leaderboard without importing
    (and thus hard-depending on) the training modules. It must agree with it.
    """
    name = (metric or "").strip().lower().replace("-", "_").replace(" ", "_")
    if name.startswith("neg_"):
        # sklearn's sign-flipped scorers are maximised by construction, so the
        # loss token in the rest of the name must not flip the direction.
        return True
    return not any(token in name for token in _LOWER_IS_BETTER)


def metric_direction(state: RunState) -> bool:
    """Resolve metric polarity, preferring what the trainer recorded."""
    log = getattr(state, "experiments", None)
    if log is not None and getattr(log, "primary_metric", ""):
        return bool(log.higher_is_better)
    return higher_is_better(state.primary_metric)


# ---------------------------------------------------------------------------
# reading the run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """One headline number, with the label a reader needs to interpret it."""

    label: str
    value: str
    note: str = ""


@dataclass(frozen=True)
class Decision:
    """One recorded choice and the reason given for it."""

    agent: str
    decision: str
    rationale: str


def successful_experiments(state: RunState) -> list[ExperimentResult]:
    """Scored, non-failed experiments, best first."""
    log = getattr(state, "experiments", None)
    if log is None:
        return []
    scored = [
        r for r in log.results if not r.failed and r.primary_score is not None
    ]
    return sorted(
        scored,
        key=lambda r: float(r.primary_score or 0.0),
        reverse=metric_direction(state),
    )


def best_experiment(state: RunState) -> ExperimentResult | None:
    """The winning experiment: the trainer's pick, else the best score."""
    log = getattr(state, "experiments", None)
    if log is None:
        return None
    chosen = log.best()
    if chosen is not None:
        return chosen
    ranked = successful_experiments(state)
    return ranked[0] if ranked else None


def experiment_label(result: ExperimentResult) -> str:
    """A display name for an experiment, annotated for baseline/tuned runs."""
    name = result.label or enum_value(result.family)
    suffixes = []
    if result.is_baseline:
        suffixes.append("baseline")
    if result.tuned:
        suffixes.append("tuned")
    return f"{name} ({', '.join(suffixes)})" if suffixes else name


def headline_metrics(state: RunState) -> list[Metric]:
    """The numbers that belong in a report header or dashboard hero row."""
    metrics: list[Metric] = []
    best = best_experiment(state)
    metric_name = state.primary_metric
    if best is not None:
        metrics.append(
            Metric(
                label=metric_name,
                value=fmt_number(best.primary_score),
                note="best model, evaluation split",
            )
        )
        metrics.append(
            Metric(
                label="Winning model",
                value=experiment_label(best),
                note=f"{len(successful_experiments(state))} candidates scored",
            )
        )
    evaluation = getattr(state, "evaluation", None)
    if evaluation is not None:
        metrics.append(
            Metric(
                label="Grade",
                value=enum_value(evaluation.overall_grade),
                note="acceptable" if evaluation.acceptable else "not acceptable",
            )
        )
    profile = getattr(state, "profile", None)
    if profile is not None:
        metrics.append(
            Metric(
                label="Rows",
                value=fmt_number(profile.n_rows),
                note=f"{profile.n_columns} columns",
            )
        )
    if getattr(state, "feature_names", None):
        metrics.append(
            Metric(label="Features", value=fmt_number(len(state.feature_names)))
        )
    tuning = getattr(state, "tuning", None)
    if tuning is not None and tuning.ran and tuning.improvement is not None:
        metrics.append(
            Metric(
                label="Tuning gain",
                value=fmt_number(tuning.improvement),
                note=f"{tuning.n_trials_completed} trials",
            )
        )
    return metrics


def run_metadata(state: RunState) -> list[tuple[str, str]]:
    """Key/value rows describing how the run was configured and how it went."""
    config = state.config
    problem = getattr(state, "problem", None)
    # Reporting is the last step, so the run is still RUNNING and has no
    # `finished_at` while this table is built. Printing "running" / "n/a" into a
    # document the reader opens after the fact reads as a crashed run, which is
    # the opposite of true — so say what is actually the case at write time.
    status = enum_value(state.status)
    if state.status is RunStatus.RUNNING:
        failed = [s for s in state.steps if s.status is StepStatus.FAILED]
        status = (
            f"completing — {len(failed)} step(s) failed, see Appendix E"
            if failed
            else "completing — all steps succeeded"
        )
    finished = (
        fmt_timestamp(state.finished_at)
        if state.finished_at
        else "at report generation (this report is the final step)"
    )
    rows: list[tuple[str, str]] = [
        ("Run ID", state.run_id),
        ("Project", config.project),
        ("Status", status),
        ("Source", f"{enum_value(config.source.kind)}: {config.source.uri or 'in-memory'}"),
        ("Task", enum_value(state.task_type) or "undetermined"),
        ("Target", state.target or "none (unsupervised)"),
        ("Primary metric", state.primary_metric),
        ("Started", fmt_timestamp(state.started_at)),
        ("Finished", finished),
    ]
    # Only when both endpoints are known. A live elapsed-time reading would make
    # the document differ on every write, and a replan re-writes it — the writer
    # is required to be idempotent.
    if state.started_at and state.finished_at:
        rows.append(
            ("Duration", fmt_duration((state.finished_at - state.started_at).total_seconds()))
        )
    if problem is not None:
        rows.append(("Task confidence", enum_value(problem.confidence)))
    splits = getattr(state, "splits", None)
    if splits is not None and splits.strategy:
        sizes = splits.sizes()
        rows.append(
            (
                "Split",
                f"{splits.strategy} — train {sizes['train']:,}, "
                f"validation {sizes['validation']:,}, test {sizes['test']:,}",
            )
        )
    rows.append(("Random state", str(config.random_state)))
    rows.append(("Replans", str(getattr(state, "replans", 0))))
    return rows


def leaderboard_table(state: RunState) -> tuple[list[str], list[list[str]]]:
    """The experiment leaderboard as a header row plus string rows."""
    results = successful_experiments(state)
    failed = [
        r
        for r in (getattr(state, "experiments", None).results if state.experiments else [])
        if r.failed
    ]
    header = ["#", "Model", state.primary_metric, "CV mean", "Train (s)", "Features"]
    rows: list[list[str]] = []
    for rank, result in enumerate(results, start=1):
        cv_mean = (
            sum(result.cv_scores) / len(result.cv_scores) if result.cv_scores else None
        )
        rows.append(
            [
                str(rank),
                experiment_label(result),
                fmt_number(result.primary_score),
                fmt_number(cv_mean),
                fmt_number(result.train_seconds, digits=3),
                str(result.n_features_in or "n/a"),
            ]
        )
    for result in failed:
        rows.append(
            [
                "—",
                experiment_label(result),
                "failed",
                "—",
                "—",
                (result.error or "")[:60] or "—",
            ]
        )
    return header, rows


def usage_rows(state: RunState) -> list[tuple[str, str]]:
    """Token and cost totals for the run."""
    usage = state.usage
    return [
        ("LLM calls", fmt_number(usage.llm_calls)),
        ("Input tokens", fmt_number(usage.input_tokens)),
        ("Output tokens", fmt_number(usage.output_tokens)),
        ("Cache read tokens", fmt_number(usage.cache_read_tokens)),
        ("Cache write tokens", fmt_number(usage.cache_write_tokens)),
        ("Estimated cost (USD)", f"${usage.cost_usd:,.4f}"),
    ]


def _append(entries: list[Decision], agent: str, decision: str, rationale: str) -> None:
    text = (rationale or "").strip()
    if text:
        entries.append(Decision(agent=agent, decision=decision.strip(), rationale=text))


def decision_log(state: RunState) -> list[Decision]:
    """Every recorded decision and its stated reason, in pipeline order.

    This is the audit trail the product promises: if an agent changed the data or
    chose a model, the reason it gave appears here verbatim.
    """
    entries: list[Decision] = []

    understanding = getattr(state, "understanding", None)
    if understanding is not None:
        _append(
            entries,
            "Dataset",
            f"Readiness: {enum_value(understanding.data_readiness)}",
            understanding.readiness_rationale,
        )

    problem = getattr(state, "problem", None)
    if problem is not None:
        _append(
            entries,
            "Problem",
            f"Task {enum_value(problem.task_type)} on target "
            f"'{problem.target_column or 'none'}'",
            problem.rationale,
        )
        _append(
            entries,
            "Problem",
            f"Primary metric {problem.primary_metric}",
            problem.metric_rationale,
        )

    plan = getattr(state, "plan", None)
    if plan is not None:
        _append(entries, "Planner", "Overall strategy", plan.summary)
        if plan.revision_reason:
            _append(
                entries,
                "Planner",
                f"Plan revision {plan.revision}",
                plan.revision_reason,
            )
        for step in plan.ordered():
            _append(entries, "Planner", f"Step {step.order}: {step.title}", step.rationale)

    cleaning = getattr(state, "cleaning", None)
    if cleaning is not None:
        for item in cleaning.decisions:
            scope = ", ".join(item.columns) if item.columns else "whole table"
            strategy = f" [{enum_value(item.strategy)}]" if item.strategy else ""
            _append(
                entries,
                "Cleaning",
                f"{enum_value(item.action)}{strategy} on {scope}",
                item.rationale,
            )
        for note in cleaning.skipped_considerations:
            _append(entries, "Cleaning", "Deliberately skipped", note)

    features = getattr(state, "features", None)
    if features is not None:
        for item in features.decisions:
            inputs = ", ".join(item.input_columns) if item.input_columns else "all features"
            _append(
                entries,
                "Features",
                f"{enum_value(item.op)} on {inputs}",
                item.rationale,
            )

    selection = getattr(state, "model_selection", None)
    if selection is not None:
        _append(entries, "Model selection", "Comparative reasoning", selection.reasoning)
        for candidate in sorted(selection.candidates, key=lambda c: c.rank):
            _append(
                entries,
                "Model selection",
                f"#{candidate.rank} {enum_value(candidate.family)} "
                f"({enum_value(candidate.suitability)})",
                candidate.rationale,
            )
        _append(
            entries,
            "Model selection",
            f"Validation: {selection.validation_strategy}",
            selection.validation_rationale,
        )

    tuning_decision = getattr(state, "tuning_decision", None)
    if tuning_decision is not None:
        _append(
            entries,
            "Tuning",
            f"{enum_value(tuning_decision.method)} "
            f"({'worthwhile' if tuning_decision.worthwhile else 'not worthwhile'})",
            tuning_decision.rationale,
        )

    explain = getattr(state, "explainability", None)
    if explain is not None:
        _append(entries, "Explainability", "Method notes", explain.method_notes)

    evaluation = getattr(state, "evaluation", None)
    if evaluation is not None:
        _append(
            entries,
            "Evaluation",
            f"Grade {enum_value(evaluation.overall_grade)} — "
            f"{'acceptable' if evaluation.acceptable else 'not acceptable'}",
            evaluation.verdict_rationale,
        )
        _append(
            entries,
            "Evaluation",
            f"Recommended action: {enum_value(evaluation.recommended_action)}",
            evaluation.action_rationale,
        )

    insights = getattr(state, "insights", None)
    if insights is not None:
        for insight in insights.insights:
            _append(
                entries,
                "Insight",
                insight.headline,
                f"{insight.supporting_evidence} Recommended: {insight.recommended_action}",
            )

    report = getattr(state, "report", None)
    if report is not None:
        _append(
            entries,
            "Deployment",
            enum_value(report.deployment.pattern),
            report.deployment.rationale,
        )

    visualisation = getattr(state, "visualization_plan", None)
    if visualisation is not None:
        for spec in visualisation.charts:
            _append(entries, "Visualization", spec.title, spec.rationale)

    return entries


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def chart_artifacts(state: RunState) -> list[ChartArtifact]:
    """Rendered chart artefacts recorded on the state, in plan order."""
    bundle = getattr(state, "visualizations", None)
    if bundle is None:
        return []
    return [a for a in bundle.artifacts if a.rendered]


def charts_for_refs(
    refs: list[str], artifacts: list[ChartArtifact]
) -> list[ChartArtifact]:
    """Resolve a section's ``chart_refs`` against rendered artefacts.

    Agents reference charts loosely — by title, by chart kind, or by a slug — so
    matching is deliberately forgiving. Unmatched references are dropped rather
    than rendered as broken links.
    """
    if not refs:
        return []
    matched: list[ChartArtifact] = []
    seen: set[int] = set()
    for ref in refs:
        needle = slugify(str(ref))
        if not needle:
            continue
        for artifact in artifacts:
            key = id(artifact)
            if key in seen:
                continue
            candidates = {
                slugify(artifact.spec.title),
                slugify(enum_value(artifact.spec.kind)),
            }
            if artifact.html_path:
                candidates.add(slugify(artifact.html_path.rsplit("/", 1)[-1]))
            # Substring matching only for references long enough to be
            # unambiguous: "bar" must not claim every chart whose slug happens
            # to contain those three letters.
            fuzzy = len(needle) >= 5 and any(
                needle in c or c in needle for c in candidates if c
            )
            if needle in candidates or fuzzy:
                matched.append(artifact)
                seen.add(key)
    return matched


CHART_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Data & quality",
        (
            "missingness",
            "class_balance",
            "histogram",
            "box",
            "bar",
            "scatter",
            "line",
            "correlation_heatmap",
        ),
    ),
    (
        "Model performance",
        (
            "leaderboard",
            "roc_curve",
            "pr_curve",
            "confusion_matrix",
            "calibration_curve",
            "learning_curve",
            "prediction_distribution",
            "time_series_forecast",
        ),
    ),
    (
        "Drivers & diagnostics",
        (
            "feature_importance",
            "shap_summary",
            "partial_dependence",
            "residuals",
            "residual_histogram",
        ),
    ),
)


def group_artifacts(
    artifacts: list[ChartArtifact],
) -> list[tuple[str, list[ChartArtifact]]]:
    """Bucket chart artefacts into reader-facing sections, preserving order."""
    remaining = list(artifacts)
    grouped: list[tuple[str, list[ChartArtifact]]] = []
    for title, kinds in CHART_GROUPS:
        bucket = [a for a in remaining if enum_value(a.spec.kind) in kinds]
        if bucket:
            grouped.append((title, bucket))
            remaining = [a for a in remaining if a not in bucket]
    if remaining:
        grouped.append(("Other charts", remaining))
    return grouped


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

BlockKind = Literal[
    "heading", "paragraph", "bullets", "numbered", "table", "code", "quote", "rule"
]


@dataclass
class MarkdownBlock:
    """One block-level element of authored Markdown."""

    kind: BlockKind
    text: str = ""
    level: int = 0
    items: list[tuple[int, str]] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


_BULLET_RE = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_NUMBER_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_RULE_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def parse_markdown(text: str) -> list[MarkdownBlock]:
    """Parse authored Markdown into block-level elements.

    Only the constructs the Report Agent actually emits are handled: ATX
    headings, paragraphs, bullet and numbered lists, pipe tables, fenced code,
    block quotes, and horizontal rules. Inline markup is left in the text for
    :func:`inline_to_reportlab` or :func:`strip_inline` to deal with.

    Args:
        text: Markdown source.

    Returns:
        Blocks in document order.
    """
    blocks: list[MarkdownBlock] = []
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(part.strip() for part in paragraph).strip()
            if joined:
                blocks.append(MarkdownBlock(kind="paragraph", text=joined))
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_paragraph()
            fence = stripped[:3]
            index += 1
            body: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith(fence):
                body.append(lines[index])
                index += 1
            index += 1
            blocks.append(MarkdownBlock(kind="code", text="\n".join(body)))
            continue

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if _RULE_RE.match(line):
            flush_paragraph()
            blocks.append(MarkdownBlock(kind="rule"))
            index += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            blocks.append(
                MarkdownBlock(
                    kind="heading",
                    level=len(heading.group(1)),
                    text=heading.group(2).strip(),
                )
            )
            index += 1
            continue

        if "|" in stripped and index + 1 < len(lines) and _TABLE_SEP_RE.match(lines[index + 1]):
            flush_paragraph()
            rows = [_split_row(line)]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_split_row(lines[index]))
                index += 1
            width = max(len(row) for row in rows)
            padded = [row + [""] * (width - len(row)) for row in rows]
            blocks.append(MarkdownBlock(kind="table", rows=padded))
            continue

        bullet = _BULLET_RE.match(line)
        number = _NUMBER_RE.match(line)
        if bullet or number:
            flush_paragraph()
            kind: BlockKind = "bullets" if bullet else "numbered"
            items: list[tuple[int, str]] = []
            while index < len(lines):
                current = lines[index]
                match = _BULLET_RE.match(current) if kind == "bullets" else _NUMBER_RE.match(current)
                if match:
                    indent = len(match.group(1).replace("\t", "  ")) // 2
                    items.append((indent, match.group(3).strip()))
                    index += 1
                    continue
                if current.strip() and not _HEADING_RE.match(current) and items and current.startswith((" ", "\t")):
                    indent, previous = items[-1]
                    items[-1] = (indent, f"{previous} {current.strip()}")
                    index += 1
                    continue
                break
            blocks.append(MarkdownBlock(kind=kind, items=items))
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip().lstrip(">").strip())
                index += 1
            blocks.append(MarkdownBlock(kind="quote", text=" ".join(quote)))
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    return blocks


_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)[^)]*\)")
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<![\w*])(\*|_)(?=\S)([^*_]+?)(?<=\S)\1(?![\w*])")
_CODE_RE = re.compile(r"`([^`]+)`")
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)


def strip_inline(text: str) -> str:
    """Remove inline Markdown markup, leaving readable plain text."""
    out = _LINK_RE.sub(r"\1", text or "")
    out = _CODE_RE.sub(r"\1", out)
    out = _BOLD_RE.sub(r"\2", out)
    out = _ITALIC_RE.sub(r"\2", out)
    out = _STRIKE_RE.sub(r"\1", out)
    return out.strip()


def inline_to_reportlab(text: str) -> str:
    """Convert inline Markdown to the mini-HTML reportlab paragraphs accept.

    Escaping happens first so authored ``&`` or ``<`` in the source cannot break
    the PDF, then the supported inline constructs are re-introduced as tags.
    """
    out = html.escape(text or "", quote=False)
    out = _CODE_RE.sub(r'<font face="Courier">\1</font>', out)
    out = _LINK_RE.sub(r'<link href="\2" color="#2a78d6">\1</link>', out)
    out = _BOLD_RE.sub(r"<b>\2</b>", out)
    out = _ITALIC_RE.sub(r"<i>\2</i>", out)
    out = _STRIKE_RE.sub(r"<strike>\1</strike>", out)
    return out


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def bullets_from_markdown(text: str, limit: int = 6, max_chars: int = 180) -> list[str]:
    """Extract slide-ready bullet points from authored Markdown.

    Existing list items win, because the author already chose them. Failing
    that, leading sentences of the prose are used — a slide needs bullets, and
    inventing new text would violate "present, do not rewrite".

    Args:
        text: Markdown source.
        limit: Maximum bullets to return.
        max_chars: Truncation length per bullet.

    Returns:
        Plain-text bullet strings.
    """
    blocks = parse_markdown(text)
    bullets: list[str] = []
    for block in blocks:
        if block.kind in ("bullets", "numbered"):
            for indent, item in block.items:
                prefix = "    " * min(indent, 2)
                bullets.append(prefix + strip_inline(item))
        if len(bullets) >= limit:
            return _truncate_all(bullets[:limit], max_chars)

    if not bullets:
        for block in blocks:
            if block.kind in ("paragraph", "quote"):
                for sentence in _sentences(strip_inline(block.text)):
                    bullets.append(sentence)
                    if len(bullets) >= limit:
                        return _truncate_all(bullets, max_chars)
    return _truncate_all(bullets[:limit], max_chars)


def _truncate_all(items: list[str], max_chars: int) -> list[str]:
    out = []
    for item in items:
        text = item.strip()
        if len(text) > max_chars:
            text = text[: max_chars - 1].rsplit(" ", 1)[0] + "…"
        out.append(text)
    return out


def markdown_to_html(text: str) -> str:
    """Render authored Markdown to an HTML fragment.

    The ``markdown`` package does the work when it is importable. The fallback
    is escaped paragraphs rather than an exception: a report that loses its
    formatting is still a report, while a crash loses everything.

    Args:
        text: Markdown source.

    Returns:
        An HTML fragment (no wrapping ``<div>``).
    """
    source = text or ""
    try:
        import markdown as markdown_lib

        # Deliberately no ``nl2br``: an agent that hard-wraps its prose would
        # otherwise get a line break inside every paragraph.
        return markdown_lib.markdown(
            source,
            extensions=["tables", "fenced_code", "sane_lists"],
            output_format="html",
        )
    except Exception:  # missing or misbehaving renderer must not lose the text
        logger_fallback = html.escape(source, quote=False)
        blocks = [b.strip() for b in logger_fallback.split("\n\n") if b.strip()]
        return "\n".join(f"<p>{b.replace(chr(10), '<br>')}</p>" for b in blocks)


def report_sections(report: FinalReport) -> list[tuple[str, str, list[str]]]:
    """Ordered ``(heading, body_markdown, chart_refs)`` triples."""
    return [
        (section.heading, section.body_markdown, list(section.chart_refs))
        for section in report.ordered_sections()
    ]


__all__ = [
    "CHART_GROUPS",
    "Decision",
    "MarkdownBlock",
    "Metric",
    "best_experiment",
    "bullets_from_markdown",
    "chart_artifacts",
    "charts_for_refs",
    "decision_log",
    "enum_value",
    "experiment_label",
    "fmt_duration",
    "fmt_number",
    "fmt_percent",
    "fmt_timestamp",
    "group_artifacts",
    "headline_metrics",
    "higher_is_better",
    "inline_to_reportlab",
    "leaderboard_table",
    "markdown_to_html",
    "metric_direction",
    "parse_markdown",
    "report_sections",
    "run_metadata",
    "slugify",
    "strip_inline",
    "successful_experiments",
    "usage_rows",
]
