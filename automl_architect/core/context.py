"""Renders a :class:`DatasetProfile` into the prompt text agents reason over.

This module is the bridge between measured statistics and model reasoning, and
it is load-bearing in two ways:

*   **It is the cached prompt prefix.** The digest is built once per run and
    reused verbatim by every agent, so it must be deterministic. No timestamps,
    no run counters, no dict iteration order that could vary — anything volatile
    here silently costs every agent a cache miss.
*   **It is the anti-hallucination budget.** Agents are told never to invent a
    number. That only works if the numbers they need are actually present, so
    the digest errs toward including a statistic over omitting it, while capping
    per-column detail so a 400-column table still fits comfortably.
"""

from __future__ import annotations

from .schemas import (
    ColumnProfile,
    DatasetProfile,
    RunConfig,
    Severity,
)

# Detail budget. A wide table gets abbreviated per-column lines rather than a
# truncated column list — the model needs to know every column exists.
FULL_DETAIL_COLUMN_LIMIT = 60
TOP_VALUES_SHOWN = 6
MAX_CORRELATIONS = 25


def _num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        if value != value:  # NaN
            return "n/a"
        if value in (float("inf"), float("-inf")):
            return "inf"
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.{digits}g}"


def _pct(fraction: float | None) -> str:
    if fraction is None:
        return "n/a"
    return f"{fraction * 100:.2f}%"


def _bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _column_line(col: ColumnProfile, *, full: bool) -> str:
    bits: list[str] = [
        f"- `{col.name}` [{col.kind.value}, dtype={col.dtype}]",
        f"missing={col.n_missing} ({_pct(col.missing_fraction)})",
        f"unique={col.n_unique}",
    ]
    if col.cardinality_ratio:
        bits.append(f"card_ratio={_num(col.cardinality_ratio, 3)}")

    if col.mean is not None or col.minimum is not None:
        bits.append(
            f"min={_num(col.minimum)} p25={_num(col.quantiles.p25) if col.quantiles else 'n/a'} "
            f"median={_num(col.quantiles.p50) if col.quantiles else 'n/a'} "
            f"p75={_num(col.quantiles.p75) if col.quantiles else 'n/a'} max={_num(col.maximum)}"
        )
        bits.append(f"mean={_num(col.mean)} std={_num(col.std)}")
        if col.skewness is not None:
            bits.append(f"skew={_num(col.skewness, 3)}")
        if full and col.kurtosis is not None:
            bits.append(f"kurtosis={_num(col.kurtosis, 3)}")
        if col.outliers and col.outliers.n_outliers:
            bits.append(
                f"outliers={col.outliers.n_outliers} ({_pct(col.outliers.fraction)}, {col.outliers.method})"
            )
        if col.zero_fraction:
            bits.append(f"zeros={_pct(col.zero_fraction)}")
        if col.negative_fraction:
            bits.append(f"negatives={_pct(col.negative_fraction)}")

    if col.top_values:
        shown = col.top_values[: TOP_VALUES_SHOWN if full else 3]
        rendered = ", ".join(
            f"{v.value!r}:{v.count}({_pct(v.fraction)})" for v in shown
        )
        more = (
            f" (+{col.n_unique - len(shown)} more)"
            if col.n_unique > len(shown)
            else ""
        )
        bits.append(f"top=[{rendered}]{more}")

    if col.mean_string_length is not None:
        bits.append(f"str_len_mean={_num(col.mean_string_length, 3)}")
        if col.max_string_length is not None:
            bits.append(f"str_len_max={col.max_string_length}")
    if col.mean_token_count is not None:
        bits.append(f"tokens_mean={_num(col.mean_token_count, 3)}")

    if col.min_timestamp or col.max_timestamp:
        bits.append(f"range={col.min_timestamp} .. {col.max_timestamp}")
        if col.inferred_frequency:
            bits.append(f"freq={col.inferred_frequency}")
        if col.n_gaps is not None:
            bits.append(f"gaps={col.n_gaps}")
        if col.is_monotonic is not None:
            bits.append(f"monotonic={str(col.is_monotonic).lower()}")

    flags: list[str] = []
    if col.is_constant:
        flags.append("CONSTANT")
    if col.is_near_zero_variance:
        flags.append("NEAR_ZERO_VARIANCE")
    if col.looks_like_id:
        flags.append("ID_LIKE")
    if col.looks_like_geo:
        flags.append("GEO_LIKE")
    if col.looks_like_text:
        flags.append("FREE_TEXT")
    if col.detected_semantic_type:
        flags.append(col.detected_semantic_type.upper())
    if flags:
        bits.append("flags=" + "|".join(flags))

    return " | ".join(bits)


def render_profile(profile: DatasetProfile) -> str:
    """The measured-facts section. Deterministic for a given profile."""
    lines: list[str] = []
    add = lines.append

    add("## MEASURED DATASET FACTS")
    add("")
    add(
        "Every figure below was computed from the actual data by deterministic code. "
        "Treat these as ground truth. Do not restate them as your own findings "
        "without interpretation, and never contradict them."
    )
    add("")
    add("### Shape and integrity")
    add(f"- rows: {profile.n_rows:,}")
    add(f"- columns: {profile.n_columns:,}")
    add(f"- memory: {_bytes(profile.memory_bytes)}")
    add(
        f"- duplicate rows: {profile.n_duplicate_rows:,} ({_pct(profile.duplicate_fraction)})"
    )
    add(
        f"- missing cells: {profile.total_missing_cells:,} "
        f"({_pct(profile.missing_cell_fraction)} of all cells)"
    )
    add("")

    if profile.target:
        target = profile.target
        add("### Target variable")
        add(f"- name: `{target.name}` (kind={target.kind.value})")
        add(f"- missing: {target.n_missing:,}")
        if target.n_classes is not None:
            add(f"- distinct classes: {target.n_classes}")
        if target.class_counts:
            rendered = ", ".join(
                f"{c.value!r}: {c.count:,} ({_pct(c.fraction)})"
                for c in target.class_counts[:12]
            )
            add(f"- class distribution: {rendered}")
        if target.imbalance_ratio is not None:
            add(
                f"- imbalance ratio (majority/minority): {_num(target.imbalance_ratio, 3)} "
                f"-> {'IMBALANCED' if target.is_imbalanced else 'reasonably balanced'}"
            )
        if target.mean is not None:
            add(
                f"- mean={_num(target.mean)} std={_num(target.std)} "
                f"skew={_num(target.skewness, 3)}"
            )
        add("")

    add("### Semantic column groups")
    add(f"- temporal: {profile.temporal_columns or 'none detected'}")
    add(f"- geographic: {profile.geo_columns or 'none detected'}")
    add(f"- free text: {profile.text_columns or 'none detected'}")
    add(f"- identifier-like: {profile.identifier_columns or 'none detected'}")
    add(f"- constant: {profile.constant_columns or 'none'}")
    add("")

    full = profile.n_columns <= FULL_DETAIL_COLUMN_LIMIT
    add(f"### Column statistics ({profile.n_columns} columns)")
    if not full:
        add(
            f"_Abbreviated: this table has more than {FULL_DETAIL_COLUMN_LIMIT} columns, "
            "so per-column detail is condensed. All columns are listed._"
        )
    for col in profile.columns:
        add(_column_line(col, full=full))
    add("")

    if profile.target_correlations:
        add("### Correlation with target (strongest first)")
        for pair in profile.target_correlations[:MAX_CORRELATIONS]:
            add(
                f"- `{pair.left}` vs `{pair.right}`: {_num(pair.coefficient, 4)} ({pair.method})"
            )
        add("")

    if profile.highly_correlated_pairs:
        add("### Highly correlated feature pairs (multicollinearity risk)")
        for pair in profile.highly_correlated_pairs[:MAX_CORRELATIONS]:
            add(
                f"- `{pair.left}` <-> `{pair.right}`: {_num(pair.coefficient, 4)} ({pair.method})"
            )
        add("")

    if profile.leakage_findings:
        add("### Target-leakage candidates")
        add(
            "_A feature whose association with the target is near-perfect is usually "
            "leakage: information unavailable at prediction time. Judge each on whether "
            "it could plausibly be known before the outcome._"
        )
        for finding in profile.leakage_findings:
            add(
                f"- `{finding.column}`: score={_num(finding.score, 4)} "
                f"severity={finding.severity.value} method={finding.method} — {finding.reason}"
            )
        add("")

    if profile.quality_issues:
        add("### Data-quality issues detected")
        order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        for issue in sorted(profile.quality_issues, key=lambda i: order[i.severity]):
            scope = f" columns={issue.columns}" if issue.columns else ""
            add(f"- [{issue.severity.value.upper()}] {issue.code}: {issue.detail}{scope}")
        add("")

    return "\n".join(lines)


def render_run_config(config: RunConfig) -> str:
    """The operator's constraints, which agents must respect."""
    lines = [
        "## RUN CONSTRAINTS",
        "",
        "These were set by the operator. Work within them; do not propose plans "
        "that ignore them.",
        "",
        f"- project: {config.project}",
        f"- source kind: {config.source.kind.value}",
        f"- time budget: {config.time_budget_seconds}s total for the whole run",
        f"- max models to train: {config.max_experiments}",
        f"- cross-validation folds: {config.cv_folds}",
        f"- holdout test fraction: {config.test_size}",
        f"- hyperparameter tuning enabled: {str(config.enable_tuning).lower()}",
        f"- explainability enabled: {str(config.enable_explainability).lower()}",
        f"- human approval required before destructive steps: "
        f"{str(config.require_approval).lower()}",
        f"- self-improvement retries allowed: {config.max_replans}",
    ]
    if config.target_column:
        lines.append(f"- operator-specified target column: `{config.target_column}`")
    if config.task_type_override:
        lines.append(
            f"- operator-forced task type: {config.task_type_override.value} "
            "(do not override this)"
        )
    if config.primary_metric_override:
        lines.append(
            f"- operator-forced primary metric: {config.primary_metric_override}"
        )
    if config.min_acceptable_score is not None:
        lines.append(
            f"- minimum acceptable primary-metric score: {config.min_acceptable_score}"
        )
    if config.fairness_attributes:
        lines.append(
            f"- fairness attributes to audit: {config.fairness_attributes}"
        )
    if config.notes:
        lines.append(f"- operator notes: {config.notes}")
    lines.append("")
    return "\n".join(lines)


def build_run_context(profile: DatasetProfile, config: RunConfig) -> str:
    """The full cached context block: constraints, then measured facts.

    Called once per run by the orchestrator immediately after profiling, then
    frozen on the state. Order is constraints-first so the shorter, more
    consistently-shaped section leads.
    """
    return "\n".join([render_run_config(config), render_profile(profile)])
