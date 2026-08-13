"""Command-line interface.

Entry points ``automl-architect`` and ``amla`` both resolve here.

The design goal for ``amla run`` is that the terminal shows what the system is
*thinking*, not just that it is busy. Every panel on screen is driven by the same
:class:`~automl_architect.core.events.EventBus` the API streams over HTTP, so the
CLI and the web UI can never disagree about what happened. Rendering happens on
the main thread from a snapshot the bus callback mutates under a lock; the
callback itself does no I/O, because it runs on the orchestrator's thread and
must not slow the run down.

Heavy imports (pandas, sklearn, uvicorn) are deferred into the command bodies
that need them, so ``amla --help`` and ``amla doctor`` stay fast even on a
half-broken install — which is exactly when you need ``doctor`` to work.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import Settings, get_settings
from .core.schemas import (
    EventKind,
    ExperimentLog,
    RunConfig,
    RunEvent,
    RunStatus,
    RunSummary,
    Severity,
    TaskType,
    params_to_dict,
)

app = typer.Typer(
    name="amla",
    help="AutoML Architect — an autonomous multi-agent AI data scientist.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
error_console = Console(stderr=True)

#: User-facing format name -> the value RunConfig.report_formats expects.
_FORMAT_ALIASES = {
    "md": "markdown",
    "markdown": "markdown",
    "html": "html",
    "pdf": "pdf",
    "pptx": "pptx",
    "ppt": "pptx",
    "json": "json",
}

_STATUS_STYLES = {
    RunStatus.PENDING: "dim",
    RunStatus.RUNNING: "cyan",
    RunStatus.AWAITING_APPROVAL: "yellow",
    RunStatus.REPLANNING: "magenta",
    RunStatus.COMPLETED: "green",
    RunStatus.FAILED: "red",
    RunStatus.CANCELLED: "yellow",
}

#: Sibling modules the pipeline needs. Checked by ``amla doctor``.
_INTERNAL_MODULES = (
    ("orchestrator", "automl_architect.runner"),
    ("agents", "automl_architect.agents"),
    ("ingestion", "automl_architect.ingestion.router"),
    ("profiling", "automl_architect.profiling.profiler"),
    ("execution", "automl_architect.execution.trainer"),
    ("reporting", "automl_architect.reporting.writer"),
    ("storage", "automl_architect.storage.repository"),
    ("api", "automl_architect.api.app"),
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _settings() -> Settings:
    return get_settings()


def _repository() -> Any:
    from .storage.repository import RunRepository

    return RunRepository()


def _manager() -> Any:
    from .api.service import get_run_manager

    return get_run_manager()


def _fail(message: str, code: int = 1) -> None:
    """Print an error and exit."""
    error_console.print(f"[bold red]error[/] {escape(message)}")
    raise typer.Exit(code)


def _enable_offline() -> None:
    """Switch the process into keyless mode before anything reads settings.

    Set through the environment rather than by mutating a ``Settings`` instance,
    because the orchestrator, agents, and API each resolve their own settings via
    ``get_settings()``. The cached singleton and the shared LLM client are both
    dropped so nothing keeps a pre-offline view.
    """
    import os

    from .config import reset_settings_cache
    from .core.llm import reset_llm_client

    os.environ["AUTOML_OFFLINE"] = "1"
    reset_settings_cache()
    reset_llm_client()
    console.print(
        "[bold yellow]offline[/] no model calls will be made; agents decide from "
        "the deterministic rule engine [dim](reasoning is rule-derived, not inferred)[/]"
    )


def _status_text(status: RunStatus | None) -> Text:
    if status is None:
        return Text("unknown", style="dim")
    return Text(status.value, style=_STATUS_STYLES.get(status, "white"))


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _stamp(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else "-"


def _short_stamp(value: datetime | None) -> str:
    return value.strftime("%m-%d %H:%M") if value else "-"


def _resolve_source(source: str, query: str | None = None) -> Any:
    from .api.schemas import data_source_from_uri

    return data_source_from_uri(source, query=query)


def _parse_task(task: str | None) -> TaskType | None:
    if not task:
        return None
    try:
        return TaskType(task.strip().lower())
    except ValueError:
        supported = ", ".join(t.value for t in TaskType if t.is_supported)
        _fail(f"unknown task type {task!r}. Supported end-to-end: {supported}")
    return None


def _parse_formats(formats: str) -> list[str]:
    out: list[str] = []
    for raw in formats.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        resolved = _FORMAT_ALIASES.get(key)
        if resolved is None:
            _fail(
                f"unknown report format {key!r}; choose from "
                f"{sorted(set(_FORMAT_ALIASES))}"
            )
        elif resolved not in out:
            out.append(resolved)
    return out or ["markdown"]


def _load_summary(run_id: str) -> RunSummary:
    summary = _manager().summary(run_id)
    if summary is None:
        _fail(f"unknown run {run_id!r}. Try 'amla list'.")
    return summary  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Live progress display
# ---------------------------------------------------------------------------


class RunDisplay:
    """Mutable snapshot of a run's progress, fed by the event bus.

    :meth:`on_event` runs on the orchestrator's thread and only touches memory
    under a lock. All rendering and printing happens on the main thread, which
    keeps rich's console single-threaded and keeps the run's critical path free
    of terminal I/O.
    """

    def __init__(self, config: RunConfig, *, history: int = 8) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._scrollback: deque[str] = deque()
        self.decisions: deque[tuple[str, str]] = deque(maxlen=history)
        self.current_step: str = "starting"
        self.current_agent: str = ""
        self.steps_done: int = 0
        self.steps_failed: int = 0
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.cache_read: int = 0
        self.cost_usd: float = 0.0
        self.llm_calls: int = 0
        self.warnings: int = 0
        self.metrics: dict[str, float] = {}
        self.thinking: str = ""
        self.started = time.monotonic()

    # -- event ingestion (runs on the run thread) --------------------------

    def on_event(self, event: RunEvent) -> None:
        """Fold one event into the snapshot. Must stay non-blocking."""
        with self._lock:
            agent = event.agent.value if event.agent else ""
            message = event.message.strip()

            if event.kind is EventKind.STEP_STARTED:
                self.current_step = message or event.step_id or "working"
                self.current_agent = agent
                self._scrollback.append(f"[cyan]>[/] {escape(self.current_step)}")
            elif event.kind is EventKind.STEP_COMPLETED:
                self.steps_done += 1
                took = f" [dim]({event.duration_seconds:.1f}s)[/]" if event.duration_seconds else ""
                self._scrollback.append(f"[green]v[/] {escape(message or self.current_step)}{took}")
            elif event.kind is EventKind.STEP_FAILED:
                self.steps_failed += 1
                self._scrollback.append(f"[red]x[/] {escape(message)}")
            elif event.kind is EventKind.STEP_SKIPPED:
                self._scrollback.append(f"[dim]- skipped: {escape(message)}[/]")
            elif event.kind is EventKind.AGENT_DECISION:
                self.decisions.append((agent, message))
                self._scrollback.append(
                    f"[magenta]{escape(agent or 'agent')}[/] {escape(message)}"
                )
            elif event.kind is EventKind.AGENT_THINKING:
                self.thinking = message[:400]
            elif event.kind is EventKind.WARNING:
                self.warnings += 1
                self._scrollback.append(f"[yellow]![/] {escape(message)}")
            elif event.kind is EventKind.METRIC_RECORDED:
                payload = params_to_dict(event.payload)
                name, value = payload.get("metric"), payload.get("value")
                if isinstance(name, str) and isinstance(value, (int, float)):
                    self.metrics[name] = float(value)
            elif event.kind is EventKind.APPROVAL_REQUESTED:
                self._scrollback.append(f"[yellow]?[/] approval needed: {escape(message)}")
            elif event.kind is EventKind.REPLAN_TRIGGERED:
                self._scrollback.append(f"[magenta]~[/] replanning: {escape(message)}")
            elif event.kind in (EventKind.RUN_FAILED, EventKind.RUN_CANCELLED):
                self._scrollback.append(f"[red]{escape(message)}[/]")

            if event.tokens_in:
                self.tokens_in += event.tokens_in
            if event.tokens_out:
                self.tokens_out += event.tokens_out
            if event.cache_read_tokens:
                self.cache_read += event.cache_read_tokens
            if event.cost_usd:
                self.cost_usd += event.cost_usd
                self.llm_calls += 1

    # -- rendering (main thread) -------------------------------------------

    def drain(self) -> list[str]:
        """Take the accumulated scrollback lines, leaving the buffer empty."""
        with self._lock:
            lines = list(self._scrollback)
            self._scrollback.clear()
        return lines

    def renderable(self, status: RunStatus | None) -> RenderableType:
        """The live panel: status, current step, recent decisions, spend."""
        with self._lock:
            elapsed = time.monotonic() - self.started
            header = Table.grid(padding=(0, 2))
            header.add_column(style="dim", justify="right")
            header.add_column()
            header.add_row("run", self.config.run_id)
            header.add_row("project", self.config.project)
            header.add_row("source", escape(self.config.source.uri or self.config.source.kind.value))
            header.add_row(
                "elapsed",
                f"{_duration(elapsed)} [dim]of {_duration(self.config.time_budget_seconds)} budget[/]",
            )
            header.add_row("status", _status_text(status))
            header.add_row(
                "steps",
                f"{self.steps_done} done"
                + (f", [red]{self.steps_failed} failed[/]" if self.steps_failed else "")
                + (f", [yellow]{self.warnings} warnings[/]" if self.warnings else ""),
            )
            header.add_row(
                "spend",
                f"[green]${self.cost_usd:.4f}[/] [dim]| {self.tokens_in:,} in / "
                f"{self.tokens_out:,} out / {self.cache_read:,} cached | "
                f"{self.llm_calls} calls[/]",
            )
            if self.metrics:
                header.add_row(
                    "metrics",
                    "  ".join(f"{k}=[bold]{v:.4g}[/]" for k, v in list(self.metrics.items())[-4:]),
                )

            blocks: list[RenderableType] = [header]

            label = f"{self.current_agent + ': ' if self.current_agent else ''}{self.current_step}"
            if status in (RunStatus.RUNNING, RunStatus.REPLANNING, None):
                blocks.append(Spinner("dots", text=Text(label, style="cyan")))
            elif status is RunStatus.AWAITING_APPROVAL:
                blocks.append(Text(f"paused: {label}", style="yellow"))
            else:
                blocks.append(Text(label, style="dim"))

            if self.decisions:
                table = Table(
                    show_header=True, header_style="dim", box=None, padding=(0, 1), expand=True
                )
                table.add_column("agent", style="magenta", no_wrap=True)
                table.add_column("decision", overflow="fold")
                for agent, message in self.decisions:
                    table.add_row(agent or "-", escape(message[:300]))
                blocks.append(table)

        return Panel(
            Group(*blocks),
            title="[bold]AutoML Architect[/]",
            border_style="cyan" if status is RunStatus.RUNNING else "dim",
        )


# ---------------------------------------------------------------------------
# amla run
# ---------------------------------------------------------------------------


@app.command()
def run(
    source: str = typer.Argument(
        ..., help="Dataset path, URL, connection string, or kaggle: slug."
    ),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Target column. Inferred when omitted."
    ),
    project: str = typer.Option("default", "--project", "-p", help="Project namespace."),
    time_budget: int = typer.Option(
        900, "--time-budget", help="Wall-clock budget for the whole run, in seconds."
    ),
    max_experiments: int = typer.Option(
        8, "--max-experiments", help="Cap on models trained."
    ),
    metric: str | None = typer.Option(
        None, "--metric", help="Force the primary metric, e.g. roc_auc, rmse."
    ),
    task: str | None = typer.Option(
        None, "--task", help="Force the task type, e.g. binary_classification."
    ),
    approve: bool = typer.Option(
        False, "--approve", help="Pause for confirmation before destructive steps."
    ),
    no_tuning: bool = typer.Option(False, "--no-tuning", help="Skip hyperparameter search."),
    formats: str = typer.Option(
        "md,html", "--format", help="Report formats: md,html,pdf,pptx,json."
    ),
    open_report: bool = typer.Option(
        False, "--open", help="Open the report when the run finishes."
    ),
    max_rows: int | None = typer.Option(None, "--max-rows", help="Sample cap for huge tables."),
    query: str | None = typer.Option(None, "--query", help="SQL query, for database sources."),
    fairness: str = typer.Option(
        "", "--fairness", help="Comma-separated attributes to audit for fairness."
    ),
    notes: str = typer.Option("", "--notes", help="Operator notes passed to the agents."),
    plain: bool = typer.Option(False, "--plain", help="Disable the live display."),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Run with zero model calls and no API key. Agents decide from a "
        "deterministic rule engine; training, tuning, SHAP, and reports are "
        "unaffected.",
    ),
) -> None:
    """Profile a dataset, plan a pipeline, train models, and write a report."""
    if offline:
        _enable_offline()
    formats_resolved = _parse_formats(formats)
    config = RunConfig(
        project=project,
        source=_resolve_source(source, query),
        target_column=target,
        task_type_override=_parse_task(task),
        primary_metric_override=metric,
        time_budget_seconds=time_budget,
        max_experiments=max_experiments,
        max_rows=max_rows,
        require_approval=approve,
        enable_tuning=not no_tuning,
        report_formats=formats_resolved,
        fairness_attributes=[a.strip() for a in fairness.split(",") if a.strip()],
        notes=notes,
    )

    manager = _manager()
    display = RunDisplay(config)
    try:
        handle = manager.start(config)
    except Exception as exc:
        _fail(f"could not start the run: {exc}")
        return
    unsubscribe = handle.bus.subscribe(display.on_event)

    console.print(
        f"[bold cyan]started[/] {handle.run_id} [dim]-> {config.source.uri or config.source.kind.value}[/]"
    )
    try:
        if plain:
            _follow_plain(handle, display, manager)
        else:
            _follow_live(handle, display, manager)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupt: requesting cancellation…[/]")
        manager.cancel(handle.run_id)
        if handle.thread is not None:
            handle.thread.join(timeout=30)
    finally:
        unsubscribe()

    summary = manager.wait(handle.run_id, timeout=10) or _load_summary(handle.run_id)
    _print_run_summary(summary, formats_resolved)
    if open_report:
        _open_report(summary, formats_resolved)
    if summary.status is RunStatus.FAILED:
        raise typer.Exit(1)


def _follow_live(handle: Any, display: RunDisplay, manager: Any) -> None:
    """Render the live panel until the run thread exits."""
    from rich.live import Live

    with Live(
        display.renderable(handle.status),
        console=console,
        refresh_per_second=8,
        vertical_overflow="visible",
    ) as live:
        while handle.is_running:
            for line in display.drain():
                console.print(line)
            _handle_pending_approvals(handle, manager, live=live)
            live.update(display.renderable(handle.status))
            time.sleep(0.12)
        for line in display.drain():
            console.print(line)
        live.update(display.renderable(handle.status))


def _follow_plain(handle: Any, display: RunDisplay, manager: Any) -> None:
    """Line-by-line output, for logs and non-TTY environments."""
    while handle.is_running:
        for line in display.drain():
            console.print(line)
        _handle_pending_approvals(handle, manager, live=None)
        time.sleep(0.2)
    for line in display.drain():
        console.print(line)


def _handle_pending_approvals(handle: Any, manager: Any, *, live: Any) -> None:
    """Prompt for any approval the run is blocked on."""
    pending = handle.control.approvals()
    for request in [item for item in pending if item.decision == "pending"]:
        if live is not None:
            live.stop()
        console.print()
        detail = Table.grid(padding=(0, 2))
        detail.add_column(style="dim", justify="right")
        detail.add_column()
        detail.add_row("step", request.step_id)
        detail.add_row("agent", request.agent.value)
        detail.add_row("severity", request.severity.value)
        if request.affected_columns:
            detail.add_row("columns", escape(", ".join(request.affected_columns[:20])))
        if request.affected_row_estimate:
            detail.add_row("rows affected", f"{request.affected_row_estimate:,}")
        for line in request.details[:8]:
            detail.add_row("", escape(line))
        console.print(
            Panel(
                Group(Text(request.action_summary, style="bold"), detail),
                title="[yellow]approval required[/]",
                border_style="yellow",
            )
        )
        if sys.stdin is None or not sys.stdin.isatty():
            # Nobody can answer. Rejecting is the safe default for a destructive
            # step, and it unblocks the run instead of stalling it for an hour.
            console.print(
                "[yellow]no interactive terminal: rejecting this step. Approve it "
                f"over the API instead: POST /api/runs/{handle.run_id}/approvals/"
                f"{request.request_id}[/]"
            )
            approved = False
            note = "auto-rejected: no interactive terminal was available"
        else:
            approved = typer.confirm("Apply this step?", default=True)
            note = None if approved else "rejected at the CLI prompt"
        resolved, _ = manager.resolve_approval(
            handle.run_id,
            request.request_id,
            "approved" if approved else "rejected",
            note=note,
            decided_by="cli",
        )
        if resolved is None:
            console.print(
                f"[red]could not record a decision for {request.request_id}; "
                "the run may stay suspended[/]"
            )
        else:
            console.print(
                f"[{'green' if approved else 'red'}]{resolved.decision}[/] "
                f"{escape(request.action_summary)}"
            )
        if live is not None:
            live.start()


def _print_run_summary(summary: RunSummary, formats: list[str]) -> None:
    """Everything worth reading after a run finishes."""
    console.print()
    head = Table.grid(padding=(0, 2))
    head.add_column(style="dim", justify="right")
    head.add_column()
    head.add_row("run", summary.run_id)
    head.add_row("status", _status_text(summary.status))
    head.add_row("duration", _duration(summary.duration_seconds))
    head.add_row(
        "spend",
        f"${summary.usage.cost_usd:.4f} over {summary.usage.llm_calls} model calls "
        f"({summary.usage.input_tokens:,} in / {summary.usage.output_tokens:,} out)",
    )
    if summary.problem:
        head.add_row("task", summary.problem.task_type.value)
        head.add_row("target", summary.problem.target_column or "-")
        head.add_row("metric", summary.problem.primary_metric)
    if summary.error:
        head.add_row("error", f"[red]{escape(summary.error)}[/]")
    console.print(
        Panel(
            head,
            title=f"[bold]{'completed' if summary.status is RunStatus.COMPLETED else summary.status.value}[/]",
            border_style=_STATUS_STYLES.get(summary.status, "dim"),
        )
    )

    if summary.experiments and summary.experiments.results:
        console.print(_leaderboard_table(summary.experiments))

    if summary.tuning and summary.tuning.ran:
        tuning = summary.tuning
        parts = [f"{tuning.n_trials_completed} trials"]
        if tuning.best_score is not None:
            parts.append(f"best={tuning.best_score:.6g}")
        if tuning.baseline_score is not None:
            parts.append(f"baseline={tuning.baseline_score:.6g}")
        if tuning.improvement is not None:
            parts.append(f"delta={tuning.improvement:+.6g}")
        console.print(f"[bold]tuning[/] {tuning.method.value}: " + ", ".join(parts))
    elif summary.tuning and summary.tuning.skipped_reason:
        console.print(f"[dim]tuning skipped: {escape(summary.tuning.skipped_reason)}[/]")

    if summary.evaluation:
        evaluation = summary.evaluation
        console.print(
            f"[bold]evaluation[/] grade [bold]{evaluation.overall_grade}[/], "
            f"{'acceptable' if evaluation.acceptable else '[red]not acceptable[/]'}, "
            f"action={evaluation.recommended_action}"
        )
        console.print(f"  [dim]{escape(evaluation.verdict_rationale[:400])}[/]")

    if summary.explainability and summary.explainability.global_attributions:
        table = Table(title="top drivers", box=None, header_style="dim", padding=(0, 1))
        table.add_column("feature")
        table.add_column("importance", justify="right")
        table.add_column("direction", style="dim")
        for item in summary.explainability.global_attributions[:8]:
            table.add_row(escape(item.feature), f"{item.importance:.4f}", item.direction)
        console.print(table)

    if summary.insights:
        console.print(
            Panel(
                escape(summary.insights.executive_summary),
                title="[bold]executive summary[/]",
                border_style="green",
            )
        )
        for insight in summary.insights.insights[:5]:
            console.print(f"  [green]*[/] {escape(insight.headline)}")

    _print_report_paths(summary, formats)

    if summary.warnings:
        console.print(f"\n[yellow]{len(summary.warnings)} warning(s):[/]")
        for warning in summary.warnings[:10]:
            console.print(f"  [yellow]![/] {escape(warning)}")
        if len(summary.warnings) > 10:
            console.print(f"  [dim]… and {len(summary.warnings) - 10} more[/]")

    console.print(
        f"\n[dim]artifacts: {escape(summary.artifact_dir or '-')}"
        f"\nask questions: amla ask {summary.run_id} \"why did it choose that model?\"[/]"
    )


def _leaderboard_table(log: ExperimentLog) -> Table:
    """The trained-model comparison, best first, failures included."""
    table = Table(
        title=f"leaderboard ({log.primary_metric})",
        header_style="dim",
        box=None,
        padding=(0, 1),
    )
    table.add_column("", width=1)
    table.add_column("family")
    table.add_column(log.primary_metric or "score", justify="right")
    table.add_column("cv mean", justify="right")
    table.add_column("train", justify="right")
    table.add_column("notes", style="dim", overflow="fold")

    def sort_key(result: Any) -> tuple[bool, float]:
        score = result.primary_score
        if score is None:
            return (True, 0.0)
        return (False, -score if log.higher_is_better else score)

    for result in sorted(log.results, key=sort_key):
        marker = "[green]*[/]" if result.experiment_id == log.best_experiment_id else ""
        if result.failed:
            table.add_row(
                marker,
                f"[red]{result.family.value}[/]",
                "-",
                "-",
                "-",
                escape((result.error or "failed")[:80]),
            )
            continue
        cv = (
            f"{sum(result.cv_scores) / len(result.cv_scores):.4g}"
            if result.cv_scores
            else "-"
        )
        notes = ", ".join(
            filter(
                None,
                [
                    "baseline" if result.is_baseline else "",
                    "tuned" if result.tuned else "",
                    f"{result.n_features_in} features" if result.n_features_in else "",
                ],
            )
        )
        table.add_row(
            marker,
            result.family.value,
            f"{result.primary_score:.6g}" if result.primary_score is not None else "-",
            cv,
            f"{result.train_seconds:.2f}s",
            escape(notes),
        )
    return table


def _print_report_paths(summary: RunSummary, formats: list[str]) -> None:
    from .api.service import available_report_formats, locate_report

    found: list[tuple[str, Path]] = []
    for key in formats or available_report_formats(summary.run_id, summary):
        path = locate_report(summary.run_id, summary, key)
        if path is not None:
            found.append((key, path))
    if not found:
        return
    console.print("\n[bold]reports[/]")
    for key, path in found:
        console.print(f"  {key:<9} {escape(str(path))}")


def _open_report(summary: RunSummary, formats: list[str]) -> None:
    """Open the most presentable report format available."""
    from .api.service import locate_report

    for key in ("html", "pdf", "pptx", "markdown", *formats):
        path = locate_report(summary.run_id, summary, key)
        if path is not None:
            typer.launch(str(path))
            return
    console.print("[yellow]no rendered report to open[/]")


# ---------------------------------------------------------------------------
# amla profile
# ---------------------------------------------------------------------------


@app.command()
def profile(
    source: str = typer.Argument(..., help="Dataset path, URL, or connection string."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column."),
    max_rows: int | None = typer.Option(None, "--max-rows", help="Row cap for the load."),
    query: str | None = typer.Option(None, "--query", help="SQL query, for database sources."),
    columns: int = typer.Option(40, "--columns", help="How many column rows to print."),
    save: Path | None = typer.Option(None, "--save", help="Write the profile JSON here."),
) -> None:
    """Profile a dataset and print the measured facts. No model calls, no cost."""
    try:
        from .ingestion.router import load_source
        from .profiling.profiler import profile_dataframe
    except ImportError as exc:
        _fail(f"the profiling pipeline is not available in this install: {exc}")
        return

    data_source = _resolve_source(source, query)
    with console.status(f"loading {escape(data_source.uri or data_source.kind.value)}…"):
        frame, ingestion = load_source(data_source, max_rows=max_rows)
    with console.status("profiling…"):
        result = profile_dataframe(frame, target=target, settings=_settings())

    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="dim", justify="right")
    overview.add_column()
    overview.add_row("source", escape(data_source.uri or data_source.kind.value))
    overview.add_row("rows", f"{result.n_rows:,}")
    overview.add_row("columns", f"{result.n_columns:,}")
    overview.add_row(
        "missing cells", f"{result.total_missing_cells:,} ({result.missing_cell_fraction:.2%})"
    )
    overview.add_row(
        "duplicate rows", f"{result.n_duplicate_rows:,} ({result.duplicate_fraction:.2%})"
    )
    overview.add_row("memory", f"{result.memory_bytes / 1_048_576:.1f} MB")
    overview.add_row("profiled in", f"{result.profile_seconds:.2f}s")
    if ingestion.truncated:
        overview.add_row("note", "[yellow]row limit truncated the load[/]")
    console.print(Panel(overview, title="[bold]dataset[/]", border_style="cyan"))

    table = Table(header_style="dim", box=None, padding=(0, 1))
    table.add_column("column", overflow="fold")
    table.add_column("kind")
    table.add_column("dtype", style="dim")
    table.add_column("missing", justify="right")
    table.add_column("unique", justify="right")
    table.add_column("range / top", overflow="fold", style="dim")
    table.add_column("flags", style="yellow")
    for column in result.columns[:columns]:
        if column.minimum is not None or column.mean is not None:
            detail = (
                f"min={_g(column.minimum)} mean={_g(column.mean)} max={_g(column.maximum)}"
                + (f" skew={_g(column.skewness)}" if column.skewness is not None else "")
            )
        elif column.top_values:
            detail = ", ".join(f"{v.value}:{v.count}" for v in column.top_values[:4])
        elif column.min_timestamp:
            detail = f"{column.min_timestamp} .. {column.max_timestamp}"
        else:
            detail = ""
        flags = [
            name
            for name, active in (
                ("const", column.is_constant),
                ("id", column.looks_like_id),
                ("text", column.looks_like_text),
                ("geo", column.looks_like_geo),
                ("nzv", column.is_near_zero_variance),
            )
            if active
        ]
        if column.detected_semantic_type:
            flags.append(column.detected_semantic_type)
        table.add_row(
            escape(column.name),
            column.kind.value,
            column.dtype,
            f"{column.missing_fraction:.1%}",
            f"{column.n_unique:,}",
            escape(detail),
            ",".join(flags),
        )
    console.print(table)
    if result.n_columns > columns:
        console.print(f"[dim]… {result.n_columns - columns} more columns (use --columns)[/]")

    if result.target:
        summary = result.target
        panel = Table.grid(padding=(0, 2))
        panel.add_column(style="dim", justify="right")
        panel.add_column()
        panel.add_row("name", summary.name)
        panel.add_row("kind", summary.kind.value)
        if summary.n_classes is not None:
            panel.add_row("classes", str(summary.n_classes))
        if summary.class_counts:
            panel.add_row(
                "distribution",
                escape(
                    ", ".join(
                        f"{c.value}: {c.count:,} ({c.fraction:.1%})"
                        for c in summary.class_counts[:10]
                    )
                ),
            )
        if summary.imbalance_ratio is not None:
            panel.add_row(
                "imbalance",
                f"{summary.imbalance_ratio:.2f}"
                + (" [yellow]imbalanced[/]" if summary.is_imbalanced else ""),
            )
        if summary.mean is not None:
            panel.add_row("mean/std", f"{_g(summary.mean)} / {_g(summary.std)}")
        console.print(Panel(panel, title="[bold]target[/]", border_style="green"))

    if result.target_correlations:
        table = Table(title="strongest target correlations", box=None, header_style="dim")
        table.add_column("pair")
        table.add_column("coefficient", justify="right")
        table.add_column("method", style="dim")
        for pair in result.target_correlations[:10]:
            table.add_row(
                escape(f"{pair.left} ~ {pair.right}"), f"{pair.coefficient:+.4f}", pair.method
            )
        console.print(table)

    if result.leakage_findings:
        console.print("\n[bold red]leakage candidates[/]")
        for finding in result.leakage_findings:
            console.print(
                f"  [red]![/] {escape(finding.column)} score={finding.score:.3f} "
                f"{_severity_text(finding.severity)} {escape(finding.reason)}"
            )

    if result.quality_issues:
        console.print("\n[bold yellow]data-quality issues[/]")
        for issue in result.quality_issues[:15]:
            console.print(
                f"  {_severity_text(issue.severity)} {issue.code}: {escape(issue.detail)}"
                + (f" [dim]{escape(str(issue.columns[:6]))}[/]" if issue.columns else "")
            )

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
        )
        console.print(f"\n[green]wrote[/] {escape(str(save))}")


def _g(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _severity_text(severity: Severity) -> str:
    """Colour a severity label.

    Rendered as a coloured word rather than ``[medium]``: rich would read the
    square brackets as a style tag and swallow the text.
    """
    style = {
        Severity.CRITICAL: "bold red",
        Severity.HIGH: "red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "dim",
        Severity.INFO: "dim",
    }.get(severity, "white")
    return f"[{style}]{severity.value:<8}[/]"


# ---------------------------------------------------------------------------
# amla list / show / events
# ---------------------------------------------------------------------------


@app.command("list")
def list_runs(
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by project."),
    limit: int = typer.Option(20, "--limit", "-n", help="How many runs to show."),
) -> None:
    """List stored runs, most recent first."""
    rows = _repository().list_run_index(project, limit)
    if not rows:
        console.print("[dim]no runs recorded yet. Start one with 'amla run <source>'.[/]")
        return
    active = set(_manager().active_ids())
    # A narrow terminal drops the columns you can get from `amla show`, rather
    # than truncating the run id — which is the one value you have to copy.
    wide = console.width >= 104
    show_project = wide and not project

    table = Table(header_style="dim", box=None, padding=(0, 1))
    table.add_column("run", no_wrap=True, min_width=16)
    if show_project:
        table.add_column("project", max_width=14, no_wrap=True, overflow="ellipsis")
    table.add_column("status", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("model", max_width=18, overflow="ellipsis")
    table.add_column("grade", justify="center", width=5)
    table.add_column("took", justify="right", style="dim", no_wrap=True)
    if wide:
        table.add_column("started", style="dim", no_wrap=True)

    for row in rows:
        status = _status_text(_safe_status(row["status"]))
        if row["run_id"] in active:
            status = Text("running", style="bold cyan")
        score = (
            f"{row['primary_metric'] or 'score'} {row['best_score']:.4g}"
            if row["best_score"] is not None
            else "-"
        )
        cells: list[Any] = [row["run_id"]]
        if show_project:
            cells.append(row["project"] or "-")
        cells.extend(
            [
                status,
                score,
                row["best_family"] or "-",
                row["grade"] or "-",
                _duration(row["duration_seconds"] or 0.0),
            ]
        )
        if wide:
            cells.append(_short_stamp(row["started_at"]))
        table.add_row(*cells)
    console.print(table)


def _match_step_record(step: Any, summary: RunSummary) -> Any | None:
    """Find the execution record for a planned step.

    The orchestrator reconciles agent-authored step ids against its canonical
    graph (``clean_data`` becomes ``clean``), so an id lookup alone would report
    every reconciled step as never having run. Title and owning agent are the
    surviving links.
    """
    for record in summary.steps:
        if record.step_id == step.step_id:
            return record
    for record in summary.steps:
        if record.title and record.title == step.title:
            return record
    return next(
        (record for record in summary.steps if record.agent is step.agent), None
    )


def _safe_status(value: str | None) -> RunStatus | None:
    try:
        return RunStatus(value) if value else None
    except ValueError:
        return None


@app.command()
def show(
    run_id: str = typer.Argument(..., help="Run id, from 'amla list'."),
    json_out: bool = typer.Option(False, "--json", help="Dump the full summary as JSON."),
) -> None:
    """Show everything recorded about one run."""
    summary = _load_summary(run_id)
    if json_out:
        console.print_json(json.dumps(summary.model_dump(mode="json"), default=str))
        return

    _print_run_summary(summary, [])

    if summary.understanding:
        console.print(
            Panel(
                escape(summary.understanding.narrative[:2500]),
                title=f"[bold]{escape(summary.understanding.headline)}[/]",
                border_style="cyan",
            )
        )
    if summary.plan:
        table = Table(title="plan", box=None, header_style="dim", padding=(0, 1))
        table.add_column("#", justify="right")
        table.add_column("step")
        table.add_column("agent", style="magenta")
        table.add_column("status")
        table.add_column("took", justify="right", style="dim")
        for step in summary.plan.ordered():
            record = _match_step_record(step, summary)
            table.add_row(
                str(step.order),
                escape(step.title),
                step.agent.value,
                record.status.value if record else "-",
                _duration(record.duration_seconds) if record else "-",
            )
        console.print(table)
    if summary.cleaning:
        console.print("\n[bold]cleaning decisions[/]")
        for decision in summary.cleaning.decisions[:20]:
            console.print(
                f"  [cyan]{decision.action.value}[/] "
                f"{escape(str(decision.columns) if decision.columns else 'table')}"
                f" — {escape(decision.rationale[:200])}"
            )
    if summary.features:
        console.print("\n[bold]feature engineering[/]")
        for decision in summary.features.decisions[:20]:
            console.print(
                f"  [cyan]{decision.op.value}[/] {escape(str(decision.input_columns))}"
                f" — {escape(decision.rationale[:200])}"
            )
    if summary.approvals:
        console.print("\n[bold]approvals[/]")
        for approval in summary.approvals:
            console.print(
                f"  [{'green' if approval.decision == 'approved' else 'yellow'}]"
                f"{approval.decision}[/] {escape(approval.action_summary)}"
                + (f" [dim]({escape(approval.note)})[/]" if approval.note else "")
            )


@app.command()
def events(
    run_id: str = typer.Argument(..., help="Run id."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new events as they arrive."),
    after: int = typer.Option(0, "--after", help="Start after this sequence number."),
    kind: str | None = typer.Option(None, "--kind", help="Filter by event kind."),
) -> None:
    """Print a run's event log, optionally following it live."""
    repository = _repository()
    manager = _manager()
    if manager.summary(run_id) is None:
        _fail(f"unknown run {run_id!r}")
    cursor = after
    try:
        while True:
            batch = repository.get_events(run_id, cursor)
            handle = manager.get(run_id)
            if handle is not None:
                seen = {event.sequence for event in batch}
                batch = sorted(
                    batch + [e for e in handle.bus.since(cursor) if e.sequence not in seen],
                    key=lambda event: event.sequence,
                )
            for event in batch:
                cursor = max(cursor, event.sequence)
                if kind and event.kind.value != kind:
                    continue
                console.print(_format_event(event))
            status = manager.status(run_id)
            finished = status is None or status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }
            if not follow or finished:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("[dim]stopped following[/]")


def _format_event(event: RunEvent) -> str:
    colour = {
        EventKind.WARNING: "yellow",
        EventKind.STEP_FAILED: "red",
        EventKind.RUN_FAILED: "red",
        EventKind.AGENT_DECISION: "magenta",
        EventKind.STEP_COMPLETED: "green",
        EventKind.RUN_COMPLETED: "green",
        EventKind.METRIC_RECORDED: "cyan",
    }.get(event.kind, "white")
    extras = []
    if event.duration_seconds:
        extras.append(f"{event.duration_seconds:.2f}s")
    if event.cost_usd:
        extras.append(f"${event.cost_usd:.4f}")
    suffix = f" [dim]({', '.join(extras)})[/]" if extras else ""
    agent = f"[magenta]{event.agent.value}[/] " if event.agent else ""
    return (
        f"[dim]{event.sequence:>4}[/] [dim]{event.at.strftime('%H:%M:%S')}[/] "
        f"[{colour}]{event.kind.value:<19}[/] {agent}{escape(event.message)}{suffix}"
    )


# ---------------------------------------------------------------------------
# amla ask / report
# ---------------------------------------------------------------------------


@app.command()
def ask(
    run_id: str = typer.Argument(..., help="Run id."),
    question: str = typer.Argument(..., help="Your question, in plain English."),
) -> None:
    """Ask a question about a run. Answers are grounded in its recorded evidence."""
    from .api.service import answer_question

    summary = _load_summary(run_id)
    try:
        with console.status("reading the run record…"):
            answer = answer_question(summary, question)
    except Exception as exc:
        _fail(f"could not answer: {type(exc).__name__}: {exc}")
        return

    console.print(
        Panel(escape(answer.answer), title=f"[bold]{escape(question)}[/]", border_style="cyan")
    )
    if answer.evidence:
        console.print("[bold]evidence[/]")
        for item in answer.evidence:
            console.print(f"  [green]-[/] {escape(item)}")
    if answer.caveats:
        console.print("[bold yellow]caveats[/]")
        for item in answer.caveats:
            console.print(f"  [yellow]![/] {escape(item)}")
    if answer.suggested_followups:
        console.print("[dim]follow-ups:[/]")
        for item in answer.suggested_followups:
            console.print(f"  [dim]? {escape(item)}[/]")
    console.print(f"[dim]confidence: {answer.confidence}[/]")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run id."),
    format: str = typer.Option("html", "--format", help="md, html, pdf, pptx, or json."),
    open_it: bool = typer.Option(False, "--open", help="Open it instead of printing the path."),
) -> None:
    """Locate a rendered report for a run."""
    from .api.service import available_report_formats, locate_report

    summary = _load_summary(run_id)
    key = _parse_formats(format)[0]
    path = locate_report(run_id, summary, key)
    if path is None:
        available = available_report_formats(run_id, summary)
        _fail(
            f"no {key} report for {run_id}"
            + (f". Available: {', '.join(available)}" if available else ". None were rendered.")
        )
        return
    if open_it:
        typer.launch(str(path))
    console.print(str(path))


# ---------------------------------------------------------------------------
# amla serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind address."),
    port: int | None = typer.Option(None, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
    log_level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """Serve the HTTP API and the event stream."""
    try:
        import uvicorn
    except ImportError:
        _fail('the API extra is not installed. Run: pip install "automl-architect[api]"')
        return

    settings = _settings()
    settings.ensure_dirs()
    bind_host = host or settings.api_host
    bind_port = port or settings.api_port
    console.print(
        f"[bold cyan]AutoML Architect[/] api on [link]http://{bind_host}:{bind_port}[/] "
        f"[dim](docs at /docs, health at /api/health)[/]"
    )
    uvicorn.run(
        "automl_architect.api.app:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level=(log_level or settings.log_level).lower(),
    )


# ---------------------------------------------------------------------------
# amla doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show every check."),
) -> None:
    """Diagnose the install: interpreter, credentials, storage, optional features."""
    from .api.service import detect_features, module_available

    settings = _settings()
    problems: list[str] = []

    table = Table(header_style="dim", box=None, padding=(0, 1))
    table.add_column("", width=2)
    table.add_column("check")
    table.add_column("result", overflow="fold")

    def add(ok: bool, name: str, detail: str, *, fatal: bool = False) -> None:
        mark = "[green]v[/]" if ok else ("[red]x[/]" if fatal else "[yellow]-[/]")
        table.add_row(mark, name, detail)
        if not ok and fatal:
            problems.append(name)

    version_ok = sys.version_info >= (3, 11)
    add(
        version_ok,
        "python",
        f"{sys.version.split()[0]} at {sys.executable}"
        + ("" if version_ok else "  [red]3.11+ required[/]"),
        fatal=True,
    )
    add(True, "automl-architect", __version__)

    credentials = settings.has_api_key()
    # Offline mode makes credentials irrelevant rather than merely optional, so a
    # missing key must not read as a broken install there.
    add(
        credentials or settings.offline,
        "credentials",
        "not required — offline mode is enabled"
        if settings.offline
        else "ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN found"
        if credentials
        else "no API key in the environment. Set ANTHROPIC_API_KEY, sign in with "
        "'ant auth login' if you use an OAuth profile, or run with --offline to "
        "use the deterministic rule engine instead",
        fatal=True,
    )
    add(
        True,
        "mode",
        "[yellow]offline[/] — deterministic rule engine, no model calls"
        if settings.offline
        else f"online — {settings.model}  [dim]effort={settings.default_effort}[/]",
    )

    writable = _probe_writable(settings)
    add(
        writable,
        "workspace",
        f"{settings.workspace.resolve()}"
        + ("" if writable else "  [red]not writable[/]"),
        fatal=True,
    )

    reachable, database_detail = _probe_database(settings)
    add(reachable, "database", database_detail, fatal=True)

    for name, module in _INTERNAL_MODULES:
        present = module_available(module)
        add(
            present,
            f"module: {name}",
            module if present else f"{module} is missing from this install",
            fatal=name in ("storage", "api"),
        )

    console.print(Panel(table, title="[bold]environment[/]", border_style="cyan"))

    features = Table(header_style="dim", box=None, padding=(0, 1))
    features.add_column("", width=2)
    features.add_column("optional feature")
    features.add_column("what it enables", overflow="fold", style="dim")
    features.add_column("fix", overflow="fold")
    missing_count = 0
    for feature in detect_features():
        if feature.available and not verbose:
            features.add_row("[green]v[/]", feature.name, feature.detail, "")
            continue
        if not feature.available:
            missing_count += 1
        features.add_row(
            "[green]v[/]" if feature.available else "[yellow]-[/]",
            feature.name,
            feature.detail,
            "" if feature.available else f"[cyan]{escape(feature.install_hint)}[/]",
        )
    console.print(Panel(features, title="[bold]optional features[/]", border_style="cyan"))

    repository_note = _probe_history(settings)
    if repository_note:
        console.print(f"[dim]{escape(repository_note)}[/]")

    if problems:
        console.print(
            f"\n[bold red]{len(problems)} blocking problem(s):[/] {', '.join(problems)}"
        )
        raise typer.Exit(1)
    if missing_count:
        console.print(
            f"\n[green]core install is healthy.[/] "
            f"[dim]{missing_count} optional feature(s) unavailable — see the fix column.[/]"
        )
    else:
        console.print("\n[bold green]everything checks out.[/]")


def _probe_writable(settings: Settings) -> bool:
    try:
        settings.ensure_dirs()
        probe = settings.workspace / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _probe_database(settings: Settings) -> tuple[bool, str]:
    """Try to reach the database, reporting the reason if that is impossible.

    Every step is guarded, including reading the URL: the default URL is derived
    by creating the workspace directory, so an unwritable workspace makes even
    the URL raise — and ``doctor`` must diagnose a broken install, not crash on it.
    """
    try:
        url = settings.resolved_database_url
    except Exception as exc:
        return False, f"could not resolve a database URL: {exc}"
    redacted = url if "@" not in url else url.split("@")[0].split("://")[0] + "://***@…"
    try:
        repository = _repository()
        if not repository.ping():
            return False, f"{escape(redacted)} did not accept a connection"
        return True, (
            f"{escape(redacted)} [dim]({repository.dialect}, "
            f"{repository.count_runs()} runs)[/]"
        )
    except Exception as exc:
        return False, f"{escape(redacted)}: {exc}"


def _probe_history(settings: Settings) -> str:
    try:
        repository = _repository()
        runs = repository.count_runs()
        fingerprints = len(repository.list_fingerprints(limit=1000))
    except Exception:
        return ""
    return (
        f"history: {runs} run(s) stored, {fingerprints} dataset fingerprint(s) available "
        "to the planner's memory"
    )


# ---------------------------------------------------------------------------
# root callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"automl-architect {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show the version."
    ),
) -> None:
    """AutoML Architect: profile, plan, model, explain — with every choice recorded."""


if __name__ == "__main__":  # pragma: no cover
    app()
