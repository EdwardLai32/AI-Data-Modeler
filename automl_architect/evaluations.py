"""Repeatable provider comparisons with identical datasets, seeds and limits."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .core.schemas import RunStatus, TaskType
from .runner import AutoMLArchitect

CASES = (
    ("churn.csv", "churned"),
    ("house_prices.csv", "sale_price"),
    ("sales_timeseries.csv", "units_sold"),
)


def evaluate(
    *,
    provider="offline",
    model=None,
    live=False,
    rows=300,
    repetitions=1,
    examples=None,
    output=None,
    max_cost_usd=None,
    cases=None,
):
    if provider not in {"offline", "anthropic", "openai"}:
        raise ValueError("Provider must be offline, anthropic or openai.")
    if provider != "offline" and not live:
        raise ValueError("Live evaluation requires --live and uses your provider credits.")
    if rows < 30 or not 1 <= repetitions <= 10:
        raise ValueError("Use at least 30 rows and 1-10 repetitions.")
    settings = Settings(offline=provider == "offline")
    selected = model or settings.model_for_provider(None if provider == "offline" else provider)
    if provider != "offline":
        settings = settings.model_copy(update={"provider": provider})
    settings = settings.model_copy(update={"model": selected})
    root = Path(examples) if examples else Path(__file__).resolve().parents[1] / "examples"
    records = []
    for filename, target in cases or CASES:
        path = root / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for repetition in range(repetitions):
            architect = AutoMLArchitect(settings=settings)
            started = time.perf_counter()
            summary = architect.analyse(
                str(path),
                target=target,
                project="harness-evaluation",
                max_rows=rows,
                max_experiments=2,
                cv_folds=2,
                random_state=42 + repetition,
                time_budget_seconds=180,
                max_llm_calls=30,
                max_llm_tokens=500_000,
                max_cost_usd=max_cost_usd,
                task_type_override=TaskType.TIME_SERIES_FORECASTING
                if filename == "sales_timeseries.csv"
                else None,
                enable_tuning=False,
                enable_explainability=False,
                enable_visualizations=False,
                enable_self_improvement=False,
                report_formats=["json"],
            )
            events = architect.last_run.bus.events
            best = summary.experiments.best() if summary.experiments else None
            records.append(
                {
                    "case": filename,
                    "dataset_sha256": digest,
                    "seed": 42 + repetition,
                    "provider": provider,
                    "model": selected if provider != "offline" else "rules",
                    "run_id": summary.run_id,
                    "status": summary.status.value,
                    "error": summary.error,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "usage": summary.usage.model_dump(),
                    "validation_score": best.primary_score if best else None,
                    "final_test_metrics": [m.model_dump() for m in summary.final_test_metrics],
                    "step_retries": sum(e.kind.value == "step_retried" for e in events),
                    "step_failures": sum(e.kind.value == "step_failed" for e in events),
                    "warning_count": len(summary.warnings),
                    "checkpoint_saved": bool(summary.checkpoint),
                }
            )
    completed = sum(r["status"] == RunStatus.COMPLETED.value for r in records)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "synthetic_examples": True,
        "rows_cap": rows,
        "repetitions": repetitions,
        "completed": completed,
        "total": len(records),
        "completion_rate": completed / max(1, len(records)),
        "runs": records,
    }
    dest = (
        Path(output)
        if output
        else settings.workspace / "evaluations" / f"{provider}-{int(time.time())}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dest, report
