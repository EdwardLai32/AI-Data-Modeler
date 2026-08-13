"""Inspect a finished run and check the claims the product makes about itself.

Usage:  .venv/Scripts/python.exe scripts/inspect_run.py [run_id]

This is a verification tool, not a demo. It answers questions that a screenshot
of a pretty report cannot:

  * Did the pipeline actually reach a terminal state, or did it degrade quietly?
  * Does every cleaning and feature decision carry a real rationale, or are some
    empty strings that satisfy the schema without satisfying the requirement?
  * Did the agents find the leakage column that was deliberately planted in the
    churn fixture, and did they act on it?
  * Did a model beat the trivial baseline? A high absolute score means nothing
    on an imbalanced target if the majority-class guess scores the same.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path("workspace/runs")
# Planted in examples/generate_datasets.py as a known-answer leakage test.
PLANTED_LEAK = "cancellation_tickets"


def latest_run() -> Path | None:
    candidates = [p for p in WORKSPACE.glob("run_*") if (p / "run_summary.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "run_summary.json").stat().st_mtime)


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def short(text: str | None, width: int = 300) -> str:
    if not text:
        return "(empty)"
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def main() -> int:
    run_dir: Path | None
    if len(sys.argv) > 1:
        run_dir = WORKSPACE / sys.argv[1]
    else:
        run_dir = latest_run()

    if run_dir is None or not (run_dir / "run_summary.json").exists():
        print("no completed run found under workspace/runs")
        return 1

    data = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    problems: list[str] = []

    rule("RUN")
    print(f"  run_id      : {data['run_id']}")
    print(f"  status      : {data['status']}")
    print(f"  duration    : {data.get('duration_seconds', 0):.1f}s")
    print(f"  replans     : {data.get('replans', 0)}")
    print(f"  error       : {data.get('error') or 'none'}")
    usage = data.get("usage") or {}
    print(
        f"  llm         : {usage.get('llm_calls', 0)} calls · "
        f"in {usage.get('input_tokens', 0):,} · out {usage.get('output_tokens', 0):,} · "
        f"cache_read {usage.get('cache_read_tokens', 0):,} · "
        f"${usage.get('cost_usd', 0):.4f}"
    )
    if data["status"] != "completed":
        problems.append(f"run status is {data['status']}, not completed")

    rule("STEPS")
    for step in data.get("steps", []):
        flag = {"completed": "ok  ", "skipped": "skip", "failed": "FAIL"}.get(
            step.get("status", ""), "?   "
        )
        print(
            f"  {flag}  {step['step_id']:<22} {step.get('duration_seconds', 0):7.1f}s  "
            f"{short(step.get('summary'), 90)}"
        )
        if step.get("status") == "failed":
            problems.append(f"step {step['step_id']} failed: {step.get('error')}")

    rule("PROBLEM FRAMING")
    problem = data.get("problem") or {}
    print(f"  task        : {problem.get('task_type')}  ({problem.get('confidence')})")
    print(f"  target      : {problem.get('target_column')}")
    print(f"  metric      : {problem.get('primary_metric')}")
    print(f"  rationale   : {short(problem.get('rationale'))}")

    rule("LEAKAGE — did it find the planted column?")
    profile = data.get("profile") or {}
    findings = profile.get("leakage_findings") or []
    detected = {f["column"] for f in findings}
    for finding in findings:
        print(
            f"  {finding['severity']:<8} {finding['column']:<26} "
            f"score={finding.get('score', 0):.4f}  {short(finding.get('reason'), 110)}"
        )
    if PLANTED_LEAK in detected:
        print(f"\n  DETECTED: '{PLANTED_LEAK}' was flagged by the profiler.")
    else:
        print(f"\n  MISS: '{PLANTED_LEAK}' was NOT flagged.")
        problems.append(f"planted leakage column '{PLANTED_LEAK}' not detected")

    dropped = set(data.get("cleaning", {}).get("columns_to_drop") or [])
    acted = PLANTED_LEAK in dropped
    print(f"  ACTED ON: {'yes — dropped by the cleaning plan' if acted else 'no'}")
    if PLANTED_LEAK in detected and not acted:
        problems.append(f"'{PLANTED_LEAK}' detected but not dropped")

    rule("CLEANING DECISIONS (rationale required)")
    cleaning = data.get("cleaning") or {}
    decisions = cleaning.get("decisions") or []
    missing_rationale = 0
    for dec in decisions:
        rationale = dec.get("rationale") or ""
        if len(rationale.strip()) < 25:
            missing_rationale += 1
        print(f"  · {dec.get('action')} {dec.get('columns') or '(table)'}")
        print(f"      -> {short(rationale, 220)}")
    print(f"\n  {len(decisions)} decisions, {missing_rationale} with a thin rationale")
    if missing_rationale:
        problems.append(f"{missing_rationale} cleaning decisions lack a real rationale")

    rule("FEATURE DECISIONS")
    features = data.get("features") or {}
    for dec in (features.get("decisions") or [])[:10]:
        print(f"  · {dec.get('op')} {dec.get('input_columns') or ''}")
        print(f"      -> {short(dec.get('rationale'), 200)}")

    rule("LEADERBOARD — did anything beat the baseline?")
    experiments = data.get("experiments") or {}
    results = experiments.get("results") or []
    metric = experiments.get("primary_metric", "")
    higher_better = experiments.get("higher_is_better", True)
    best_id = experiments.get("best_experiment_id")
    baseline_score = None
    winner_score = None
    for res in sorted(
        results,
        key=lambda r: (r.get("primary_score") if r.get("primary_score") is not None else -1e18),
        reverse=higher_better,
    ):
        mark = " <- best" if res.get("experiment_id") == best_id else ""
        base = " [baseline]" if res.get("is_baseline") else ""
        score = res.get("primary_score")
        status = "FAILED" if res.get("failed") else f"{score:.4f}" if score is not None else "n/a"
        print(
            f"  {res.get('family'):<24} {metric}={status:<10} "
            f"train={res.get('train_seconds', 0):6.2f}s{base}{mark}"
        )
        if res.get("is_baseline") and score is not None:
            baseline_score = score
        if res.get("experiment_id") == best_id:
            winner_score = score

    if baseline_score is not None and winner_score is not None:
        delta = winner_score - baseline_score
        beat = delta > 0 if higher_better else delta < 0
        print(
            f"\n  winner {winner_score:.4f} vs baseline {baseline_score:.4f} "
            f"-> {'BEATS baseline' if beat else 'DOES NOT beat baseline'}"
        )
        if not beat:
            problems.append("winning model does not beat the trivial baseline")
    elif baseline_score is None:
        problems.append("no baseline model was trained — comparison impossible")

    rule("EVALUATION")
    evaluation = data.get("evaluation") or {}
    print(f"  acceptable  : {evaluation.get('acceptable')}")
    print(f"  grade       : {evaluation.get('overall_grade')}")
    print(f"  action      : {evaluation.get('recommended_action')}")
    bias = evaluation.get("bias_variance") or {}
    print(
        f"  fit         : {bias.get('verdict')} "
        f"(train={bias.get('train_score')}, test={bias.get('test_score')})"
    )
    print(f"  rationale   : {short(evaluation.get('verdict_rationale'), 400)}")

    rule("BUSINESS INSIGHTS")
    for ins in (data.get("insights") or {}).get("insights", [])[:4]:
        print(f"  · {short(ins.get('headline'), 160)}")
        print(f"      action  : {short(ins.get('recommended_action'), 160)}")
        print(f"      evidence: {short(ins.get('supporting_evidence'), 160)}")

    rule("ARTIFACTS")
    bundle = data.get("report_bundle") or {}
    for key, value in bundle.items():
        if isinstance(value, str) and value:
            exists = Path(value).exists()
            print(f"  {'ok  ' if exists else 'MISS'}  {key}: {value}")
            if not exists:
                problems.append(f"report bundle claims {key} at {value}, file absent")
    charts = list((run_dir / "charts").glob("*.html")) if (run_dir / "charts").exists() else []
    print(f"  charts rendered: {len(charts)}")

    rule("VERDICT")
    if problems:
        print(f"  {len(problems)} problem(s):")
        for item in problems:
            print(f"    - {item}")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
