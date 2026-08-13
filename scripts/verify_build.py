"""Integration check: does the assembled platform actually import and line up?

Run:  .venv/Scripts/python.exe scripts/verify_build.py

Checks three things, in increasing strictness:
  1. every module imports;
  2. the executor/agent contract symbols named in the build brief exist with the
     right callable shape;
  3. the agent registry resolves each AgentName to a class with the required
     attributes.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import traceback

MODULES = [
    "automl_architect",
    "automl_architect.config",
    "automl_architect.core",
    "automl_architect.core.schemas",
    "automl_architect.core.llm",
    "automl_architect.core.agent",
    "automl_architect.core.state",
    "automl_architect.core.context",
    "automl_architect.profiling",
    "automl_architect.profiling.profiler",
    "automl_architect.ingestion",
    "automl_architect.ingestion.router",
    "automl_architect.ingestion.files",
    "automl_architect.ingestion.sql",
    "automl_architect.ingestion.cloud",
    "automl_architect.ingestion.rest",
    "automl_architect.execution",
    "automl_architect.execution.model_zoo",
    "automl_architect.execution.metrics",
    "automl_architect.execution.trainer",
    "automl_architect.execution.tuner",
    "automl_architect.execution.cleaning_ops",
    "automl_architect.execution.feature_ops",
    "automl_architect.execution.splitter",
    "automl_architect.execution.explainer",
    "automl_architect.execution.diagnostics",
    "automl_architect.agents",
    "automl_architect.orchestrator",
    "automl_architect.orchestrator.engine",
    "automl_architect.orchestrator.graph",
    "automl_architect.orchestrator.policies",
    "automl_architect.runner",
    "automl_architect.reporting",
    "automl_architect.reporting.charts",
    "automl_architect.reporting.writer",
    "automl_architect.storage",
    "automl_architect.storage.repository",
    "automl_architect.storage.memory",
    "automl_architect.api.app",
    "automl_architect.cli",
]

# module -> required callables, from the interface contract in the build brief.
CONTRACT = {
    "automl_architect.profiling.profiler": ["profile_dataframe"],
    "automl_architect.ingestion.router": ["load_source"],
    "automl_architect.execution.splitter": ["make_splits", "resolve_split_strategy"],
    "automl_architect.execution.cleaning_ops": ["apply_cleaning_plan"],
    "automl_architect.execution.feature_ops": ["apply_feature_plan"],
    "automl_architect.execution.model_zoo": [
        "available_families",
        "is_available",
        "build_estimator",
        "default_search_space",
        "supports_proba",
    ],
    "automl_architect.execution.metrics": [
        "primary_metric_for",
        "higher_is_better",
        "score_predictions",
    ],
    "automl_architect.execution.trainer": ["run_experiments", "fit_final_model"],
    "automl_architect.execution.tuner": ["run_tuning"],
    "automl_architect.execution.explainer": ["compute_explanations"],
    "automl_architect.execution.diagnostics": ["compute_diagnostics"],
    "automl_architect.reporting.charts": ["render_charts"],
    "automl_architect.reporting.writer": ["write_report"],
    "automl_architect.runner": ["analyse"],
}


def main() -> int:
    print("=" * 72)
    print("1. MODULE IMPORTS")
    print("=" * 72)
    failed: dict[str, str] = {}
    loaded: dict[str, object] = {}
    for name in MODULES:
        try:
            loaded[name] = importlib.import_module(name)
            print(f"  ok    {name}")
        except Exception as exc:  # noqa: BLE001 - this script reports, not raises
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL  {name}")
            print(f"        {type(exc).__name__}: {exc}")
            tb = traceback.format_exc().strip().splitlines()
            for line in tb[-4:-1]:
                print(f"        {line.strip()}")

    print()
    print("=" * 72)
    print("2. EXECUTOR CONTRACT")
    print("=" * 72)
    missing: list[str] = []
    for module_name, symbols in CONTRACT.items():
        module = loaded.get(module_name)
        if module is None:
            print(f"  skip  {module_name} (import failed)")
            continue
        for symbol in symbols:
            fn = getattr(module, symbol, None)
            if fn is None:
                missing.append(f"{module_name}.{symbol}")
                print(f"  MISS  {module_name}.{symbol}")
            elif not callable(fn):
                missing.append(f"{module_name}.{symbol} (not callable)")
                print(f"  BAD   {module_name}.{symbol} is not callable")
            else:
                params = ", ".join(list(inspect.signature(fn).parameters)[:4])
                print(f"  ok    {module_name}.{symbol}({params})")

    print()
    print("=" * 72)
    print("3. AGENT REGISTRY")
    print("=" * 72)
    agent_problems: list[str] = []
    try:
        from automl_architect.core.schemas import AgentName

        agents_mod = importlib.import_module("automl_architect.agents")
        getter = getattr(agents_mod, "get_agent", None)
        if getter is None:
            agent_problems.append("agents.get_agent missing")
            print("  MISS  automl_architect.agents.get_agent")
        else:
            for member in AgentName:
                try:
                    cls = getter(member)
                    has = all(
                        hasattr(cls, attr)
                        for attr in ("output_model", "instructions", "build_prompt")
                    )
                    effort = getattr(cls, "effort", "?")
                    tokens = getattr(cls, "max_tokens", "?")
                    flag = "ok  " if has else "BAD "
                    if not has:
                        agent_problems.append(f"{member.value}: missing attrs")
                    print(
                        f"  {flag}  {member.value:<16} -> {cls.__name__:<24} "
                        f"effort={effort} max_tokens={tokens}"
                    )
                except Exception as exc:  # noqa: BLE001
                    agent_problems.append(f"{member.value}: {exc}")
                    print(f"  FAIL  {member.value:<16} {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        agent_problems.append(str(exc))
        print(f"  FAIL  registry unavailable: {type(exc).__name__}: {exc}")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  imports failed        : {len(failed)} / {len(MODULES)}")
    print(f"  contract symbols miss : {len(missing)}")
    print(f"  agent registry issues : {len(agent_problems)}")
    if failed:
        print("\n  failing modules:")
        for name, err in failed.items():
            print(f"    - {name}: {err}")
    if missing:
        print("\n  missing symbols:")
        for item in missing:
            print(f"    - {item}")
    if agent_problems:
        print("\n  agent issues:")
        for item in agent_problems:
            print(f"    - {item}")

    return 1 if (failed or missing or agent_problems) else 0


if __name__ == "__main__":
    sys.exit(main())
