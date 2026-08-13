"""Measure the structured-output schema each agent asks the API to compile.

A live run lost the Evaluation Agent to ``Grammar compilation timed out``. That
is a server-side budget on compiling the JSON schema into a sampling grammar, so
schema size is an operational property, not just a design one. This ranks the
agent output models so an outlier is visible before it costs a run.
"""

from __future__ import annotations

import json

from automl_architect.agents import get_agent
from automl_architect.core.schemas import AgentName


def describe(schema: dict) -> tuple[int, int, int]:
    """Return (serialised chars, number of $defs, max nesting depth)."""
    text = json.dumps(schema)
    defs = len(schema.get("$defs", {}))

    def depth(node: object, level: int = 0) -> int:
        if isinstance(node, dict):
            return max((depth(v, level + 1) for v in node.values()), default=level)
        if isinstance(node, list):
            return max((depth(v, level + 1) for v in node), default=level)
        return level

    return len(text), defs, depth(schema)


def main() -> int:
    rows: list[tuple[str, str, int, int, int]] = []
    for name in AgentName:
        try:
            agent_cls = get_agent(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name.value}: unavailable ({exc})")
            continue
        model = agent_cls.output_model
        chars, defs, nesting = describe(model.model_json_schema())
        rows.append((name.value, model.__name__, chars, defs, nesting))

    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"{'agent':<18} {'output model':<26} {'chars':>7} {'$defs':>6} {'depth':>6}")
    print("-" * 68)
    for agent, model_name, chars, defs, nesting in rows:
        flag = "  <-- largest" if chars == rows[0][2] else ""
        print(f"{agent:<18} {model_name:<26} {chars:>7,} {defs:>6} {nesting:>6}{flag}")

    print()
    print(
        "Compilation cost scales with size and nesting. The largest schema is the "
        "one most exposed to a compile timeout under load, which is why\n"
        "core/llm.py now retries that specific 400 instead of treating it as fatal."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
