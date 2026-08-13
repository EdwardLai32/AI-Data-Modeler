"""Workflow orchestration: the step graph, its policies, and the engine.

The graph and the policies import nothing heavier than the core contracts, so
they are safe to import for inspection (the API's ``/graph`` view, tests, docs).
:class:`~automl_architect.orchestrator.engine.Orchestrator` is exposed lazily
because it pulls in the agent and executor layers on first use.
"""

from __future__ import annotations

from typing import Any

from .graph import (
    CANONICAL_STEPS,
    EVALUATION_STEP_ID,
    PLANNABLE_STEP_IDS,
    PROLOGUE_STEP_IDS,
    REQUIRED_STEP_IDS,
    UNSUPPORTED_INTENTS,
    PlanReconciliation,
    ResolvedStep,
    StepDefinition,
    canonical_steps,
    describe_graph,
    get_step,
    reconcile_plan,
    resolve_step_id,
    step_index,
    unsupported_intent,
)
from .policies import (
    ApprovalPolicy,
    BudgetPolicy,
    BudgetVerdict,
    ReplanDecision,
    ReplanPolicy,
    RetryPolicy,
)

__all__ = [
    "CANONICAL_STEPS",
    "EVALUATION_STEP_ID",
    "PLANNABLE_STEP_IDS",
    "PROLOGUE_STEP_IDS",
    "REQUIRED_STEP_IDS",
    "UNSUPPORTED_INTENTS",
    "ApprovalPolicy",
    "BudgetPolicy",
    "BudgetVerdict",
    "Orchestrator",
    "PlanReconciliation",
    "ReplanDecision",
    "ReplanPolicy",
    "ResolvedStep",
    "RetryPolicy",
    "StepDefinition",
    "canonical_steps",
    "describe_graph",
    "get_step",
    "reconcile_plan",
    "resolve_agent_class",
    "resolve_step_id",
    "state_from_summary",
    "step_index",
    "unsupported_intent",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy import shim
    if name in ("Orchestrator", "resolve_agent_class", "state_from_summary"):
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
