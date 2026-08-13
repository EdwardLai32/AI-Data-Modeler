"""Agent registry.

One import-time decision shapes this module: **nothing is imported eagerly.**
An agent module pulls in whatever it needs to reason about its own domain, and a
syntax error or a missing optional dependency in any one of thirteen siblings
would otherwise break ``import automl_architect.agents`` for all of them — and
with it the CLI, the API, and the orchestrator. So the registry stores dotted
locations, resolves them on first use, and caches the class.

:data:`AGENTS` still behaves like a ``dict[AgentName, type[BaseAgent]]`` for
callers (subscript, ``in``, ``get``, iteration), but resolution happens on access
rather than on import.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from ..core.errors import ConfigurationError
from ..core.schemas import AgentName

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the import lazy
    from ..core.agent import BaseAgent
    from ..core.llm import LLMClient

#: ``AgentName`` -> (module name inside this package, candidate class names).
#: Several class names are accepted per entry so a sibling module that spells its
#: class ``FeaturesAgent`` rather than ``FeatureAgent`` still resolves; failing
#: that, :func:`_resolve` scans the module for the class whose ``name`` matches.
_LOCATIONS: dict[AgentName, tuple[str, tuple[str, ...]]] = {
    AgentName.DATASET: ("dataset", ("DatasetAgent", "DatasetUnderstandingAgent")),
    AgentName.PROBLEM: ("problem", ("ProblemAgent", "ProblemIdentificationAgent")),
    AgentName.PLANNER: ("planner", ("PlannerAgent", "PlanningAgent")),
    AgentName.CLEANING: ("cleaning", ("CleaningAgent", "DataCleaningAgent")),
    AgentName.FEATURES: ("features", ("FeatureAgent", "FeaturesAgent", "FeatureEngineeringAgent")),
    AgentName.MODEL_SELECTION: (
        "model_selection",
        ("ModelSelectionAgent", "ModelSelectorAgent"),
    ),
    AgentName.EXPERIMENT: ("experiment", ("ExperimentAgent", "ExperimentationAgent")),
    AgentName.TUNING: ("tuning", ("TuningAgent", "HyperparameterAgent")),
    AgentName.EXPLAIN: ("explain", ("ExplainAgent", "ExplainabilityAgent")),
    AgentName.EVALUATION: ("evaluation", ("EvaluationAgent", "EvaluatorAgent")),
    AgentName.INSIGHT: ("insight", ("InsightAgent", "BusinessInsightAgent")),
    AgentName.VISUALIZATION: ("visualization", ("VisualizationAgent", "VisualisationAgent")),
    AgentName.REPORT: ("report", ("ReportAgent", "ReportingAgent")),
}

_CACHE: dict[AgentName, type[Any]] = {}


def _coerce_name(name: AgentName | str) -> AgentName:
    if isinstance(name, AgentName):
        return name
    try:
        return AgentName(str(name).strip().lower())
    except ValueError as exc:
        valid = ", ".join(a.value for a in AgentName)
        raise ConfigurationError(
            f"unknown agent '{name}'; expected one of: {valid}"
        ) from exc


def _resolve(name: AgentName) -> type[Any]:
    """Import an agent module and return its agent class, caching the result."""
    cached = _CACHE.get(name)
    if cached is not None:
        return cached

    module_name, candidates = _LOCATIONS[name]
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
    except Exception as exc:  # noqa: BLE001 - see below
        # Deliberately broader than ImportError. A sibling can fail to import for
        # reasons that are not ImportError at all — a NameError at module scope, a
        # SyntaxError, or a MissingDependencyError raised while building a
        # module-level constant — and the point of this registry is that none of
        # those may escape as anything other than ConfigurationError. Narrowing
        # this to ImportError would make ``AGENTS.get()`` raise and
        # ``available_agents()`` explode on exactly the broken sibling they exist
        # to tolerate.
        raise ConfigurationError(
            f"the '{name.value}' agent is not available: "
            f"could not import {__name__}.{module_name} "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    for candidate in candidates:
        found = getattr(module, candidate, None)
        if isinstance(found, type):
            _CACHE[name] = found
            return found

    # Fall back to identity: whichever exported class declares this agent name.
    for attribute in vars(module).values():
        if (
            isinstance(attribute, type)
            and getattr(attribute, "name", None) is name
            and getattr(attribute, "output_model", None) is not None
        ):
            _CACHE[name] = attribute
            return attribute

    raise ConfigurationError(
        f"module {__name__}.{module_name} defines no agent class for "
        f"'{name.value}' (looked for {list(candidates)})"
    )


class _AgentRegistry(Mapping[AgentName, type["BaseAgent[Any]"]]):
    """Read-only mapping of agent name to class, resolved on access."""

    def __getitem__(self, key: AgentName | str) -> type[BaseAgent[Any]]:
        return _resolve(_coerce_name(key))

    def __iter__(self) -> Iterator[AgentName]:
        return iter(_LOCATIONS)

    def __len__(self) -> int:
        return len(_LOCATIONS)

    def get(  # type: ignore[override]
        self, key: AgentName | str, default: Any = None
    ) -> type[BaseAgent[Any]] | Any:
        """Mapping-compatible lookup.

        ``Mapping.get`` only swallows ``KeyError``, and resolution failures here
        raise :class:`ConfigurationError`, so the override is what makes
        ``AGENTS.get(name)`` behave the way callers expect for an agent whose
        module is absent.
        """
        try:
            return self[key]
        except ConfigurationError:
            return default

    def __contains__(self, key: object) -> bool:
        if isinstance(key, AgentName):
            return key in _LOCATIONS
        if isinstance(key, str):
            try:
                return _coerce_name(key) in _LOCATIONS
            except ConfigurationError:
                return False
        return False

    def __repr__(self) -> str:
        return f"<AgentRegistry: {len(_LOCATIONS)} agents>"


#: Mapping of every :class:`AgentName` to its agent class.
AGENTS: Mapping[AgentName, type["BaseAgent[Any]"]] = _AgentRegistry()


def get_agent(name: AgentName | str) -> type["BaseAgent[Any]"]:
    """Return the agent class registered for ``name``.

    Args:
        name: An :class:`AgentName` member or its string value.

    Returns:
        The agent class, not an instance. Use :func:`create_agent` for an
        instance wired to an LLM client.

    Raises:
        ConfigurationError: If the name is unknown, or the module implementing it
            cannot be imported or exposes no agent class.
    """
    return _resolve(_coerce_name(name))


def create_agent(
    name: AgentName | str, llm: "LLMClient | None" = None
) -> "BaseAgent[Any]":
    """Instantiate the agent registered for ``name``.

    Args:
        name: An :class:`AgentName` member or its string value.
        llm: Optional client override. Defaults to the process-wide client, so
            prompt-cache prefixes and usage totals are shared across agents.

    Returns:
        A ready-to-run agent instance.

    Raises:
        ConfigurationError: If the agent cannot be resolved.
    """
    return get_agent(name)(llm=llm)


def available_agents() -> list[AgentName]:
    """The agent names whose implementing module imports successfully.

    Cheap enough to call from a health check, and useful during development when
    only part of the agent set exists.
    """
    ready: list[AgentName] = []
    for name in _LOCATIONS:
        try:
            _resolve(name)
        except ConfigurationError:
            continue
        ready.append(name)
    return ready


def __getattr__(attribute: str) -> Any:
    """Expose ``agents.DatasetAgent`` and friends without importing everything."""
    for name, (_, candidates) in _LOCATIONS.items():
        if attribute in candidates:
            return _resolve(name)
    raise AttributeError(f"module {__name__!r} has no attribute {attribute!r}")


__all__ = [
    "AGENTS",
    "AgentName",
    "available_agents",
    "create_agent",
    "get_agent",
]
