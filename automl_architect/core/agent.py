"""Agent base class.

Each agent is a narrow contract: a system prompt describing its role, a prompt
builder that hands the model *measured* facts, and a Pydantic output type. The
base class owns everything cross-cutting — event emission, usage accounting,
prompt-cache wiring, and grounding checks.

The grounding checks matter more than they look. The single most damaging
failure mode in a system like this is an agent confidently naming a column that
does not exist, which then explodes three steps later inside pandas.
:meth:`BaseAgent.keep_known_columns` filters agent-supplied column references
against the real schema and records what it dropped, so a hallucination becomes
a logged warning instead of a stack trace.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from .errors import AgentError
from .events import EventKind
from .llm import Effort, LLMClient, PromptBlock, StructuredResult, build_system, get_llm_client
from .schemas import AgentName
from .state import RunState

logger = logging.getLogger(__name__)

TOut = TypeVar("TOut", bound=BaseModel)


class BaseAgent(ABC, Generic[TOut]):
    """One reasoning specialist."""

    #: Identity used in events, reports, and the step graph.
    name: AgentName
    #: Human-readable label for UI and logs.
    title: str = ""
    #: Schema the model must return.
    output_model: type[TOut]
    #: Reasoning depth. Planning and evaluation warrant more than narration.
    effort: Effort = "high"
    #: Output budget. Covers thinking *and* visible tokens on Opus 5.
    max_tokens: int = 16_000

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_llm_client()
        if not self.title:
            self.title = self.name.value.replace("_", " ").title() + " Agent"

    # -- subclass contract -------------------------------------------------

    @abstractmethod
    def instructions(self, state: RunState) -> str:
        """The agent-specific system prompt: role, method, and output discipline."""

    @abstractmethod
    def build_prompt(self, state: RunState) -> str:
        """The user turn: measured facts and the concrete question to answer."""

    def postprocess(self, value: TOut, state: RunState) -> TOut:
        """Ground the model's output against reality. Override to repair or reject."""
        return value

    def apply(self, state: RunState, value: TOut) -> None:
        """Write the result into the blackboard. Override in every subclass."""
        raise NotImplementedError

    def decision_summary(self, value: TOut) -> str:
        """One line for the event stream and the UI."""
        return f"{self.title} produced {type(value).__name__}"

    # -- grounding helpers -------------------------------------------------

    def known_columns(self, state: RunState) -> set[str]:
        frame = state.df
        if frame is not None:
            return set(map(str, frame.columns))
        if state.profile:
            return {c.name for c in state.profile.columns}
        return set()

    def keep_known_columns(
        self,
        columns: list[str],
        state: RunState,
        *,
        context: str = "",
    ) -> list[str]:
        """Drop column references that do not exist, and say so loudly."""
        known = self.known_columns(state)
        if not known:
            return columns
        kept = [c for c in columns if c in known]
        unknown = [c for c in columns if c not in known]
        if unknown:
            state.add_warning(
                f"{self.title} referenced {len(unknown)} unknown column(s) "
                f"{unknown[:8]}{'...' if len(unknown) > 8 else ''}"
                f"{f' in {context}' if context else ''}; ignoring them."
            )
        return kept

    # -- prompt assembly ---------------------------------------------------

    def system_blocks(self, state: RunState) -> list[PromptBlock]:
        return build_system(
            run_context=state.run_context,
            agent_instructions=self.instructions(state),
        )

    # -- execution ---------------------------------------------------------

    def call(self, state: RunState) -> StructuredResult:
        """Invoke the model once, with events and accounting. No state writes."""
        prompt = self.build_prompt(state)
        system = self.system_blocks(state)
        started = time.perf_counter()

        state.bus.emit(
            EventKind.LLM_CALL,
            f"{self.title} reasoning (effort={self.effort})",
            agent=self.name,
        )

        result = self.llm.structured(
            output_model=self.output_model,
            user=prompt,
            system=system,
            effort=self.effort,
            max_tokens=self.max_tokens,
        )

        usage = result.usage
        state.usage.input_tokens += usage.input_tokens
        state.usage.output_tokens += usage.output_tokens
        state.usage.cache_read_tokens += usage.cache_read_tokens
        state.usage.cache_write_tokens += usage.cache_write_tokens
        state.usage.llm_calls += usage.calls
        state.usage.cost_usd += usage.cost_usd

        if result.thinking:
            state.bus.thinking(self.name, result.thinking)

        state.bus.emit(
            EventKind.LLM_CALL,
            f"{self.title} responded",
            agent=self.name,
            duration_seconds=time.perf_counter() - started,
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cost_usd=usage.cost_usd,
        )
        if result.fallback_used:
            state.add_warning(
                f"{self.title}: the request was declined by the primary model and "
                f"re-served by a fallback ({result.model})."
            )
        return result

    def offline_call(self, state: RunState) -> TOut:
        """Produce this agent's decision from measured state, with no model call.

        The rule engine in :mod:`automl_architect.core.offline` derives the same
        output type from the profile and whatever the execution layer has already
        computed. The result still flows through :meth:`postprocess` and
        :meth:`apply`, so every grounding check that guards a model's output
        guards this one too.
        """
        from . import offline

        started = time.perf_counter()
        state.bus.emit(
            EventKind.LOG,
            f"{self.title} deciding offline (deterministic rule engine, no model call)",
            agent=self.name,
        )
        value: TOut = offline.decide(self.output_model, state, self)  # type: ignore[assignment]
        state.bus.emit(
            EventKind.LOG,
            f"{self.title} decided offline",
            agent=self.name,
            duration_seconds=time.perf_counter() - started,
        )
        return value

    def run(self, state: RunState) -> TOut:
        """Full cycle: reason, ground, record, apply."""
        offline_mode = bool(getattr(state.settings, "offline", False))
        try:
            value: TOut = (
                self.offline_call(state) if offline_mode else self.call(state).value
            )
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised with agent context
            raise AgentError(self.name.value, str(exc)) from exc

        value = self.postprocess(value, state)
        self.apply(state, value)

        state.bus.decision(self.name, self.decision_summary(value))
        return value


class HybridAgent(BaseAgent[TOut], ABC):
    """An agent whose facts come from an executor before it reasons.

    The Experiment and Explainability agents are the canonical cases: real
    training and real SHAP values are computed deterministically, then the agent
    interprets them. Subclasses implement :meth:`compute` and can read the
    result from ``state`` inside :meth:`build_prompt`.
    """

    @abstractmethod
    def compute(self, state: RunState) -> None:
        """Run the deterministic work whose results this agent will interpret."""

    def run(self, state: RunState) -> TOut:
        self.compute(state)
        return super().run(state)
