"""The single place this package talks to Claude.

Design notes worth knowing before editing:

*   **One call shape.** Everything goes through ``client.beta.messages.parse``,
    which accepts ``output_format`` (Pydantic model in, validated instance out),
    ``cache_control``, ``fallbacks``, and ``thinking`` on the same request. That
    lets structured output, prompt caching, and refusal fallbacks share a code
    path instead of three divergent ones.
*   **Prompt caching is a prefix match.** ``system`` blocks are ordered
    stable-first: a platform preamble identical across every agent and run,
    then a run-context digest identical across agents *within* a run, then the
    agent-specific instructions. Cache breakpoints sit on the first two, so the
    Nth agent in a run reads the first two blocks from cache. Putting anything
    volatile (a timestamp, a step counter) into blocks 0 or 1 silently destroys
    this â€” keep churn in the user turn.
*   **Adaptive thinking is on.** Opus 5 thinks by default; ``max_tokens`` caps
    thinking *plus* visible output, so the budgets here are sized for both.
    Disabling thinking is only legal at effort ``high`` or below, so
    :meth:`_thinking_config` downgrades effort rather than letting the API 400.
"""

from __future__ import annotations

import logging
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..config import FALLBACK_BETA, Settings, estimate_cost_usd, get_settings
from .errors import (
    LLMOfflineError,
    LLMOutputError,
    LLMRefusalError,
    LLMTransientError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Effort = Literal["low", "medium", "high", "xhigh", "max"]

_TRANSIENT = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

#: Substrings that mark a 400 as load-dependent rather than a malformed request.
#:
#: Structured output compiles the JSON schema into a sampling grammar server-side.
#: A large schema â€” ``EvaluationVerdict`` nests six submodels and several enum
#: lists â€” can exceed the compiler's budget under load and come back as
#: ``invalid_request_error``. That reads like a permanent client error and is
#: not: compilation is cached for ~24h once it succeeds, so a retry usually
#: sticks. Treating it as fatal cost this platform its quality gate on a live
#: run, since the Evaluation Agent is the agent with the largest schema.
_RETRYABLE_BAD_REQUEST = (
    "grammar compilation timed out",
    "grammar is too complex",
    "overloaded",
    "please try again",
)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PromptBlock:
    """One ``system`` block, plus whether it should carry a cache breakpoint.

    Set ``cache=True`` only on content that is byte-identical across many
    requests. Anthropic allows at most 4 breakpoints per request.
    """

    text: str
    cache: bool = False


PLATFORM_PREAMBLE = """\
You are one specialist agent inside AutoML Architect, an autonomous multi-agent \
data science platform. A deterministic orchestrator invokes you, hands you \
verified facts, and applies your decisions with real library code \
(pandas, scikit-learn, XGBoost, LightGBM, Optuna, SHAP).

How this system works, and what it means for you:

1.  You reason; you do not compute. Every statistic you are given was measured \
from the actual data by deterministic code. Never invent a number, and never \
contradict a measured value. If a figure you need is absent, say so and reason \
from what you do have.
2.  Every decision must carry its reasoning. Your output schema requires a \
rationale on each decision, and that rationale must cite the specific evidence \
that drove it â€” a skewness value, a cardinality count, a class ratio, a \
correlation. "Best practice" is not a rationale. "Skewness of 3.4 puts the mean \
far from the bulk of the distribution, so the median imputes more \
representatively" is.
3.  Your output is consumed by code, not read by a human first. It is parsed \
directly into typed objects and executed. Emit only what the schema asks for.
4.  Fit the decision to this dataset. A plan that would suit any dataset is a \
plan you have not thought about. Reference the actual columns, sizes, and \
distributions in front of you.
5.  Prefer the simplest choice that the evidence supports. Do not add \
transformations, features, or models whose value you cannot argue for. \
Complexity you cannot justify is complexity that will cost accuracy or \
explainability later.
6.  Be candid about uncertainty and risk. If the data is too small for a claim, \
if a feature risks leakage, if a metric will mislead on this class balance â€” \
say it plainly in the field provided. Downstream agents and a human reviewer \
both depend on you flagging it rather than smoothing it over.
"""


def build_system(
    *,
    run_context: str | None = None,
    agent_instructions: str,
    include_preamble: bool = True,
) -> list[PromptBlock]:
    """Assemble system blocks in stable-to-volatile order for cache hits."""
    blocks: list[PromptBlock] = []
    if include_preamble:
        blocks.append(PromptBlock(PLATFORM_PREAMBLE, cache=True))
    if run_context:
        blocks.append(PromptBlock(run_context, cache=True))
    blocks.append(PromptBlock(agent_instructions, cache=False))
    return blocks


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    @classmethod
    def from_response(cls, usage: Any) -> Usage:
        item = cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            calls=1,
        )
        item.cost_usd = estimate_cost_usd(
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            cache_write_tokens=item.cache_write_tokens,
            cache_read_tokens=item.cache_read_tokens,
        )
        return item

    def merge(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.calls += other.calls
        self.cost_usd += other.cost_usd

    @property
    def total_prompt_tokens(self) -> int:
        """Full prompt size. ``input_tokens`` alone is the *uncached* remainder."""
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


@dataclass(slots=True)
class LLMResult:
    """What a call returns, alongside its cost and reasoning trace."""

    text: str = ""
    thinking: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    model: str = ""
    seconds: float = 0.0
    attempts: int = 1
    fallback_used: bool = False


@dataclass(slots=True)
class StructuredResult(LLMResult):
    """A validated Pydantic instance plus the same telemetry."""

    value: Any = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin, opinionated wrapper over the Anthropic SDK.

    Not a general-purpose abstraction â€” it exposes exactly the two call shapes
    this platform needs (validated structured output, and long-form prose) and
    accumulates usage across a run.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._injected = client
        self._sdk: anthropic.Anthropic | None = client
        self.usage = Usage()

    @property
    def _client(self) -> anthropic.Anthropic:
        """The SDK client, constructed on first use rather than at import.

        Constructing ``anthropic.Anthropic()`` raises when no credential is
        resolvable, so building it eagerly would make *instantiating an agent*
        require an API key even in offline mode, where no call is ever made.
        Deferring it means credentials are needed only by code that actually
        talks to the API.
        """
        if self._sdk is None:
            if self.settings.offline:
                raise LLMOfflineError(
                    "the LLM client was asked for a connection while offline mode "
                    "is enabled"
                )
            # The SDK's own retry handles transport-level failures
            # (429/5xx/connection) with correct backoff; our loop above it handles
            # schema and truncation problems, which need a modified request rather
            # than a replay.
            self._sdk = anthropic.Anthropic(
                max_retries=self.settings.llm_max_retries,
                timeout=self.settings.llm_timeout_seconds,
            )
        return self._sdk

    @_client.setter
    def _client(self, value: anthropic.Anthropic) -> None:
        """Allow direct injection, which is how tests swap in a stub transport."""
        self._sdk = value

    def _refuse_if_offline(self, what: str) -> None:
        """Fail loudly rather than reaching for credentials that should not be used.

        Offline runs are routed away from this class at the agent layer. Reaching
        here means a code path was missed, and a clear error beats an accidental
        billed call.
        """
        if self.settings.offline:
            raise LLMOfflineError(
                f"{what} was called while offline mode is enabled; this code path "
                "should have been routed to automl_architect.core.offline"
            )

    # -- internals --------------------------------------------------------

    def _system_param(self, blocks: list[PromptBlock]) -> list[dict[str, Any]]:
        """Render system blocks, marking the stable ones with a cache breakpoint.

        The TTL is deliberately not the 5-minute default. Agents in one run take
        60-400s each, so a 5-minute entry expires before the next agent reuses
        it â€” a measured run wrote 67k tokens to cache and read back zero, paying
        the write premium eleven times for nothing. A 1-hour entry costs 2x to
        write instead of 1.25x but is read at 0.1x by every later agent, which
        breaks even at three calls and there are eleven.
        """
        out: list[dict[str, Any]] = []
        caching = self.settings.enable_prompt_caching
        ttl = self.settings.cache_ttl
        for block in blocks:
            if not block.text.strip():
                continue
            entry: dict[str, Any] = {"type": "text", "text": block.text}
            if caching and block.cache:
                cache_control: dict[str, Any] = {"type": "ephemeral"}
                if ttl and ttl != "5m":
                    cache_control["ttl"] = ttl
                entry["cache_control"] = cache_control
            out.append(entry)
        return out

    def _thinking_config(self, effort: Effort) -> tuple[dict[str, Any] | None, Effort]:
        """Return the ``thinking`` param and a possibly-corrected effort.

        Opus 5 rejects ``{"type": "disabled"}`` above effort ``high``, so when
        thinking is switched off we clamp effort instead of emitting a 400.
        """
        if self.settings.enable_thinking:
            return {"type": "adaptive", "display": "summarized"}, effort
        if effort in ("xhigh", "max"):
            logger.warning(
                "thinking disabled at effort=%s is rejected by the API; "
                "clamping effort to 'high'",
                effort,
            )
            effort = "high"
        return {"type": "disabled"}, effort

    def _fallback_kwargs(self) -> dict[str, Any]:
        if not self.settings.enable_refusal_fallback:
            return {}
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}

    @staticmethod
    def _extract(message: Any) -> tuple[str, str, bool]:
        """Pull visible text, thinking summary, and a fallback flag off a message."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        fallback_used = False
        for block in getattr(message, "content", []) or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "thinking":
                summary = getattr(block, "thinking", "") or ""
                if summary:
                    thinking_parts.append(summary)
            elif kind == "fallback":
                fallback_used = True
        # Sticky fallback turns carry no `fallback` block, so also check usage.
        iterations = getattr(getattr(message, "usage", None), "iterations", None) or []
        if any(getattr(i, "type", "") == "fallback_message" for i in iterations):
            fallback_used = True
        return "\n".join(text_parts), "\n\n".join(thinking_parts), fallback_used

    @staticmethod
    def _check_refusal(message: Any) -> None:
        if getattr(message, "stop_reason", None) != "refusal":
            return
        details = getattr(message, "stop_details", None)
        raise LLMRefusalError(
            getattr(details, "category", None),
            getattr(details, "explanation", None),
        )

    def _record(self, usage: Usage) -> None:
        self.usage.merge(usage)

    @staticmethod
    def _sleep(attempt: int) -> None:
        time.sleep(min(2.0**attempt + random.uniform(0, 0.5), 20.0))

    # -- public API -------------------------------------------------------

    def structured(
        self,
        *,
        output_model: type[T],
        user: str,
        system: list[PromptBlock] | None = None,
        agent_instructions: str | None = None,
        run_context: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
        max_attempts: int = 3,
    ) -> StructuredResult:
        """Call the model and return a validated ``output_model`` instance.

        Retries on truncation (raising ``max_tokens``) and on schema-validation
        failure (re-asking with the validator's complaint appended), because
        both are fixed by changing the request rather than replaying it.
        """
        self._refuse_if_offline("LLMClient.structured")
        if system is None:
            if agent_instructions is None:
                raise ValueError("provide either `system` or `agent_instructions`")
            system = build_system(
                run_context=run_context, agent_instructions=agent_instructions
            )

        effort = effort or self.settings.default_effort  # type: ignore[assignment]
        budget = max_tokens or self.settings.max_output_tokens
        thinking, effort = self._thinking_config(effort)  # type: ignore[arg-type]

        prompt = user
        last_error: Exception | None = None
        started = time.perf_counter()

        for attempt in range(1, max_attempts + 1):
            try:
                # `output_format=<pydantic class>` is what makes `parse()` return a
                # validated instance: it massages the schema into the subset the
                # grammar compiler accepts and re-checks constraints client-side.
                # The SDK marks the parameter deprecated in favour of
                # `output_config.format`, but that field only takes a raw schema
                # dict â€” passing the class raises "ModelMetaclass is not JSON
                # serializable" â€” so migrating today would mean hand-rolling both
                # the schema massaging and the validation. The warning is
                # silenced rather than the call changed, and this comment is the
                # reminder to revisit when the SDK accepts a model class there.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*output_format.*deprecated.*",
                        category=DeprecationWarning,
                    )
                    message = self._client.beta.messages.parse(
                        model=self.settings.model,
                        max_tokens=budget,
                        system=self._system_param(system),
                        messages=[{"role": "user", "content": prompt}],
                        output_format=output_model,
                        output_config={"effort": effort},
                        **({"thinking": thinking} if thinking else {}),
                        **self._fallback_kwargs(),
                    )
            except _TRANSIENT as exc:
                # The SDK already exhausted its own retries by this point.
                last_error = exc
                logger.warning(
                    "transient LLM failure (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt == max_attempts:
                    raise LLMTransientError(str(exc)) from exc
                self._sleep(attempt)
                continue
            except anthropic.BadRequestError as exc:
                message = str(exc).lower()
                if any(marker in message for marker in _RETRYABLE_BAD_REQUEST):
                    last_error = exc
                    logger.warning(
                        "schema compilation rejected (attempt %d/%d): %s",
                        attempt,
                        max_attempts,
                        exc,
                    )
                    if attempt == max_attempts:
                        raise LLMTransientError(
                            f"structured output unavailable after {max_attempts} "
                            f"attempts: {exc}"
                        ) from exc
                    # Longer than the transient backoff: the server needs room to
                    # finish compiling, and the result is cached once it does.
                    time.sleep(min(5.0 * attempt, 30.0))
                    continue
                # Genuinely malformed request. Never retryable, and the message
                # is the only useful diagnostic.
                raise LLMOutputError(f"request rejected: {exc}") from exc

            usage = Usage.from_response(message.usage)
            self._record(usage)
            self._check_refusal(message)

            text, thinking_text, fallback_used = self._extract(message)

            if message.stop_reason == "max_tokens":
                last_error = LLMOutputError("response truncated at max_tokens")
                if attempt == max_attempts:
                    raise last_error
                budget = min(budget * 2, 128_000)
                logger.warning("response truncated; retrying with max_tokens=%d", budget)
                continue

            value = getattr(message, "parsed_output", None)
            if value is None:
                # Schema-constrained output should always parse; if it did not,
                # re-ask once with the raw text so the model can self-correct.
                last_error = LLMOutputError("model returned no parseable output")
                if attempt == max_attempts:
                    raise last_error
                prompt = (
                    f"{user}\n\nYour previous reply could not be parsed into the "
                    f"required schema. Return only a valid object matching it."
                )
                continue

            if not isinstance(value, output_model):
                try:
                    value = output_model.model_validate(
                        value if isinstance(value, dict) else value.model_dump()
                    )
                except PydanticValidationError as exc:
                    last_error = LLMOutputError(f"schema validation failed: {exc}")
                    if attempt == max_attempts:
                        raise last_error from exc
                    prompt = (
                        f"{user}\n\nYour previous reply failed validation with:\n"
                        f"{exc}\n\nReturn a corrected object."
                    )
                    continue

            return StructuredResult(
                value=value,
                text=text,
                thinking=thinking_text,
                usage=usage,
                stop_reason=message.stop_reason,
                model=getattr(message, "model", self.settings.model),
                seconds=time.perf_counter() - started,
                attempts=attempt,
                fallback_used=fallback_used,
            )

        raise last_error or LLMOutputError("structured call failed without a cause")

    def text(
        self,
        *,
        user: str,
        system: list[PromptBlock] | None = None,
        agent_instructions: str | None = None,
        run_context: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Free-form prose. Streams, so long outputs cannot hit an HTTP timeout."""
        self._refuse_if_offline("LLMClient.text")
        if system is None:
            if agent_instructions is None:
                raise ValueError("provide either `system` or `agent_instructions`")
            system = build_system(
                run_context=run_context, agent_instructions=agent_instructions
            )

        effort = effort or self.settings.default_effort  # type: ignore[assignment]
        budget = max_tokens or 32_000
        thinking, effort = self._thinking_config(effort)  # type: ignore[arg-type]
        started = time.perf_counter()

        try:
            with self._client.beta.messages.stream(
                model=self.settings.model,
                max_tokens=budget,
                system=self._system_param(system),
                messages=[{"role": "user", "content": user}],
                output_config={"effort": effort},
                **({"thinking": thinking} if thinking else {}),
                **self._fallback_kwargs(),
            ) as stream:
                message = stream.get_final_message()
        except _TRANSIENT as exc:
            raise LLMTransientError(str(exc)) from exc

        usage = Usage.from_response(message.usage)
        self._record(usage)
        self._check_refusal(message)
        text, thinking_text, fallback_used = self._extract(message)

        return LLMResult(
            text=text,
            thinking=thinking_text,
            usage=usage,
            stop_reason=message.stop_reason,
            model=getattr(message, "model", self.settings.model),
            seconds=time.perf_counter() - started,
            fallback_used=fallback_used,
        )

    def count_tokens(self, *, system: list[PromptBlock], user: str) -> int:
        """Exact token count for a prompt. Never estimate with a third-party tokenizer."""
        self._refuse_if_offline("LLMClient.count_tokens")
        result = self._client.messages.count_tokens(
            model=self.settings.model,
            system=self._system_param(system),
            messages=[{"role": "user", "content": user}],
        )
        return result.input_tokens


_shared: LLMClient | None = None


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Process-wide client, so prompt-cache prefixes and usage totals are shared."""
    global _shared
    if _shared is None:
        _shared = LLMClient(settings=settings)
    return _shared


def reset_llm_client() -> None:
    global _shared
    _shared = None
