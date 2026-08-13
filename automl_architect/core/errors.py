"""Exception hierarchy.

The split that matters is :class:`RetryableError` vs :class:`FatalError`: the
orchestrator's retry policy branches on those base classes rather than on
string matching, so a new error type slots in by choosing its parent.
"""

from __future__ import annotations


class AutoMLArchitectError(Exception):
    """Root of every error this package raises."""


class RetryableError(AutoMLArchitectError):
    """Transient. The orchestrator may re-run the step."""


class FatalError(AutoMLArchitectError):
    """Permanent for this configuration. Retrying will not help."""


# --- configuration -------------------------------------------------------


class ConfigurationError(FatalError):
    """Bad or missing configuration, including absent credentials."""


# --- ingestion -----------------------------------------------------------


class IngestionError(FatalError):
    """A source could not be read at all."""


class UnsupportedSourceError(IngestionError):
    """No connector is registered for the requested source kind."""


class MissingDependencyError(FatalError):
    """An optional dependency is required for the requested feature."""

    def __init__(self, package: str, feature: str, extra: str | None = None) -> None:
        hint = f"pip install {extra or package}"
        super().__init__(
            f"{feature} requires the optional package '{package}'. Install it with: {hint}"
        )
        self.package = package
        self.feature = feature


class ValidationError(FatalError):
    """Loaded data failed structural validation."""


# --- LLM -----------------------------------------------------------------


class LLMError(AutoMLArchitectError):
    """Base for model-invocation failures."""


class LLMTransientError(RetryableError, LLMError):
    """Rate limit, overload, timeout, or connection failure."""


class LLMRefusalError(FatalError, LLMError):
    """The model declined the request (``stop_reason == "refusal"``)."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(
            f"Model declined the request (category={category or 'unspecified'}): "
            f"{explanation or 'no explanation provided'}"
        )
        self.category = category
        self.explanation = explanation


class LLMOutputError(RetryableError, LLMError):
    """The response did not conform to the requested schema, or was truncated."""


class LLMOfflineError(FatalError, LLMError):
    """The API was reached for while offline mode is enabled.

    Offline runs are routed to the deterministic rule engine at the agent layer,
    so arriving here means a code path was missed. Failing loudly is deliberate:
    silently constructing a client would turn a configuration mistake into an
    unexpected billed call against credentials the operator asked not to use.
    """


class BudgetExceededError(FatalError):
    """A token, cost, or wall-clock budget was exhausted."""


# --- agents & orchestration ---------------------------------------------


class AgentError(AutoMLArchitectError):
    """An agent failed to produce a usable decision."""

    def __init__(self, agent: str, message: str) -> None:
        super().__init__(f"[{agent}] {message}")
        self.agent = agent


class AgentRetryableError(RetryableError, AgentError):
    """Agent failure worth another attempt."""


class ExecutionError(AutoMLArchitectError):
    """A deterministic executor failed while applying an agent's decision."""


class StepFailedError(AutoMLArchitectError):
    """A plan step exhausted its retries."""

    def __init__(self, step_id: str, cause: str) -> None:
        super().__init__(f"Step '{step_id}' failed: {cause}")
        self.step_id = step_id
        self.cause = cause


class ReplanRequested(AutoMLArchitectError):
    """Control-flow signal: evaluation rejected the model, revise the plan.

    Not an error condition — the self-improvement loop raises this to unwind
    back to the planner.
    """

    def __init__(self, reason: str, recommended_action: str) -> None:
        super().__init__(f"Replan requested ({recommended_action}): {reason}")
        self.reason = reason
        self.recommended_action = recommended_action


class ApprovalRequired(AutoMLArchitectError):
    """Control-flow signal: a destructive step is waiting on a human."""

    def __init__(self, request_id: str, step_id: str, summary: str) -> None:
        super().__init__(f"Approval required for step '{step_id}': {summary}")
        self.request_id = request_id
        self.step_id = step_id
        self.summary = summary


class ApprovalRejected(FatalError):
    """A human declined a destructive step."""


class RunCancelled(AutoMLArchitectError):
    """The run was cancelled by a caller."""


class TrainingError(ExecutionError):
    """Model fitting failed."""


class NoViableModelError(FatalError):
    """Every candidate model failed to train."""
