"""Retry, budget, approval, and replan policies.

These four decisions are pulled out of the engine on purpose. Each is a pure(ish)
rule over run state that benefits from being read and tested in isolation, and
each encodes a judgement that would otherwise be buried in control flow:

*   **Retry** branches on the exception *hierarchy*, never on message text. A
    transient LLM failure or a schema violation is worth another attempt; a
    missing optional dependency will fail identically forever.
*   **Budget** protects the deliverable. When the clock runs short the answer is
    to drop tuning, explainability, and extra charts — not to ship a truncated
    report — and to record which corner was cut and why.
*   **Approval** is the human-in-the-loop gate. It is idempotent across
    suspension and resume, because a run that asked twice for the same
    permission would be indistinguishable from a run that ignored the answer.
*   **Replan** turns an :class:`EvaluationVerdict` into a restart point. It is
    the only place that decides the self-improvement loop should go round again.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Sequence

from ..core.errors import (
    AgentError,
    ApprovalRejected,
    ApprovalRequired,
    ConfigurationError,
    FatalError,
    ReplanRequested,
    RetryableError,
    RunCancelled,
)
from ..core.schemas import (
    AgentName,
    ApprovalRequest,
    EventKind,
    EvaluationVerdict,
    Severity,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from ..core.state import RunState
    from .graph import StepDefinition

logger = logging.getLogger(__name__)

#: A replan re-runs training end to end. Below this much remaining wall clock
#: there is no point starting one — the run would suspend mid-training and ship
#: nothing at all.
MIN_REPLAN_SECONDS = 120.0

#: ``recommended_action`` -> the step a revised plan should restart from.
#: ``None`` means the action is not something re-running the pipeline can fix.
ACTION_RESTART_STEP: dict[str, str | None] = {
    "accept": None,
    "retry_cleaning": "clean",
    "retry_feature_engineering": "engineer_features",
    "retry_model_selection": "select_models",
    # A rejected model needs the widest possible change, so restart at the
    # earliest step a revised plan can influence.
    "reject": "clean",
    # No amount of replanning conjures more rows.
    "collect_more_data": None,
}


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def _causes(exc: BaseException, limit: int = 8) -> Iterable[BaseException]:
    """Walk the ``__cause__``/``__context__` chain, cycle-safe."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(seen) < limit:
        if id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


@dataclass(slots=True)
class RetryPolicy:
    """How many times to re-run a failed step, and how long to wait.

    Attributes:
        attempts: Total attempts per step, including the first.
        backoff_seconds: Delay before the second attempt.
        multiplier: Exponential growth factor for later attempts.
        max_backoff_seconds: Ceiling on any single wait.
        jitter: Fraction of the delay added at random, to de-correlate retries.
    """

    attempts: int = 2
    backoff_seconds: float = 1.5
    multiplier: float = 2.0
    max_backoff_seconds: float = 30.0
    jitter: float = 0.25

    def is_retryable(self, exc: BaseException) -> bool:
        """Whether re-running the step could plausibly succeed.

        Control-flow signals (approval, cancellation, replan) are never retried,
        and neither is anything under :class:`FatalError` — a missing optional
        dependency or a model refusal fails identically on every attempt.

        ``AgentError`` needs care: :meth:`BaseAgent.run` wraps *any* exception
        from the model call in a bare ``AgentError``, which is not itself a
        ``RetryableError``. So a rate limit would look permanent unless the cause
        chain is inspected, which is what this does.
        """
        if isinstance(exc, (ApprovalRequired, ApprovalRejected, RunCancelled, ReplanRequested)):
            return False
        if isinstance(exc, RetryableError):
            return True
        if isinstance(exc, FatalError):
            return False
        if isinstance(exc, AgentError):
            return not any(isinstance(cause, FatalError) for cause in _causes(exc))
        return False

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        """Whether ``attempt`` (1-based) may be followed by another."""
        return attempt < max(1, self.attempts) and self.is_retryable(exc)

    def delay_for(self, attempt: int) -> float:
        """Seconds to wait after a failed ``attempt`` (1-based)."""
        base = self.backoff_seconds * (self.multiplier ** max(0, attempt - 1))
        capped = min(base, self.max_backoff_seconds)
        return capped * (1.0 + random.uniform(0.0, self.jitter))

    def sleep(self, attempt: int) -> float:
        """Block for :meth:`delay_for` and return the delay actually used."""
        delay = self.delay_for(attempt)
        time.sleep(delay)
        return delay


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BudgetVerdict:
    """Whether a step may run, and the reason when it may not."""

    allowed: bool
    reason: str = ""
    over_budget: bool = False


@dataclass(slots=True)
class BudgetPolicy:
    """Spends the wall-clock budget on the deliverable first.

    Attributes:
        reserve_seconds: Time held back for evaluation, insights, and report
            writing. Optional steps are dropped once the remainder falls into
            this reserve.
        min_optional_seconds: An optional step is not worth starting with less
            than this much time left, regardless of the reserve.
    """

    reserve_seconds: float = 90.0
    min_optional_seconds: float = 45.0

    def decide(self, state: RunState, step: StepDefinition) -> BudgetVerdict:
        """Whether ``step`` should run given the time left on the run."""
        remaining = state.time_remaining
        if not step.budget_optional:
            if remaining <= 0.0:
                return BudgetVerdict(
                    allowed=True,
                    over_budget=True,
                    reason=(
                        f"the {state.config.time_budget_seconds}s time budget is spent, "
                        f"but '{step.step_id}' is required to produce a usable result, "
                        "so it ran anyway"
                    ),
                )
            return BudgetVerdict(allowed=True)

        floor = max(self.reserve_seconds, self.min_optional_seconds)
        if remaining <= floor:
            return BudgetVerdict(
                allowed=False,
                over_budget=remaining <= 0.0,
                reason=(
                    f"only {remaining:.0f}s of the {state.config.time_budget_seconds}s "
                    f"budget remain (reserve is {floor:.0f}s), so the optional step "
                    f"'{step.step_id}' was skipped to protect evaluation and reporting"
                ),
            )
        return BudgetVerdict(allowed=True)

    def allowance_for(self, state: RunState, step: StepDefinition) -> float:
        """Seconds an optional step may consume without eating the reserve."""
        remaining = state.time_remaining
        if not step.budget_optional:
            return max(0.0, remaining)
        return max(5.0, remaining - self.reserve_seconds)

    def chart_budget(self, state: RunState) -> int | None:
        """Maximum charts to render, or ``None`` for no cap.

        Charts are the most elastic cost in the run: each is a plotly render plus
        a file write, and the report degrades gracefully with fewer of them.
        """
        remaining = state.time_remaining
        if remaining > 300.0:
            return None
        if remaining > 180.0:
            return 8
        if remaining > 90.0:
            return 4
        return 2

    def summary(self, state: RunState) -> str:
        """One line describing the budget position, for logs and events."""
        return (
            f"{state.elapsed_seconds:.0f}s elapsed, {state.time_remaining:.0f}s of "
            f"{state.config.time_budget_seconds}s remaining"
        )


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ApprovalPolicy:
    """Suspends the run before a destructive step when a human must sign off.

    Attributes:
        require_approval: Mirrors ``RunConfig.require_approval``. When false the
            policy is a no-op and never touches state.
    """

    require_approval: bool = False

    @staticmethod
    def latest_for_step(state: RunState, step_id: str) -> ApprovalRequest | None:
        """The most recent request raised for ``step_id``, if any."""
        matches = [a for a in state.approvals if a.step_id == step_id]
        return matches[-1] if matches else None

    def check(
        self,
        state: RunState,
        *,
        step_id: str,
        agent: AgentName | None,
        action_summary: str,
        details: Sequence[str] = (),
        affected_columns: Sequence[str] = (),
        affected_row_estimate: int = 0,
        severity: Severity = Severity.MEDIUM,
        destructive: bool = True,
    ) -> None:
        """Gate a destructive action on a recorded human decision.

        Raises:
            ApprovalRequired: The run must suspend and be resumed later.
            ApprovalRejected: A human declined this step.
        """
        if not (self.require_approval and destructive):
            return

        prior = self.latest_for_step(state, step_id)
        if prior is not None:
            if prior.decision == "approved":
                return
            if prior.decision == "rejected":
                raise ApprovalRejected(
                    f"step '{step_id}' was rejected by "
                    f"{prior.decided_by or 'a reviewer'}"
                    f"{f': {prior.note}' if prior.note else ''}"
                )
            # Still pending: re-suspend against the same request rather than
            # queueing a second one, so resume(request_id) stays unambiguous.
            raise ApprovalRequired(prior.request_id, step_id, prior.action_summary)

        request = ApprovalRequest(
            step_id=step_id,
            agent=agent or AgentName.PLANNER,
            action_summary=action_summary,
            details=list(details),
            affected_columns=list(affected_columns),
            affected_row_estimate=int(affected_row_estimate),
            severity=severity,
        )
        state.approvals.append(request)
        state.bus.emit(
            EventKind.APPROVAL_REQUESTED,
            action_summary,
            agent=agent,
            step_id=step_id,
            payload={
                "request_id": request.request_id,
                "severity": severity.value,
                "affected_columns": list(affected_columns)[:50],
                "affected_row_estimate": int(affected_row_estimate),
            },
        )
        raise ApprovalRequired(request.request_id, step_id, action_summary)

    def resolve(
        self,
        state: RunState,
        request_id: str,
        approved: bool,
        note: str | None = None,
        *,
        decided_by: str = "operator",
    ) -> ApprovalRequest:
        """Record a human decision on a pending request.

        Args:
            state: The suspended run's state.
            request_id: Id from the :class:`ApprovalRequired` signal.
            approved: The decision.
            note: Optional free-text justification, kept in the audit trail.
            decided_by: Who decided.

        Returns:
            The updated request. Re-resolving an already-decided request is a
            no-op so a duplicated callback cannot flip a decision.

        Raises:
            ConfigurationError: No request with that id exists on this run.
        """
        request = next(
            (a for a in state.approvals if a.request_id == request_id), None
        )
        if request is None:
            known = [a.request_id for a in state.approvals if a.decision == "pending"]
            raise ConfigurationError(
                f"no approval request '{request_id}' on run {state.run_id}; "
                f"pending requests are {known or 'none'}"
            )
        if request.decision != "pending":
            logger.warning(
                "approval %s was already %s; ignoring the new decision",
                request_id,
                request.decision,
            )
            return request

        request.decision = "approved" if approved else "rejected"
        request.decided_at = datetime.now(timezone.utc)
        request.decided_by = decided_by
        request.note = note
        state.bus.emit(
            EventKind.APPROVAL_RESOLVED,
            f"step '{request.step_id}' {request.decision} by {decided_by}"
            f"{f': {note}' if note else ''}",
            agent=request.agent,
            step_id=request.step_id,
            payload={"request_id": request_id, "decision": request.decision},
        )
        return request


# ---------------------------------------------------------------------------
# Replan
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReplanDecision:
    """Whether to go round the self-improvement loop again, and from where."""

    should_replan: bool
    restart_step_id: str | None = None
    reason: str = ""


@dataclass(slots=True)
class ReplanPolicy:
    """Turns an evaluation verdict into a restart point.

    Attributes:
        max_replans: Hard bound from ``RunConfig.max_replans``.
        enabled: Mirrors ``RunConfig.enable_self_improvement``.
    """

    max_replans: int = 2
    enabled: bool = True

    def decide(
        self,
        verdict: EvaluationVerdict | None,
        *,
        replans_done: int,
        min_acceptable_score: float | None = None,
        best_score: float | None = None,
        higher_is_better: bool = True,
        time_remaining: float | None = None,
    ) -> ReplanDecision:
        """Decide whether the pipeline should be revised and re-run.

        Args:
            verdict: The Evaluation Agent's output. ``None`` disables the loop —
                without a quality gate there is nothing to react to.
            replans_done: How many replans this run has already spent.
            min_acceptable_score: Operator floor from the run config.
            best_score: The best primary-metric score achieved so far.
            higher_is_better: Direction of the primary metric.
            time_remaining: Seconds left in the wall-clock budget.

        Returns:
            A :class:`ReplanDecision` whose ``reason`` is always populated, so
            the run record explains both replans and refusals to replan.
        """
        if verdict is None:
            return ReplanDecision(
                False, reason="no evaluation verdict was produced, so no replan"
            )

        below_floor = self._below_floor(
            min_acceptable_score, best_score, higher_is_better
        )
        action = str(verdict.recommended_action)

        if verdict.acceptable and not below_floor and action == "accept":
            return ReplanDecision(
                False,
                reason=(
                    f"evaluation accepted the model (grade {verdict.overall_grade}), "
                    "so the plan stands"
                ),
            )

        if not self.enabled:
            return ReplanDecision(
                False,
                reason=(
                    "evaluation recommended "
                    f"'{action}' but self-improvement is disabled in the run config"
                ),
            )
        if replans_done >= max(0, self.max_replans):
            return ReplanDecision(
                False,
                reason=(
                    f"evaluation recommended '{action}' but the replan budget "
                    f"({self.max_replans}) is exhausted; shipping the best model found"
                ),
            )
        if time_remaining is not None and time_remaining < MIN_REPLAN_SECONDS:
            return ReplanDecision(
                False,
                reason=(
                    f"evaluation recommended '{action}' but only "
                    f"{time_remaining:.0f}s remain, which is not enough to retrain; "
                    "shipping the best model found"
                ),
            )

        restart = ACTION_RESTART_STEP.get(action)
        if restart is None:
            if below_floor:
                # The agent had no actionable retry but the operator's floor is
                # unmet; feature engineering is the cheapest lever that can move
                # the score without more data.
                restart = "engineer_features"
            else:
                return ReplanDecision(
                    False,
                    reason=(
                        f"evaluation recommended '{action}', which re-running the "
                        "pipeline cannot address"
                    ),
                )

        if below_floor and best_score is not None:
            reason = (
                f"best {('' if higher_is_better else 'lower-is-better ')}score "
                f"{best_score:.6g} misses the operator floor of "
                f"{min_acceptable_score:.6g}; replanning from '{restart}'"
            )
        else:
            reason = (
                f"evaluation graded the model {verdict.overall_grade} and recommended "
                f"'{action}'; replanning from '{restart}'"
            )
        return ReplanDecision(True, restart_step_id=restart, reason=reason)

    @staticmethod
    def _below_floor(
        floor: float | None, score: float | None, higher_is_better: bool
    ) -> bool:
        if floor is None or score is None:
            return False
        return score < floor if higher_is_better else score > floor


__all__ = [
    "ACTION_RESTART_STEP",
    "MIN_REPLAN_SECONDS",
    "ApprovalPolicy",
    "BudgetPolicy",
    "BudgetVerdict",
    "ReplanDecision",
    "ReplanPolicy",
    "RetryPolicy",
]
