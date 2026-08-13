"""Top-level entry points.

Everything below is convenience over :class:`~automl_architect.orchestrator.engine.Orchestrator`.
The value it adds is turning the three things a caller actually has — a file
path, a dataframe, or a fully specified :class:`DataSource` — into a validated
:class:`RunConfig`, and giving the run somewhere to be remembered so that dataset
memory has history to consult on the next run.

Typical use::

    from automl_architect import analyse

    summary = analyse("data/churn.csv", target="churned", time_budget_seconds=600)
    print(summary.report.executive_summary)

The British and American spellings are both exported; ``analyze`` is an alias for
the same function, not a second implementation.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from .config import Settings, get_settings
from .core.errors import ConfigurationError
from .core.schemas import (
    DataSource,
    QuestionAnswer,
    RunConfig,
    RunSummary,
)
from .orchestrator.engine import Orchestrator, state_from_summary

logger = logging.getLogger(__name__)


def to_data_source(source: Any, **options: Any) -> DataSource:
    """Coerce a caller-supplied source into a :class:`DataSource`.

    Inference is delegated to :func:`automl_architect.ingestion.router.infer_source`
    rather than reimplemented here. That module owns the connector registry, so it
    is the only place that knows which kinds are actually loadable — and keeping
    one table means ``analyse("data.parquet")`` and ``amla run data.parquet``
    cannot disagree about what that string means.

    Args:
        source: A ``DataSource``, a filesystem path or URI (``str``/``Path``), or
            an in-memory pandas DataFrame. Dataframes are parked in the ingestion
            layer's frame registry and referenced by token, since ``DataSource``
            is serialisable and cannot carry the data itself.
        **options: Extra connector options, recorded on the source.

    Returns:
        A ``DataSource`` ready for the ingestion router.

    Raises:
        ConfigurationError: The source kind could not be inferred, or the
            ingestion layer is unavailable.
    """
    if isinstance(source, DataSource) and not options:
        return source
    try:
        from .ingestion.router import infer_source
    except ImportError as exc:  # pragma: no cover - depends on sibling module
        raise ConfigurationError(
            f"the ingestion layer is unavailable, so '{source}' cannot be resolved "
            f"into a data source: {exc}"
        ) from exc
    return infer_source(source, **options)


def build_run_config(
    source: Any, *, target: str | None = None, **config_kwargs: Any
) -> RunConfig:
    """Assemble a :class:`RunConfig`, validating the keyword arguments.

    Unknown keywords are rejected rather than dropped: silently ignoring
    ``time_budget=60`` (instead of ``time_budget_seconds``) would produce a run
    that quietly disobeys its operator.

    Args:
        source: Anything :func:`to_data_source` accepts.
        target: The target column, if known. Equivalent to
            ``target_column=...``.
        **config_kwargs: Any other field of :class:`RunConfig`.

    Returns:
        A validated run configuration.
    """
    known = set(RunConfig.model_fields)
    unknown = sorted(set(config_kwargs) - known)
    if unknown:
        raise ConfigurationError(
            f"unknown run configuration option(s) {unknown}; valid options are "
            f"{sorted(known - {'source'})}"
        )
    if target is not None:
        config_kwargs.setdefault("target_column", target)
    config_kwargs["source"] = to_data_source(source)
    return RunConfig(**config_kwargs)


class AutoMLArchitect:
    """The library facade: analyse a dataset, then ask questions about the run.

    Args:
        settings: Process settings; the cached singleton by default.
        repository: Persistence backend. ``None`` builds the default
            :class:`~automl_architect.storage.repository.RunRepository` lazily so
            runs accumulate into dataset memory; pass ``False`` to run without
            any persistence.

    Example:
        >>> architect = AutoMLArchitect()                      # doctest: +SKIP
        >>> summary = architect.analyse("sales.csv", target="revenue")
        >>> architect.ask(summary.run_id, "Why was XGBoost chosen?").answer
    """

    def __init__(
        self, settings: Settings | None = None, repository: Any | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self._repository = repository
        self._repository_resolved = repository is not None
        self._last: Orchestrator | None = None

    # -- wiring ------------------------------------------------------------

    @property
    def repository(self) -> Any | None:
        """The persistence backend, built on first use unless disabled."""
        if not self._repository_resolved:
            self._repository_resolved = True
            try:
                from .storage.repository import RunRepository

                self._repository = RunRepository(settings=self.settings)
            except Exception as exc:  # noqa: BLE001 - persistence is optional
                logger.warning(
                    "run persistence is unavailable (%s); dataset memory and run "
                    "history will be disabled",
                    exc,
                )
                self._repository = None
        return self._repository or None

    @property
    def last_run(self) -> Orchestrator | None:
        """The most recent orchestrator, for resuming or cancelling a run."""
        return self._last

    # -- running -----------------------------------------------------------

    def analyse(
        self, source: Any, *, target: str | None = None, **config_kwargs: Any
    ) -> RunSummary:
        """Run the full pipeline against one dataset.

        Args:
            source: A path, URI, :class:`DataSource`, or pandas DataFrame.
            target: The column to predict. Omit it to let the agents propose one.
            **config_kwargs: Any :class:`RunConfig` field, e.g.
                ``time_budget_seconds``, ``max_experiments``, ``require_approval``.

        Returns:
            The run summary. Check ``status`` — an ``AWAITING_APPROVAL`` run needs
            :meth:`resume` before it will finish.
        """
        config = build_run_config(source, target=target, **config_kwargs)
        orchestrator = Orchestrator(
            config, settings=self.settings, repository=self.repository
        )
        self._last = orchestrator
        return orchestrator.run()

    def resume(
        self, request_id: str, approved: bool, note: str | None = None
    ) -> RunSummary:
        """Approve or reject a suspended step on the most recent run.

        Args:
            request_id: Id of the pending :class:`ApprovalRequest`.
            approved: Whether the destructive step may proceed.
            note: Optional justification for the audit trail.

        Returns:
            The summary of the continued run.
        """
        if self._last is None:
            raise ConfigurationError(
                "no run to resume; resume() only continues a run started by this "
                "instance in this process"
            )
        return self._last.resume(request_id, approved, note)

    def cancel(self) -> None:
        """Stop the most recent run at its next step boundary."""
        if self._last is not None:
            self._last.cancel()

    # -- reading -----------------------------------------------------------

    def get_run(self, run_id: str) -> RunSummary | None:
        """Load a persisted run summary, from storage or its artifact directory."""
        repository = self.repository
        if repository is not None:
            try:
                summary = repository.get_run(run_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not read run %s from storage: %s", run_id, exc)
                summary = None
            if summary is not None:
                return summary

        path = self.settings.runs_dir / run_id / "run_summary.json"
        if not path.exists():
            return None
        try:
            return RunSummary.model_validate(json.loads(path.read_text("utf-8")))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not parse %s: %s", path, exc)
            return None

    def list_runs(self, project: str | None = None, limit: int = 50) -> list[RunSummary]:
        """List persisted runs, newest first. Empty when persistence is off."""
        repository = self.repository
        if repository is None:
            return []
        try:
            return repository.list_runs(project=project, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not list runs: %s", exc)
            return []

    def ask(self, run_id: str, question: str) -> QuestionAnswer:
        """Answer a natural-language question about a finished run.

        The answer is grounded in that run's recorded state: its profile, plan,
        decisions, leaderboard, and evaluation. Nothing is recomputed, so a
        question about data the run never measured is answered as unknown rather
        than guessed.

        Args:
            run_id: The run to interrogate.
            question: A plain-language question.

        Returns:
            The agent's :class:`QuestionAnswer`.

        Raises:
            ConfigurationError: The run is unknown, or the QA agent is
                unavailable in this installation.
        """
        summary = self.get_run(run_id)
        if summary is None:
            raise ConfigurationError(
                f"run '{run_id}' was not found in storage or under "
                f"{self.settings.runs_dir}"
            )

        try:
            from .agents.qa import QAAgent
        except ImportError as exc:
            raise ConfigurationError(
                f"the question-answering agent is unavailable: {exc}"
            ) from exc

        state = state_from_summary(summary, settings=self.settings)
        state.extras["question"] = question
        repository = self.repository
        if repository is not None:
            try:
                state.extras["events"] = repository.get_events(run_id)
            except Exception as exc:  # noqa: BLE001 - evidence is best-effort
                logger.debug("could not load events for %s: %s", run_id, exc)

        answer = _invoke_qa(QAAgent, state, question)
        if isinstance(answer, QuestionAnswer):
            return answer
        try:
            return QuestionAnswer.model_validate(answer)
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(
                f"the QA agent returned {type(answer).__name__}, not a QuestionAnswer"
            ) from exc


def _invoke_qa(agent_class: type, state: Any, question: str) -> Any:
    """Call a QA agent whose exact entry point is not pinned by the contract.

    The agent reads the question from ``state.extras['question']`` and is driven
    through ``BaseAgent.run``; ``ask``/``answer`` taking the question as an
    argument are accepted as alternatives so this module does not force a
    particular shape on the agents layer.
    """
    accepts_question = False
    try:
        accepts_question = "question" in inspect.signature(agent_class).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-level callables
        accepts_question = False

    agent = agent_class(question=question) if accepts_question else agent_class()
    state.extras.setdefault("question", question)

    order = ("run", "ask", "answer")
    last_error: Exception | None = None
    for method_name in order:
        method = getattr(agent, method_name, None)
        if not callable(method):
            continue
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):  # pragma: no cover
            parameters = {}
        try:
            if len(parameters) >= 2:
                return method(state, question)
            return method(state)
        except TypeError as exc:  # signature mismatch: try the next shape
            last_error = exc
            continue
    raise ConfigurationError(
        f"{agent_class.__name__} exposes no usable entry point "
        f"(tried run/ask/answer): {last_error}"
    )


def analyse(source: Any, *, target: str | None = None, **kwargs: Any) -> RunSummary:
    """Analyse a dataset end to end with default wiring.

    Args:
        source: A path, URI, :class:`DataSource`, or pandas DataFrame.
        target: The column to predict. Omit it to let the agents propose one.
        **kwargs: Any :class:`RunConfig` field.

    Returns:
        The completed :class:`RunSummary`.

    Example:
        >>> summary = analyse("data/churn.csv", target="churned")  # doctest: +SKIP
        >>> summary.evaluation.overall_grade                       # doctest: +SKIP
        'B'
    """
    return AutoMLArchitect().analyse(source, target=target, **kwargs)


#: American-spelling alias for :func:`analyse`.
analyze = analyse


__all__ = [
    "AutoMLArchitect",
    "analyse",
    "analyze",
    "build_run_config",
    "to_data_source",
]
