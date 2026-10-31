"""Failure-injection checks for the runtime, providers, credentials and recovery."""

import json
import multiprocessing
import threading
import time
from types import SimpleNamespace as NS

import httpx
import joblib
import pandas as pd
import pytest
from openai import OpenAI
from pydantic import BaseModel

from automl_architect.config import Settings
from automl_architect.core.errors import (
    BudgetExceeded,
    ConfigurationError,
    LLMRefusalError,
    RunCancelled,
)
from automl_architect.core.events import EventBus
from automl_architect.core.llm import LLMClient
from automl_architect.core.schemas import DataSource, RunConfig, SourceKind
from automl_architect.core.state import RunState
from automl_architect.storage.checkpoints import load_checkpoint, save_checkpoint


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize(
    "settings_provider,settings_model,run_provider,run_model,expected",
    [
        ("openai", "gpt-custom", None, None, "gpt-custom"),
        ("openai", "claude-opus-5", None, None, "gpt-default"),
        ("openai", "gpt-custom", None, "gpt-run", "gpt-run"),
        ("anthropic", "claude-custom", "openai", None, "gpt-default"),
        ("openai", "gpt-custom", "anthropic", None, "claude-opus-5"),
        ("anthropic", "claude-custom", None, None, "claude-custom"),
    ],
)
def test_run_honors_model_precedence(
    tmp_path, settings_provider, settings_model, run_provider, run_model, expected
):
    from automl_architect.orchestrator.engine import Orchestrator

    settings = Settings(
        _env_file=None,
        workspace=tmp_path,
        offline=True,
        provider=settings_provider,
        model=settings_model,
        openai_model="gpt-default",
    )
    config = RunConfig(
        source=DataSource(kind=SourceKind.CSV, uri="unused.csv"),
        provider=run_provider,
        model=run_model,
    )
    run = Orchestrator(config, settings=settings)
    assert run.state.settings.model == expected
    assert run.state.config.model == expected


def run_state(tmp_path, **limits):
    settings = Settings(
        _env_file=None,
        workspace=tmp_path,
        offline=False,
        openai_api_key="never-persist-this-key",
        provider="openai",
        model="test-model",
        input_price_per_million=1,
        output_price_per_million=2,
    )
    config = RunConfig(source=DataSource(kind=SourceKind.CSV, uri="unused.csv"), **limits)
    state = RunState(config=config, settings=settings, bus=EventBus(config.run_id))
    state.mark_started()
    return state


def openai_response(refusal=False):
    content = (
        [{"type": "refusal", "refusal": "No"}]
        if refusal
        else [{"type": "output_text", "text": '{"answer":"ok"}', "annotations": []}]
    )
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "status": "completed",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content,
            }
        ],
        "usage": {
            "input_tokens": 40,
            "output_tokens": 10,
            "total_tokens": 50,
            "input_tokens_details": {"cached_tokens": 12},
        },
    }


def sdk_client(handler):
    return OpenAI(
        api_key="test-key",
        base_url="https://provider.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_openai_structured_contract_and_retry_budget(tmp_path):
    requests = []

    def transport(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                429, json={"error": {"message": "busy", "type": "rate_limit_error"}}
            )
        return httpx.Response(200, json=openai_response())

    state = run_state(tmp_path)
    llm = LLMClient(state.settings, client=sdk_client(transport))
    llm.bind_run(state)
    llm._sleep = lambda _: None
    result = llm.structured(
        output_model=Answer, user="hello", agent_instructions="Return JSON", max_tokens=100
    )
    assert result.value.answer == "ok"
    assert len(requests) == state.usage.llm_calls == 2
    assert state.usage.input_tokens == 28
    assert state.usage.cache_read_tokens == 12
    assert state.usage.cost_usd > 0
    assert requests[-1]["store"] is False
    assert requests[-1]["text"]["format"]["type"] == "json_schema"
    assert state.extras["reserved_llm_tokens"] > 50  # unknown failed request stays reserved


def test_request_cap_stops_retry_before_network(tmp_path):
    calls = []

    def transport(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"message": "busy"}})

    state = run_state(tmp_path, max_llm_calls=1)
    llm = LLMClient(state.settings, client=sdk_client(transport))
    llm.bind_run(state)
    llm._sleep = lambda _: None
    with pytest.raises(BudgetExceeded):
        llm.structured(output_model=Answer, user="hi", agent_instructions="test", max_tokens=100)
    assert len(calls) == 1


@pytest.mark.parametrize("limit", [{"max_llm_tokens": 1}, {"max_cost_usd": 0.000001}])
def test_reservation_blocks_unaffordable_call(tmp_path, limit):
    state = run_state(tmp_path, **limit)

    def forbidden(_):
        pytest.fail("The budget should stop the network request.")

    llm = LLMClient(state.settings, client=sdk_client(forbidden))
    llm.bind_run(state)
    with pytest.raises(BudgetExceeded):
        llm.structured(output_model=Answer, user="test", agent_instructions="test", max_tokens=100)


def test_refusal_is_terminal_and_usage_is_recorded(tmp_path):
    state = run_state(tmp_path)
    llm = LLMClient(
        state.settings, client=sdk_client(lambda _: httpx.Response(200, json=openai_response(True)))
    )
    llm.bind_run(state)
    with pytest.raises(LLMRefusalError):
        llm.structured(output_model=Answer, user="test", agent_instructions="test", max_tokens=100)
    assert state.usage.llm_calls == 1
    assert state.usage.output_tokens == 10


def test_authentication_error_does_not_leak_key_or_retry(tmp_path):
    state = run_state(tmp_path)
    calls = []

    def reject(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "never-persist-this-key"}})

    llm = LLMClient(state.settings, client=sdk_client(reject))
    with pytest.raises(ConfigurationError) as error:
        llm.structured(output_model=Answer, user="test", agent_instructions="test", max_tokens=100)
    assert "never-persist" not in str(error.value)
    import traceback

    assert "never-persist" not in "".join(traceback.format_exception(error.value))
    assert len(calls) == 1


def test_anthropic_validation_retry_counts_both_responses(tmp_path):
    state = run_state(tmp_path)
    state.settings = state.settings.model_copy(
        update={"provider": "anthropic", "model": "test-claude"}
    )
    replies = iter([None, Answer(answer="ok")])

    def parse(**kwargs):
        return NS(
            parsed_output=next(replies),
            content=[],
            model="test-claude",
            stop_reason="end_turn",
            usage=NS(input_tokens=10, output_tokens=5),
        )

    fake = NS(beta=NS(messages=NS(parse=parse)))
    llm = LLMClient(state.settings, client=fake)
    llm.bind_run(state)
    result = llm.structured(
        output_model=Answer, user="hi", agent_instructions="test", max_tokens=100
    )
    assert result.usage.calls == 2
    assert state.usage.llm_calls == 2
    assert state.usage.output_tokens == 10


def test_checkpoint_restores_data_approvals_and_event_cursor(tmp_path):
    from automl_architect.core.schemas import AgentName, ApprovalRequest

    state = run_state(tmp_path)
    state.raw_df = pd.DataFrame({"x": [1, 2, 3]})
    state.bus.log("before restart")
    state.approvals = [
        ApprovalRequest(step_id="clean", agent=AgentName.CLEANING, action_summary="Review cleaning")
    ]
    name = save_checkpoint(state)
    # Settings, including API keys, are not part of the serialized state.
    payload = joblib.load(tmp_path / "runs" / state.run_id / "checkpoints" / name)
    assert "settings" not in payload
    restored = load_checkpoint(state.settings, state.run_id, EventBus(state.run_id))
    pd.testing.assert_frame_equal(restored.raw_df, state.raw_df)
    assert restored.approvals == state.approvals
    assert restored.bus.log("after restart").sequence > state.bus.last_sequence


def test_corrupted_checkpoint_is_rejected_before_deserialization(tmp_path, monkeypatch):
    state = run_state(tmp_path)
    name = save_checkpoint(state)
    path = tmp_path / "runs" / state.run_id / "checkpoints" / name
    path.write_bytes(path.read_bytes() + b"tampered")
    monkeypatch.setattr(
        joblib, "load", lambda _: pytest.fail("Unauthenticated pickle must not load")
    )
    with pytest.raises(ValueError, match="authentication"):
        load_checkpoint(state.settings, state.run_id, EventBus(state.run_id))


def _sleeping_worker(*args):
    time.sleep(30)


@pytest.mark.parametrize("cancel", [False, True])
def test_computation_worker_is_stopped_on_deadline_or_cancel(tmp_path, monkeypatch, cancel):
    from automl_architect.execution import runtime

    state = run_state(tmp_path, time_budget_seconds=1 if not cancel else 30)
    monkeypatch.setattr(runtime, "_worker", _sleeping_worker)
    original_children = {p.pid for p in multiprocessing.active_children()}
    timer = threading.Timer(0.4, lambda: state.extras.update(cancel_requested=True))
    if cancel:
        timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(RunCancelled if cancel else BudgetExceeded):
            runtime.run_computation(state, "train")
    finally:
        timer.cancel()
    assert time.monotonic() - start < 8
    assert {p.pid for p in multiprocessing.active_children()} <= original_children
