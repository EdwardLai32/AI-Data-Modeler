"""HTTP contract, driven through FastAPI's ``TestClient``.

Runs execute on a background thread, so the shape of these tests is: start a run,
poll until it reaches a terminal state, then assert on what the API exposes. The
fake LLM is installed as the process-wide client before the client is built, which
is what makes that safe — the worker thread resolves ``get_llm_client()`` to the
same fake, so nothing reaches the network from a thread the test cannot see.

The assertions focus on the contract a frontend actually depends on: documented
status codes, a resumable event cursor, artifacts served from inside the run
directory and nowhere else, and 404 rather than 500 for an unknown run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from automl_architect.config import reset_settings_cache
from automl_architect.core.schemas import RunStatus

from .conftest import CHURN_TARGET, FakeLLMClient, import_or_skip

#: Every read-side test needs a completed run, which is a full pipeline.
pytestmark = pytest.mark.slow

TERMINAL = {
    RunStatus.COMPLETED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
    RunStatus.AWAITING_APPROVAL.value,
}

#: A run small enough to finish inside a test, but complete enough that the
#: report and artifact endpoints have something real to serve. The row cap is
#: what keeps the wall clock down; nothing here changes which code paths run.
SMALL_RUN: dict[str, Any] = {
    "time_budget_seconds": 300,
    "max_experiments": 2,
    "max_rows": 900,
    "cv_folds": 3,
    "enable_tuning": False,
    "enable_explainability": False,
    "report_formats": ["markdown", "json"],
}


@pytest.fixture(scope="module")
def api_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One workspace and database for the whole module."""
    return tmp_path_factory.mktemp("api_ws")


@pytest.fixture(autouse=True)
def _pin_api_workspace(api_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Share one workspace and database across this module's tests.

    conftest's ``_offline_env`` gives every test a fresh workspace, which is right
    almost everywhere. Here it is not: a run started by one test has to be
    readable by the next, and both the run index and the artifact files live in
    that workspace. Module-level autouse fixtures run *after* the conftest ones,
    so re-pointing here keeps this module isolated from every other while letting
    its tests share one completed run instead of each paying for a pipeline.
    """
    monkeypatch.setenv("AUTOML_WORKSPACE", str(api_workspace))
    monkeypatch.setenv(
        "AUTOML_DATABASE_URL", f"sqlite+pysqlite:///{(api_workspace / 'api.db').as_posix()}"
    )
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def client(installed_fake_llm: FakeLLMClient) -> Iterator[Any]:
    """A TestClient over the real app, with the fake LLM already installed."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app_module = import_or_skip("automl_architect.api.app", feature="api/app.py")
    service = import_or_skip("automl_architect.api.service", feature="api/service.py")

    # A manager cached from an earlier test holds a stale settings snapshot.
    if hasattr(service, "reset_run_manager"):
        service.reset_run_manager()

    app = app_module.create_app()
    with TestClient(app) as test_client:
        yield test_client

    if hasattr(service, "reset_run_manager"):
        service.reset_run_manager()


def start_run(client: Any, churn_csv: Path, **overrides: Any) -> str:
    """Start a run over HTTP and return its id."""
    body = {"uri": str(churn_csv), "target_column": CHURN_TARGET, **SMALL_RUN, **overrides}
    response = client.post("/api/runs", json=body)
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def wait_for_terminal(client: Any, run_id: str, timeout: float = 300.0) -> dict[str, Any]:
    """Poll ``GET /api/runs/{id}`` until the run stops moving.

    Args:
        client: The test client.
        run_id: The run to follow.
        timeout: Seconds to wait before failing the test.

    Returns:
        The final run summary payload.
    """
    deadline = time.monotonic() + timeout
    payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        if response.status_code == 200:
            payload = response.json()
            if payload.get("status") in TERMINAL:
                return payload
        time.sleep(0.25)
    pytest.fail(f"run {run_id} did not finish within {timeout}s (last: {payload.get('status')})")


#: Run id of the module's one completed run, created on first use.
_COMPLETED: dict[str, str] = {}


@pytest.fixture
def finished_run(client: Any, churn_csv: Path) -> tuple[Any, dict[str, Any]]:
    """One completed run, shared across every read-side assertion.

    Executed once and then re-read from the shared database, because a full
    pipeline is tens of seconds and roughly fifteen assertions below only read the
    result. The re-read is itself worth something: it proves the summary survives
    a round trip through storage rather than only existing in the manager's memory.
    """
    if "run_id" not in _COMPLETED:
        run_id = start_run(client, churn_csv)
        created = wait_for_terminal(client, run_id)
        assert created["status"] == RunStatus.COMPLETED.value, created.get("error")
        _COMPLETED["run_id"] = run_id

    response = client.get(f"/api/runs/{_COMPLETED['run_id']}")
    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["status"] == RunStatus.COMPLETED.value
    return client, summary


# ===========================================================================
# Surface
# ===========================================================================


class TestSurface:
    def test_openapi_schema_is_generated(self, client: Any) -> None:
        """A broken response model shows up here before it shows up in a browser."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"]
        assert "/api/runs" in schema["paths"]

    def test_root_points_at_the_docs(self, client: Any) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "docs" in response.json()

    def test_health_reports_readiness(self, client: Any) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload
        assert any(key in payload for key in ("status", "ok", "healthy"))

    def test_health_does_not_leak_credentials(self, client: Any) -> None:
        """Health output is the most-scraped endpoint; it must not carry a password."""
        body = client.get("/api/health").text.lower()
        for secret in ("sk-ant", "password", "secret", "authorization"):
            assert secret not in body

    def test_expected_routes_are_registered(self, client: Any) -> None:
        paths = set(client.get("/openapi.json").json()["paths"])
        for expected in (
            "/api/health",
            "/api/runs",
            "/api/runs/{run_id}",
            "/api/runs/{run_id}/events",
        ):
            assert expected in paths, f"{expected} is not registered"

    def test_cors_headers_are_present(self, client: Any) -> None:
        """The browser UI is served from a different origin than the API."""
        response = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
        assert response.headers.get("access-control-allow-origin")


# ===========================================================================
# Starting runs
# ===========================================================================


class TestStartRun:
    def test_accepted_and_returns_follow_urls(self, client: Any, churn_csv: Path) -> None:
        response = client.post(
            "/api/runs",
            json={"uri": str(churn_csv), "target_column": CHURN_TARGET, **SMALL_RUN},
        )
        assert response.status_code == 202, response.text
        payload = response.json()
        assert payload["run_id"].startswith("run_")
        assert payload["events_url"].endswith("/events")
        assert payload["stream_url"].endswith("/stream")
        wait_for_terminal(client, payload["run_id"])

    def test_empty_body_is_rejected(self, client: Any) -> None:
        assert client.post("/api/runs", content=b"").status_code == 422

    def test_malformed_json_is_rejected(self, client: Any) -> None:
        response = client.post(
            "/api/runs",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_source_is_rejected(self, client: Any) -> None:
        assert client.post("/api/runs", json={"target_column": "y"}).status_code == 422

    def test_out_of_range_budget_is_rejected(self, client: Any, churn_csv: Path) -> None:
        """Validation lives in the request model, so a bad budget never reaches a thread."""
        response = client.post(
            "/api/runs",
            json={"uri": str(churn_csv), "target_column": CHURN_TARGET, "cv_folds": 500},
        )
        assert response.status_code == 422

    def test_upload_then_run(self, client: Any, churn_csv: Path) -> None:
        """The browser path: POST the file, then start a run against the stored copy."""
        with churn_csv.open("rb") as handle:
            response = client.post(
                "/api/upload", files={"file": ("churn.csv", handle, "text/csv")}
            )
        if response.status_code == 404:
            pytest.skip("no upload endpoint registered")
        assert response.status_code in (200, 201), response.text
        upload = response.json()
        assert upload.get("path")

        started = client.post(
            "/api/runs",
            json={"upload_path": upload["path"], "target_column": CHURN_TARGET, **SMALL_RUN},
        )
        assert started.status_code == 202, started.text
        summary = wait_for_terminal(client, started.json()["run_id"])
        assert summary["status"] == RunStatus.COMPLETED.value, summary.get("error")

    def test_multipart_start(self, client: Any, churn_csv: Path) -> None:
        """One-shot form submit: file plus target in a single multipart request."""
        with churn_csv.open("rb") as handle:
            response = client.post(
                "/api/runs",
                files={"file": ("churn.csv", handle, "text/csv")},
                # Form values arrive as strings, so this also checks that the
                # request model coerces "false" and "600" rather than rejecting them.
                data={
                    "target_column": CHURN_TARGET,
                    "max_experiments": "2",
                    "max_rows": "600",
                    "enable_tuning": "false",
                    "enable_explainability": "false",
                    "report_formats": "json",
                },
            )
        assert response.status_code == 202, response.text
        summary = wait_for_terminal(client, response.json()["run_id"])
        assert summary["status"] == RunStatus.COMPLETED.value, summary.get("error")
        assert summary["config"]["enable_tuning"] is False
        assert summary["config"]["max_rows"] == 600


# ===========================================================================
# Reading runs
# ===========================================================================


class TestReadRun:
    def test_summary_is_a_full_run_summary(self, finished_run) -> None:
        _, summary = finished_run
        from automl_architect.core.schemas import RunSummary

        model = RunSummary.model_validate(summary)
        assert model.status is RunStatus.COMPLETED
        assert model.problem is not None
        assert model.experiments is not None

    def test_unknown_run_is_404(self, client: Any) -> None:
        """A 500 here would make the UI show a crash instead of "not found"."""
        assert client.get("/api/runs/run_does_not_exist").status_code == 404

    def test_runs_are_listed(self, finished_run) -> None:
        client, summary = finished_run
        response = client.get("/api/runs", params={"limit": 10})
        assert response.status_code == 200
        body = response.json()
        rows = body.get("runs", body) if isinstance(body, dict) else body
        assert any(row.get("run_id") == summary["run_id"] for row in rows)

    def test_list_limit_is_validated(self, client: Any) -> None:
        assert client.get("/api/runs", params={"limit": 0}).status_code == 422
        assert client.get("/api/runs", params={"limit": 100_000}).status_code == 422

    def test_experiments_are_exposed(self, finished_run) -> None:
        client, summary = finished_run
        response = client.get(f"/api/runs/{summary['run_id']}/experiments")
        if response.status_code == 404:
            pytest.skip("no experiments endpoint")
        assert response.status_code == 200
        body = response.json()
        rows = body.get("experiments", body) if isinstance(body, dict) else body
        assert rows

    def test_projects_are_listed(self, finished_run) -> None:
        client, _ = finished_run
        response = client.get("/api/projects")
        if response.status_code == 404:
            pytest.skip("no projects endpoint")
        assert response.status_code == 200


# ===========================================================================
# Events
# ===========================================================================


class TestEvents:
    def test_events_are_paged_from_a_cursor(self, finished_run) -> None:
        """The cursor is how a reconnecting client avoids replaying the whole log."""
        client, summary = finished_run
        run_id = summary["run_id"]

        first = client.get(f"/api/runs/{run_id}/events")
        assert first.status_code == 200
        body = first.json()
        events = body.get("events", body) if isinstance(body, dict) else body
        assert events, "no events recorded for a completed run"

        sequences = [event["sequence"] for event in events]
        assert sequences == sorted(sequences)

        after = sequences[len(sequences) // 2]
        second = client.get(f"/api/runs/{run_id}/events", params={"after": after})
        assert second.status_code == 200
        tail_body = second.json()
        tail = tail_body.get("events", tail_body) if isinstance(tail_body, dict) else tail_body
        assert all(event["sequence"] > after for event in tail)

    def test_events_for_an_unknown_run_are_404(self, client: Any) -> None:
        assert client.get("/api/runs/run_nope/events").status_code == 404

    def test_stream_emits_server_sent_events(self, finished_run) -> None:
        """A finished run's stream must still replay history and then close."""
        client, summary = finished_run
        run_id = summary["run_id"]
        with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
            if response.status_code == 404:
                pytest.skip("no stream endpoint")
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            body = ""
            for chunk in response.iter_text():
                body += chunk
                if len(body) > 512:
                    break
        assert "data:" in body

    def test_stream_events_carry_json_payloads(self, finished_run) -> None:
        client, summary = finished_run
        with client.stream("GET", f"/api/runs/{summary['run_id']}/stream") as response:
            if response.status_code != 200:
                pytest.skip("no stream endpoint")
            collected = ""
            for chunk in response.iter_text():
                collected += chunk
                if collected.count("data:") >= 2:
                    break
        payloads = [
            line.partition("data:")[2].strip()
            for line in collected.splitlines()
            if line.startswith("data:")
        ]
        assert payloads
        decoded = json.loads(payloads[0])
        assert isinstance(decoded, dict)


# ===========================================================================
# Artifacts
# ===========================================================================


class TestArtifacts:
    def test_report_is_served(self, finished_run) -> None:
        client, summary = finished_run
        response = client.get(f"/api/runs/{summary['run_id']}/report")
        if response.status_code == 404:
            pytest.skip("no report was produced for this run")
        assert response.status_code == 200
        assert response.content

    def test_artifacts_are_listed(self, finished_run) -> None:
        client, summary = finished_run
        response = client.get(f"/api/runs/{summary['run_id']}/artifacts")
        if response.status_code == 404:
            pytest.skip("no artifact listing endpoint")
        assert response.status_code == 200
        body = response.json()
        entries = body.get("artifacts", body) if isinstance(body, dict) else body
        assert isinstance(entries, list)

    def test_a_listed_artifact_can_be_fetched(self, finished_run) -> None:
        client, summary = finished_run
        listing = client.get(f"/api/runs/{summary['run_id']}/artifacts")
        if listing.status_code != 200:
            pytest.skip("no artifact listing endpoint")
        body = listing.json()
        entries = body.get("artifacts", body) if isinstance(body, dict) else body
        if not entries:
            pytest.skip("run produced no artifacts")
        relative = entries[0].get("relative_path") or entries[0].get("path")
        response = client.get(f"/api/runs/{summary['run_id']}/artifacts/{relative}")
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "attack",
        [
            "../../../../etc/passwd",
            "..%2f..%2f..%2fetc%2fpasswd",
            "/etc/passwd",
            "....//....//pyproject.toml",
        ],
    )
    def test_artifact_path_traversal_is_refused(self, finished_run, attack: str) -> None:
        """The artifact route takes a free-form path, which makes this mandatory."""
        client, summary = finished_run
        response = client.get(f"/api/runs/{summary['run_id']}/artifacts/{attack}")
        assert response.status_code in (400, 403, 404), (
            f"path traversal with {attack!r} returned {response.status_code}"
        )
        if response.status_code == 200:  # pragma: no cover - the bug case
            assert b"root:" not in response.content


# ===========================================================================
# Control
# ===========================================================================


class TestControl:
    def test_cancel_an_unknown_run_is_404(self, client: Any) -> None:
        assert client.post("/api/runs/run_nope/cancel").status_code == 404

    def test_cancel_a_finished_run_is_handled(self, finished_run) -> None:
        client, summary = finished_run
        response = client.post(f"/api/runs/{summary['run_id']}/cancel")
        assert response.status_code in (200, 202, 409)

    def test_approvals_are_listed_and_decidable(self, client: Any, churn_csv: Path) -> None:
        """The human-in-the-loop path, over HTTP."""
        run_id = start_run(client, churn_csv, require_approval=True)
        summary = wait_for_terminal(client, run_id)
        if summary["status"] != RunStatus.AWAITING_APPROVAL.value:
            pytest.skip("no destructive step required approval")

        listing = client.get(f"/api/runs/{run_id}/approvals")
        assert listing.status_code == 200
        body = listing.json()
        approvals = body.get("approvals", body) if isinstance(body, dict) else body
        pending = [a for a in approvals if a["decision"] == "pending"]
        assert pending

        decision = client.post(
            f"/api/runs/{run_id}/approvals/{pending[0]['request_id']}",
            json={"decision": "approved", "note": "approved by the test"},
        )
        assert decision.status_code == 200, decision.text
        assert decision.json()["approval"]["decision"] == "approved"
        wait_for_terminal(client, run_id)

    def test_deciding_an_unknown_approval_is_404(self, finished_run) -> None:
        client, summary = finished_run
        response = client.post(
            f"/api/runs/{summary['run_id']}/approvals/apr_nope",
            json={"decision": "approved"},
        )
        assert response.status_code == 404


# ===========================================================================
# Questions
# ===========================================================================


class TestAsk:
    def test_question_is_answered_from_the_run(self, finished_run) -> None:
        client, summary = finished_run
        response = client.post(
            f"/api/runs/{summary['run_id']}/ask",
            json={"question": "Why was this metric chosen?"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["answer"].strip()
        assert payload["confidence"] in ("high", "medium", "low")

    def test_asking_about_an_unknown_run_is_404(self, client: Any) -> None:
        response = client.post("/api/runs/run_nope/ask", json={"question": "what happened?"})
        assert response.status_code == 404

    def test_empty_question_is_rejected(self, finished_run) -> None:
        client, summary = finished_run
        response = client.post(f"/api/runs/{summary['run_id']}/ask", json={"question": ""})
        assert response.status_code in (400, 422)


# ===========================================================================
# Offline guarantee
# ===========================================================================


def test_api_runs_never_touch_the_network(
    client: Any, churn_csv: Path, installed_fake_llm: FakeLLMClient
) -> None:
    """A run driven from a worker thread still resolves to the injected fake.

    The offline guarantee is only as good as its weakest path, and the API's is the
    subtlest: the orchestrator executes on a thread the test never touches, so it
    resolves its client through the process-wide ``get_llm_client()`` rather than
    from an injected argument. This test needs its own run — the shared one was
    created against a previous test's fake.
    """
    run_id = start_run(client, churn_csv, max_experiments=2)
    summary = wait_for_terminal(client, run_id)
    assert summary["status"] == RunStatus.COMPLETED.value, summary.get("error")
    assert installed_fake_llm.n_calls >= 8, (
        "the worker thread produced a completed run without going through the fake"
    )
    assert all(call.system for call in installed_fake_llm.calls)
