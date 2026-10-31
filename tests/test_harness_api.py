from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from automl_architect.api.app import create_app
from automl_architect.config import Settings
from automl_architect.core.llm import LLMClient


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_byok_is_memory_only_and_revocable(tmp_path, monkeypatch, provider):
    settings = Settings(workspace=tmp_path, offline=True)
    seen = []

    def probe(self, **kwargs):
        seen.append(self.settings)
        return SimpleNamespace(value=SimpleNamespace(status="ok"))

    monkeypatch.setattr(LLMClient, "structured", probe)
    app = create_app(settings)
    client = TestClient(app)
    secret = "sk-test-do-not-persist"
    response = client.post(
        "/api/connections", json={"provider": provider, "model": "test-model", "api_key": secret}
    )
    assert response.status_code == 200, response.text
    assert secret not in response.text
    assert len(seen) == 1
    assert seen[0].enable_thinking is (provider == "anthropic")
    identifier = response.json()["connection_id"]
    stored = app.state.connections.get(identifier)
    assert getattr(stored, f"{provider}_api_key").get_secret_value() == secret
    assert secret not in stored.model_dump_json()
    assert stored.offline is False
    assert not list(tmp_path.rglob("*.env"))
    assert client.delete(f"/api/connections/{identifier}").status_code == 200
    with pytest.raises(ValueError, match="expired"):
        app.state.connections.get(identifier)


def test_invalid_connection_request_never_echoes_key(tmp_path):
    client = TestClient(create_app(Settings(workspace=tmp_path, offline=True)))
    response = client.post(
        "/api/connections", json={"provider": "wrong", "model": "x", "api_key": "secret-value"}
    )
    assert response.status_code == 422
    assert "secret-value" not in response.text


def test_server_token_and_browser_session(tmp_path):
    client = TestClient(
        create_app(Settings(workspace=tmp_path, offline=True, api_token="owner-token"))
    )
    assert client.get("/api/providers").status_code == 401
    assert (
        client.get("/api/providers", headers={"Authorization": "Bearer owner-token"}).status_code
        == 200
    )
    assert client.post("/api/session", json={"token": "wrong"}).status_code == 401
    login = client.post("/api/session", json={"token": "owner-token"})
    assert login.status_code == 200
    assert "HttpOnly" in login.headers["set-cookie"]
    assert "owner-token" not in login.headers["set-cookie"]
    assert client.get("/api/providers").status_code == 200
    assert (
        client.post(
            "/api/connections", json={}, headers={"origin": "https://attacker.example"}
        ).status_code
        == 403
    )


def test_unconfigured_remote_server_rejects_requests(tmp_path):
    client = TestClient(create_app(Settings(workspace=tmp_path)), client=("203.0.113.9", 50000))
    assert client.get("/api/providers").status_code == 403


def test_local_server_rejects_dns_rebinding_host(tmp_path):
    client = TestClient(create_app(Settings(workspace=tmp_path)))
    assert client.get("/api/providers", headers={"host": "attacker.example"}).status_code == 403


@pytest.mark.parametrize("provider, model", [("openai", "gpt-example"), ("anthropic", "claude-example")])
def test_provider_catalog_preserves_custom_defaults(tmp_path, provider, model):
    client = TestClient(create_app(Settings(workspace=tmp_path, provider=provider, model=model)))
    catalog = client.get("/api/providers").json()["providers"]
    defaults = {p["id"]: p["default_model"] for p in catalog}
    assert defaults[provider] == model
    for entry in catalog:
        prefix = "claude-" if entry["id"] == "anthropic" else "gpt-"
        ids = [option["id"] for option in entry["models"]]
        assert entry["default_model"] in ids
        assert len(ids) == len(set(ids))
        assert all(identifier.startswith(prefix) for identifier in ids)
        assert all(option["label"] for option in entry["models"])
