"""Credentials must work from the documented dotenv setup without being exported."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from automl_architect.config import Settings
from automl_architect.core.llm import LLMClient


@pytest.fixture
def credential_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.mark.parametrize(
    ("variable", "argument"),
    [("ANTHROPIC_API_KEY", "api_key"), ("ANTHROPIC_AUTH_TOKEN", "auth_token")],
)
def test_dotenv_credential_reaches_sdk(credential_env, monkeypatch, variable, argument):
    (credential_env / ".env").write_text(
        f"{variable}=test-dotenv-credential\nANTHROPIC_BASE_URL=https://gateway.example.test\n",
        encoding="utf-8",
    )
    constructor = Mock()
    monkeypatch.setattr("automl_architect.core.llm.anthropic.Anthropic", constructor)
    settings = Settings(offline=False)
    assert settings.has_api_key()
    assert LLMClient(settings)._client is constructor.return_value
    assert constructor.call_args.kwargs[argument] == "test-dotenv-credential"
    assert constructor.call_args.kwargs["base_url"] == "https://gateway.example.test"
    assert "test-dotenv-credential" not in repr(settings)
    assert "test-dotenv-credential" not in settings.model_dump_json()
    assert "anthropic_api_key" not in settings.model_dump()
    assert "anthropic_auth_token" not in settings.model_dump()


def test_environment_wins_over_local_dotenv_then_dotenv(credential_env, monkeypatch):
    (credential_env / ".env").write_text("ANTHROPIC_API_KEY=base-key\n", encoding="utf-8")
    (credential_env / ".env.local").write_text("ANTHROPIC_API_KEY=local-key\n", encoding="utf-8")
    assert Settings().anthropic_api_key.get_secret_value() == "local-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")
    assert Settings().anthropic_api_key.get_secret_value() == "environment-key"


def test_blank_credentials_are_not_reported_as_ready(credential_env):
    (credential_env / ".env").write_text(
        "ANTHROPIC_API_KEY=\nANTHROPIC_AUTH_TOKEN=\n", encoding="utf-8"
    )
    assert not Settings().has_api_key()


@pytest.mark.parametrize("cap", [0, -1])
def test_upload_limit_must_be_positive(cap):
    with pytest.raises(ValidationError, match="max_upload_bytes"):
        Settings(max_upload_bytes=cap)
