"""The documented keyless setup must work through the actual CLI parser."""

from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from automl_architect.cli import app
from automl_architect.config import get_settings, reset_settings_cache


@pytest.mark.parametrize("command", ["serve", "doctor"])
def test_offline_cli_does_not_require_credentials(command, monkeypatch):
    monkeypatch.setenv("AUTOML_OFFLINE", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    reset_settings_cache()
    uvicorn_run = Mock()
    if command == "serve":
        pytest.importorskip("uvicorn")
        monkeypatch.setattr("uvicorn.run", uvicorn_run)
    result = CliRunner().invoke(app, [command, "--offline"])
    assert result.exit_code == 0, result.output
    assert get_settings().offline
    if command == "serve":
        uvicorn_run.assert_called_once()
        assert uvicorn_run.call_args.args[0] == "automl_architect.api.app:app"
    else:
        assert "not required" in result.output
