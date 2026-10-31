"""Upload regressions exercise HTTP handling without training a model."""

import asyncio

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from automl_architect.api import routes
from automl_architect.config import Settings


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    settings = Settings(workspace=tmp_path, max_upload_bytes=1024, _env_file=None)
    monkeypatch.setattr(routes, "_settings", lambda: settings)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client, settings


@pytest.mark.parametrize("size, status", [(1024, 200), (1025, 413), (0, 422)])
def test_raw_upload_boundary(upload_client, size, status):
    client, settings = upload_client
    response = client.post(
        "/api/upload?filename=data.csv",
        content=b"x" * size,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["size_bytes"] == size
    else:
        assert not list(settings.workspace.rglob("*.csv"))


@pytest.mark.parametrize("native", [False, True], ids=["fallback", "starlette"])
@pytest.mark.parametrize("endpoint", ["/api/upload", "/api/runs"])
def test_multipart_oversize_never_starts_or_stores(upload_client, monkeypatch, native, endpoint):
    if native:
        pytest.importorskip("python_multipart")
    monkeypatch.setattr(routes, "_starlette_can_parse_multipart", lambda: native)
    client, settings = upload_client
    response = client.post(endpoint, files={"file": ("data.csv", b"x" * 1025, "text/csv")})
    assert response.status_code == 413, response.text
    assert not list(settings.workspace.rglob("*.csv"))


@pytest.mark.parametrize("native", [False, True], ids=["fallback", "starlette"])
def test_multipart_preserves_binary_bytes(upload_client, monkeypatch, native):
    if native:
        pytest.importorskip("python_multipart")
    monkeypatch.setattr(routes, "_starlette_can_parse_multipart", lambda: native)
    client, _settings = upload_client
    payload = b"\x00\xff\x80\r\nvalue\r\n"
    response = client.post("/api/upload", files={"file": ("data.parquet", payload)})
    assert response.status_code == 200, response.text
    from pathlib import Path

    assert Path(response.json()["path"]).read_bytes() == payload


@pytest.mark.parametrize("native", [False, True], ids=["fallback", "starlette"])
def test_multiple_files_are_rejected(upload_client, monkeypatch, native):
    if native:
        pytest.importorskip("python_multipart")
    monkeypatch.setattr(routes, "_starlette_can_parse_multipart", lambda: native)
    client, _settings = upload_client
    response = client.post(
        "/api/upload",
        files=[("file", ("one.csv", b"a")), ("file", ("two.csv", b"b"))],
    )
    assert response.status_code in (400, 422), response.text


def test_chunked_upload_stops_reading_at_limit():
    chunks = iter([b"1234", b"5678", b"must-not-be-read"])
    received = []

    async def receive():
        chunk = next(chunks)
        received.append(chunk)
        return {"type": "http.request", "body": chunk, "more_body": True}

    request = Request({"type": "http", "headers": []}, receive)
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes._read_capped_body(request, cap=6))
    assert error.value.status_code == 413
    assert received == [b"1234", b"5678"]
