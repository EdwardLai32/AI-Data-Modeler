"""Ephemeral BYOK connections. Keys never enter RunConfig or persisted state."""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ..core.llm import LLMClient

router = APIRouter(prefix="/api", tags=["AI connections"])

# Curated text models for the dashboard. A connection test verifies account access.
# Sources: platform.claude.com/docs/en/models/overview and
# developers.openai.com/api/docs/models (reviewed 2026-10-30).
MODEL_OPTIONS = {
    "anthropic": (
        ("claude-opus-5-5", "Claude Opus 5.5"),
        ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
        ("claude-haiku-5-5", "Claude Haiku 5.5"),
        ("claude-opus-5", "Claude Opus 5"),
    ),
    "openai": (
        ("gpt-6-astra", "GPT-6 Astra"),
        ("gpt-6.1-sol", "GPT-6.1 Sol"),
        ("gpt-6.1-terra", "GPT-6.1 Terra"),
        ("gpt-6-luna", "GPT-6 Luna"),
        ("gpt-4.1-mini", "GPT-4.1 mini"),
    ),
}


class ConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["anthropic", "openai"]
    model: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._:/-]+$")
    api_key: SecretStr = Field(exclude=True, repr=False)
    test: bool = True
    input_price_per_million: float | None = Field(default=None, ge=0, le=10000)
    output_price_per_million: float | None = Field(default=None, ge=0, le=10000)


class ConnectionStore:
    def __init__(self, settings):
        self.settings = settings
        self._items = {}
        self._lock = threading.Lock()

    def build_settings(self, request):
        if not request.api_key.get_secret_value().strip():
            raise ValueError("An API key is required.")
        return self.settings.model_copy(
            update={
                "provider": request.provider,
                "model": request.model,
                "offline": False,
                "anthropic_api_key": request.api_key if request.provider == "anthropic" else None,
                "anthropic_auth_token": None,
                "anthropic_base_url": None,
                "openai_api_key": request.api_key if request.provider == "openai" else None,
                "input_price_per_million": request.input_price_per_million,
                "output_price_per_million": request.output_price_per_million,
            }
        )

    def add(self, settings):
        with self._lock:
            self._items = {k: v for k, v in self._items.items() if v[0] > time.time()}
            if len(self._items) >= 32:
                raise ValueError("Too many active connections; disconnect an unused connection.")
            identifier = secrets.token_urlsafe(32)
            self._items[identifier] = (time.time() + 3600, settings)
            return identifier

    def get(self, identifier):
        with self._lock:
            item = self._items.get(identifier)
            if item is None or item[0] <= time.time():
                self._items.pop(identifier, None)
                raise ValueError("Connection expired. Connect your provider again.")
            return item[1]

    def delete(self, identifier):
        with self._lock:
            self._items.pop(identifier, None)


class ConnectionProbe(BaseModel):
    status: Literal["ok"]


@router.get("/providers")
def providers(request: Request):
    settings = request.app.state.settings
    catalog = []
    for provider, options in MODEL_OPTIONS.items():
        default_model = settings.model_for_provider(provider)
        models = [{"id": model, "label": label} for model, label in options]
        if default_model not in {model["id"] for model in models}:
            models.insert(0, {"id": default_model, "label": default_model})
        catalog.append({"id": provider, "default_model": default_model, "models": models})
    return {
        "providers": catalog,
        "offline": settings.offline,
        "configured_provider": settings.provider,
        "credentials_configured": settings.has_api_key(),
    }


@router.post("/connections")
async def connect(payload: ConnectionRequest, request: Request):
    store = request.app.state.connections
    try:
        settings = store.build_settings(payload)
        if payload.test:
            probe = settings.model_copy(
                update={
                    # Current Claude models require adaptive thinking, even for probes.
                    "enable_thinking": settings.provider == "anthropic",
                    "default_effort": "low",
                    "enable_refusal_fallback": False,
                    "llm_timeout_seconds": 30.0,
                    "llm_max_retries": 0,
                }
            )
            await asyncio.to_thread(
                LLMClient(probe).structured,
                output_model=ConnectionProbe,
                agent_instructions="This is a connection check. Return status ok.",
                user="Confirm the connection.",
                max_tokens=1024,
                max_attempts=1,
            )
        identifier = store.add(settings)
    except Exception as exc:
        # Provider exceptions can contain request details. Never echo them.
        raise HTTPException(
            400,
            f"Connection failed ({type(exc).__name__}). Check your key, model access, and provider availability.",
        ) from None
    return {
        "connection_id": identifier,
        "provider": settings.provider,
        "model": settings.model,
        "tested": payload.test,
        "expires_in_seconds": 3600,
    }


@router.delete("/connections/{connection_id}")
def disconnect(connection_id: str, request: Request):
    request.app.state.connections.delete(connection_id)
    return {"disconnected": True}


def settings_for_connection(request, connection_id, provider=None, model=None):
    if not connection_id:
        return None
    try:
        settings = request.app.state.connections.get(connection_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if (provider and provider != settings.provider) or (model and model != settings.model):
        raise HTTPException(400, "Selected provider/model does not match this connection.")
    return settings


class RecoveryRequest(BaseModel):
    connection_id: str | None = Field(default=None, exclude=True, repr=False)


@router.post("/runs/{run_id}/recover", status_code=202)
async def recover(run_id: str, payload: RecoveryRequest, request: Request):
    from .routes import get_manager

    manager = get_manager()
    summary = await asyncio.to_thread(manager.summary, run_id)
    if summary is None:
        raise HTTPException(404, "Unknown run.")
    settings = settings_for_connection(
        request, payload.connection_id, summary.config.provider, summary.config.model
    )
    try:
        handle = await asyncio.to_thread(manager.recover, run_id, settings=settings)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"run_id": handle.run_id, "status": handle.status.value}
