"""Process-wide settings, loaded from environment and ``.env``.

Nothing here is dataset- or run-specific — per-run choices live on
:class:`~automl_architect.core.schemas.RunConfig`. This module answers "how is
this deployment wired up", not "what should this run do".
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The model id is fixed rather than configurable-by-default on purpose: every
# agent prompt in this package is written and tuned against Opus 5's behaviour
# (adaptive thinking on by default, effort ladder through `max`).
DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens for claude-opus-5.
PRICE_INPUT_PER_MTOK = 5.00
PRICE_OUTPUT_PER_MTOK = 25.00
PRICE_CACHE_WRITE_PER_MTOK = 6.25  # 1.25x input
PRICE_CACHE_READ_PER_MTOK = 0.50  # 0.10x input


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_prefix="AUTOML_",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Explicit aliases also load SDK settings from .env/.env.local. Pydantic
    # does not export dotenv entries into os.environ for the SDK to discover.
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY", exclude=True, repr=False
    )
    anthropic_auth_token: SecretStr | None = Field(
        default=None, validation_alias="ANTHROPIC_AUTH_TOKEN", exclude=True, repr=False
    )
    anthropic_base_url: str | None = Field(
        default=None, validation_alias="ANTHROPIC_BASE_URL", exclude=True, repr=False
    )

    # --- model -----------------------------------------------------------
    provider: Literal["anthropic", "openai"] = "anthropic"
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY", exclude=True, repr=False
    )
    openai_model: str = "gpt-4.1-mini"
    # Prices are operator-supplied estimates, never silently borrowed from a
    # different model. A dollar cap requires both rates to be configured.
    input_price_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_price_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    api_token: SecretStr | None = Field(default=None, exclude=True, repr=False)
    allow_remote_sources: bool = False
    enable_checkpoints: bool = True
    isolate_training: bool = True
    include_sample_values_in_prompts: bool = False
    model: str = Field(default=DEFAULT_MODEL, description="Selected provider's model ID.")
    default_effort: str = Field(
        default="high",
        description="Reasoning effort when an agent does not override it. "
        "One of low|medium|high|xhigh|max.",
    )
    max_output_tokens: int = 16000
    llm_timeout_seconds: float = 600.0
    llm_max_retries: int = 3
    enable_prompt_caching: bool = True
    cache_ttl: str = Field(
        default="1h",
        description=(
            "TTL for cached prompt prefixes: '5m' or '1h'. Defaults to 1h because "
            "agent steps in a single run take 60-400s each, so a 5-minute entry "
            "expires before the next agent reuses it — every call then pays the "
            "cache-write premium and reads nothing back."
        ),
    )
    enable_refusal_fallback: bool = Field(
        default=False,
        description="Send fallbacks='default' so a policy decline is re-served "
        "by Anthropic's recommended fallback model instead of failing the run.",
    )
    enable_thinking: bool = True
    offline: bool = Field(
        default=False,
        description=(
            "Run with zero LLM calls and zero credentials. Every agent falls back "
            "to a deterministic rule engine that derives its decisions — and its "
            "rationales — from the measured profile. Profiling, training, tuning, "
            "SHAP, charts, and reports are unaffected: they were always "
            "deterministic. Set via AUTOML_OFFLINE=1 or `amla run --offline`."
        ),
    )

    # --- storage ---------------------------------------------------------
    workspace: Path = Field(
        default=Path("./workspace"),
        description="Root for artifacts, models, charts, and reports.",
    )
    database_url: str = Field(
        default="",
        description="SQLAlchemy URL. Defaults to SQLite under the workspace.",
    )

    # --- execution budgets ----------------------------------------------
    max_profile_rows: int = Field(
        default=250_000,
        description="Row cap for expensive profiling passes (correlations, outliers). "
        "The full row count is always reported; only the statistics sample.",
    )
    max_train_rows: int = Field(
        default=500_000, description="Row cap before training subsamples."
    )
    n_jobs: int = -1
    default_random_state: int = 42

    # --- api -------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    max_upload_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1,
        description="Maximum uploaded dataset size in bytes. The fallback form "
        "parser also counts multipart encoding overhead against this limit.",
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- observability ---------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False
    mlflow_tracking_uri: str = Field(
        default="", description="Optional. Empty disables MLflow logging."
    )

    @field_validator("cache_ttl")
    @classmethod
    def _check_ttl(cls, value: str) -> str:
        allowed = {"5m", "1h"}
        low = value.lower().strip()
        if low not in allowed:
            raise ValueError(f"cache_ttl must be one of {sorted(allowed)}")
        return low

    @field_validator("default_effort")
    @classmethod
    def _check_effort(cls, value: str) -> str:
        allowed = {"low", "medium", "high", "xhigh", "max"}
        low = value.lower()
        if low not in allowed:
            raise ValueError(f"default_effort must be one of {sorted(allowed)}")
        return low

    @field_validator("workspace")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return Path(os.path.expandvars(str(value))).expanduser()

    # --- derived ---------------------------------------------------------

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.workspace.mkdir(parents=True, exist_ok=True)
        db_path = (self.workspace / "automl_architect.db").resolve()
        return f"sqlite+pysqlite:///{db_path.as_posix()}"

    @property
    def runs_dir(self) -> Path:
        return self.workspace / "runs"

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_for_provider(self, provider: Literal["anthropic", "openai"] | None = None) -> str:
        """Resolve a provider default while preserving its configured model."""
        selected = provider or self.provider
        if selected == "openai":
            if self.provider == "openai" and self.model != DEFAULT_MODEL:
                return self.model
            return self.openai_model
        return self.model if self.provider == "anthropic" else DEFAULT_MODEL

    def has_api_key(self) -> bool:
        """Whether *some* Anthropic credential is resolvable.

        Credentials can come from the process environment or dotenv files.
        SDK-managed credential sources are not checked here.
        """
        if self.provider == "openai":
            return bool(self.openai_api_key and self.openai_api_key.get_secret_value())
        return bool(
            (self.anthropic_api_key and self.anthropic_api_key.get_secret_value())
            or (self.anthropic_auth_token and self.anthropic_auth_token.get_secret_value())
        )

    def ensure_dirs(self) -> None:
        for path in (self.workspace, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached singleton. Used by tests that patch the environment."""
    get_settings.cache_clear()


def estimate_cost_usd(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Cost of one call in USD, at claude-opus-5 list prices."""
    return (
        input_tokens * PRICE_INPUT_PER_MTOK
        + output_tokens * PRICE_OUTPUT_PER_MTOK
        + cache_write_tokens * PRICE_CACHE_WRITE_PER_MTOK
        + cache_read_tokens * PRICE_CACHE_READ_PER_MTOK
    ) / 1_000_000
