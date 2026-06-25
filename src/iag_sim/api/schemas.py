"""Request / response models for the run API.

The request fields mirror the `Settings` aliases verbatim (UPPERCASE `MUREX_*`),
so the body maps 1:1 onto a per-request `Settings(**alias_kwargs)` with no manual
renaming. `populate_by_name=True` also accepts the snake_case attribute names.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    """Lifecycle of a single run. No QUEUED — runs start immediately or are
    rejected (one-at-a-time)."""

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class ResultCode(str, Enum):
    """Outcome of a SUCCEEDED run — mirrors the CLI exit codes 0/2/1."""

    MATCH = "MATCH"  # before/after identical (CLI exit 0)
    DIFFERENCES = "DIFFERENCES"  # accounting differences found (CLI exit 2)
    NO_COMPARISON = "NO_COMPARISON"  # one/both envs yielded no data (CLI exit 1)


class TradePayload(BaseModel):
    """One trade to simulate. `extra` carries any additional identifier columns
    the Murex screen needs (book, portfolio, value date, ...)."""

    trade_id: str
    extra: dict[str, str] = Field(default_factory=dict)


class RunRequest(BaseModel):
    """Body of POST /runs. Empty `run_id` starts a new run; a non-empty `run_id`
    resumes that run-dir folder name from its on-disk checkpoint."""

    model_config = ConfigDict(populate_by_name=True)

    run_id: str = ""

    # Murex connection — aliases match the Settings field aliases exactly.
    murex_before_url: str = Field(alias="MUREX_BEFORE_URL")
    murex_after_url: str = Field(alias="MUREX_AFTER_URL")
    murex_user: str = Field(alias="MUREX_USER")
    murex_pass: str = Field(alias="MUREX_PASS")

    murex_login_group: str = Field(default="", alias="MUREX_LOGIN_GROUP")
    murex_before_group: str | None = Field(default=None, alias="MUREX_BEFORE_GROUP")
    murex_after_group: str | None = Field(default=None, alias="MUREX_AFTER_GROUP")

    murex_channel: str = Field(default="web", alias="MUREX_CHANNEL")
    murex_before_channel: str | None = Field(default=None, alias="MUREX_BEFORE_CHANNEL")
    murex_after_channel: str | None = Field(default=None, alias="MUREX_AFTER_CHANNEL")

    max_concurrency: int = Field(default=4, alias="MAX_CONCURRENCY", ge=1, le=64)

    trades: list[TradePayload] = Field(default_factory=list)


class StartRunResponse(BaseModel):
    """202 body returned the moment a run is accepted."""

    run_id: str
    status: RunStatus
    message: str


class RunStatusResponse(BaseModel):
    """GET /runs/{run_id} body. `summary` is the raw dict from `run_graph_async`
    (run_dir, before_csv, after_csv, trades_ok_*, failures, comparison, diff) with
    run_id + total_execution_seconds stamped on, matching the CLI's final JSON."""

    run_id: str
    status: RunStatus
    result_code: ResultCode | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class RunListItem(BaseModel):
    run_id: str
    status: RunStatus
    result_code: ResultCode | None = None
    started_at: str | None = None
    finished_at: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
