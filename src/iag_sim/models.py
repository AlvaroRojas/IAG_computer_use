"""Shared domain models. Deliberately small and JSON-serializable so they can
flow through async results and (optionally) a LangGraph checkpointed state."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class EnvName(str, Enum):
    BEFORE = "before"
    AFTER = "after"


class TradeTask(BaseModel):
    """One trade to simulate. `extra` carries any additional identifiers the
    Murex screen needs (book, portfolio, value date, ...)."""

    trade_id: str
    extra: dict[str, str] = {}


class WorkerResult(BaseModel):
    """Outcome of simulating one trade in one environment. Fully serializable."""

    trade_id: str
    env: EnvName
    csv_path: str | None = None
    ok: bool = False
    # A trusted zero-posting result: the simulation ran and exported a header-only
    # CSV (0 data rows). ok=True, but the coverage ledger tracks it apart from
    # postings-bearing results so empty-both can be proven a MATCH.
    empty: bool = False
    error: str | None = None
    turns: int = 0
