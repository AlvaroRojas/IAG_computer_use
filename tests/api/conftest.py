"""Fixtures for the API tests.

Deterministic: no browser, Docker, or network. `run_graph_async` is monkeypatched
with a fake coroutine, so the manager exercises its full lifecycle without the
engine. cwd is moved to `tmp_path` so the repo's real `.env` is NOT loaded by the
per-request `Settings` (the root autouse fixture already strips alias env vars);
the provider therefore defaults to openai + the dummy OPENAI_API_KEY set here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iag_sim.api.app import create_app

VALID_KEY = "test-key-123"


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path


@pytest.fixture
def app(output_dir, monkeypatch):
    monkeypatch.chdir(output_dir)  # no .env here -> Settings stays deterministic
    monkeypatch.setenv("IAG_SIM_API_KEY", VALID_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # default provider cred
    return create_app(output_dir=output_dir)


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def auth():
    return {"X-API-Key": VALID_KEY}


@pytest.fixture
def payload():
    """A valid new-run body (UPPERCASE keys, as the caller sends)."""
    return {
        "run_id": "",
        "MUREX_BEFORE_URL": "https://before.example",
        "MUREX_AFTER_URL": "https://after.example",
        "MUREX_USER": "svc",
        "MUREX_PASS": "secret",
        "MUREX_LOGIN_GROUP": "GRP",
        "MUREX_CHANNEL": "web",
        "MAX_CONCURRENCY": 1,
        "trades": [{"trade_id": "1472107"}],
    }


def _match_summary(run_dir, trades) -> dict:
    return {
        "run_dir": str(run_dir),
        "before_csv": None,
        "after_csv": None,
        "trades_ok_before": len(trades),
        "trades_ok_after": len(trades),
        "failures": [],
        "comparison": None,
        "diff": {
            "matches": True,
            "mismatched_rows": 0,
            "rows_only_before": 0,
            "rows_only_after": 0,
        },
    }


@pytest.fixture
def recorder(monkeypatch):
    """Patch run_graph_async with a fast fake that records its calls and returns a
    MATCH summary. Returns the list of recorded calls."""
    calls: list[SimpleNamespace] = []

    async def fake(trades, settings, run_dir, *, thread_id=None, resume=False):
        calls.append(
            SimpleNamespace(
                trades=trades, settings=settings, run_dir=run_dir, resume=resume
            )
        )
        return _match_summary(run_dir, trades)

    import iag_sim.api.run_manager as rm

    monkeypatch.setattr(rm, "run_graph_async", fake)
    return calls


@pytest.fixture
def slow_engine(monkeypatch):
    """Patch run_graph_async with a fake that stays RUNNING long enough to test
    the one-at-a-time 409 (cancelled when the client closes)."""
    import asyncio

    async def fake(trades, settings, run_dir, *, thread_id=None, resume=False):
        await asyncio.sleep(30)
        return _match_summary(run_dir, trades)

    import iag_sim.api.run_manager as rm

    monkeypatch.setattr(rm, "run_graph_async", fake)
