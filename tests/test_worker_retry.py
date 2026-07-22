"""`run_worker` always returns a WorkerResult — retry exhaustion is data, not an
exception.

Tenacity raises `RetryError` once `stop_after_attempt` is reached with a still-
failing RESULT, and its message is the opaque
`RetryError[<Future ... returned WorkerResult>]`. Letting that escape would abort
the whole graph run and hand the API that string as the run error, hiding the real
per-trade cause. These tests pin the unwrap.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from tenacity import wait_none

from iag_sim.models import EnvName, WorkerResult, TradeTask
from iag_sim.orchestration import worker as worker_mod


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Kill the exponential wait — these tests exercise control flow, not timing."""
    monkeypatch.setattr(worker_mod._simulate_with_retry.retry, "wait", wait_none())


@pytest.fixture
def res():
    """Minimal stand-in for Resources: only the semaphore is touched (the fake
    simulate_trade replaces everything the harness/backend would do)."""
    sem = asyncio.Semaphore(1)
    return SimpleNamespace(
        semaphore_for=lambda env: sem,
        harness_for=lambda env: None,
        settings=None,
        backend=None,
        run_dir=None,
    )


@pytest.fixture
def trade():
    return TradeTask(trade_id="1472107")


def _result(**kw) -> WorkerResult:
    return WorkerResult(trade_id="1472107", env=EnvName.BEFORE, **kw)


@pytest.mark.asyncio
async def test_exhausted_retries_return_last_result(res, trade, monkeypatch):
    calls = 0

    async def always_fails(**kwargs):
        nonlocal calls
        calls += 1
        return _result(ok=False, error=f"export gate failed (attempt {calls})")

    monkeypatch.setattr(worker_mod, "simulate_trade", always_fails)

    result = await worker_mod.run_worker(res, trade, EnvName.BEFORE)

    assert calls == 3  # stop_after_attempt(3)
    assert result.ok is False
    assert result.error == "export gate failed (attempt 3)"
    assert "RetryError" not in (result.error or "")


@pytest.mark.asyncio
async def test_success_after_a_failure_is_returned_unchanged(res, trade, monkeypatch):
    calls = 0

    async def flaky(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _result(ok=False, error="transient")
        return _result(ok=True, csv_path="/tmp/x.csv")

    monkeypatch.setattr(worker_mod, "simulate_trade", flaky)

    result = await worker_mod.run_worker(res, trade, EnvName.BEFORE)
    assert result.ok is True
    assert result.csv_path == "/tmp/x.csv"


@pytest.mark.asyncio
async def test_raised_exception_still_propagates(res, trade, monkeypatch):
    """Only automation FAILURES are swallowed. A genuine crash (a bug, a dead
    Docker daemon) must keep bubbling — it is not a per-trade result."""

    async def crashes(**kwargs):
        raise RuntimeError("docker daemon unreachable")

    monkeypatch.setattr(worker_mod, "simulate_trade", crashes)

    with pytest.raises(RuntimeError, match="docker daemon unreachable"):
        await worker_mod.run_worker(res, trade, EnvName.BEFORE)
