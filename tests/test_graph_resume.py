"""Durable SQLite resume: a worker that completed before a crash must NOT
re-run when the graph is re-invoked from the on-disk checkpoint; the unfinished
worker must."""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from iag_sim.models import EnvName, TradeTask, WorkerResult
from iag_sim.orchestration import graph as graphmod
from iag_sim.orchestration.resources import Resources


def _fake_resources(tmp_path: Path) -> Resources:
    return Resources(
        settings=None, backend=None, run_dir=tmp_path, semaphores={}, harnesses={}
    )


async def test_sqlite_resume_skips_completed_workers(tmp_path, monkeypatch):
    calls: list[tuple[str, str]] = []
    fail_after = {"on": True}

    async def fake_run_worker(res, trade, env):
        calls.append((trade.trade_id, env.value))
        if env.value == "after" and fail_after["on"]:
            raise RuntimeError("simulated crash mid-run")
        return WorkerResult(
            trade_id=trade.trade_id, env=env, ok=True, csv_path=f"{env.value}.csv"
        )

    # Stub the worker + postprocess so the graph is pure/deterministic.
    monkeypatch.setattr(graphmod, "run_worker", fake_run_worker)
    monkeypatch.setattr(
        graphmod, "postprocess",
        lambda results, settings, run_dir: {"n_results": len(results)},
    )

    res = _fake_resources(tmp_path)
    trades = [TradeTask(trade_id="594")]
    db = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "run-x"}}

    # First pass: "after" crashes; the whole invoke raises.
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        g = graphmod.build_graph(res, None, tmp_path, checkpointer=saver)
        with pytest.raises(Exception):
            await g.ainvoke(
                {"trades": [t.model_dump() for t in trades], "results": []},
                config=config,
            )
    assert ("594", "before") in calls
    assert ("594", "after") in calls

    # Resume from the durable checkpoint: "before" already done -> not re-run.
    calls.clear()
    fail_after["on"] = False
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        g = graphmod.build_graph(res, None, tmp_path, checkpointer=saver)
        final = await g.ainvoke(None, config=config)

    assert ("594", "before") not in calls, "completed worker should replay, not re-run"
    assert ("594", "after") in calls, "unfinished worker must re-run"
    assert final["summary"] == {"n_results": 2}


async def test_db_persists_on_disk(tmp_path, monkeypatch):
    async def fake_run_worker(res, trade, env):
        return WorkerResult(trade_id=trade.trade_id, env=env, ok=True, csv_path="x.csv")

    monkeypatch.setattr(graphmod, "run_worker", fake_run_worker)
    monkeypatch.setattr(graphmod, "postprocess", lambda r, s, d: {"ok": True})

    db = tmp_path / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        g = graphmod.build_graph(_fake_resources(tmp_path), None, tmp_path, checkpointer=saver)
        await g.ainvoke(
            {"trades": [TradeTask(trade_id="1").model_dump()], "results": []},
            config={"configurable": {"thread_id": "t"}},
        )
    assert db.exists() and db.stat().st_size > 0
