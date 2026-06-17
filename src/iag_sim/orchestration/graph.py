"""Alternative engine: LangGraph StateGraph with Send-API fan-out.

orchestrator --Send(worker) per (trade x env)--> [parallel workers] --> aggregate

Pick this over the async runner when you want **durable checkpoint/resume**: a
crashed (or killed) run re-invoked with the same run dir replays the workers that
already completed — from the on-disk SQLite checkpoint — instead of re-running
them. Workers still in flight when the process died re-run from scratch.

Checkpointer:
  - Production: `AsyncSqliteSaver` over `<run_dir>/checkpoints.sqlite` — durable,
    survives process exit, so resume works across separate `iag-sim` invocations.
  - `build_graph` defaults to an in-memory `MemorySaver` when no checkpointer is
    passed (used by unit tests; gives no cross-process resume).

On resume the runtime resources (Docker containers / Playwright contexts) and the
Murex login are always re-established — only completed *worker results* replay.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

try:  # Send moved across versions
    from langgraph.types import Send
except ImportError:  # pragma: no cover
    from langgraph.constants import Send  # type: ignore

from ..config import Settings
from ..models import EnvName, TradeTask, WorkerResult
from .postprocess import postprocess
from .resources import Resources, open_resources
from .state import GraphState
from .worker import run_worker

# Checkpoint DB filename inside each run dir. Stable so a resume of the same run
# dir reattaches to the same thread history.
CHECKPOINT_DB = "checkpoints.sqlite"


def build_graph(res: Resources, settings: Settings, run_dir: Path, checkpointer=None):
    """Compile the fan-out graph. `checkpointer` defaults to an in-memory saver
    (tests); pass an `AsyncSqliteSaver` for durable resume."""

    def fan_out(state: GraphState) -> list[Send]:
        return [
            Send("worker", {"trade": trade, "env": env.value})
            for trade in state["trades"]
            for env in (EnvName.BEFORE, EnvName.AFTER)
        ]

    async def worker(payload: dict) -> dict:
        trade = TradeTask(**payload["trade"])
        env = EnvName(payload["env"])
        result = await run_worker(res, trade, env)
        return {"results": [result.model_dump(mode="json")]}

    def aggregate(state: GraphState) -> dict:
        results = [WorkerResult(**r) for r in state["results"]]
        return {"summary": postprocess(results, settings, run_dir)}

    builder = StateGraph(GraphState)
    builder.add_node("worker", worker)
    builder.add_node("aggregate", aggregate)
    builder.add_conditional_edges(START, fan_out, ["worker"])
    builder.add_edge("worker", "aggregate")
    builder.add_edge("aggregate", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())


@asynccontextmanager
async def _sqlite_saver(run_dir: Path):
    """Durable SQLite checkpointer over `<run_dir>/checkpoints.sqlite`. Imported
    lazily so the async engine works even if the sqlite extra is absent."""
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "langgraph engine needs 'langgraph-checkpoint-sqlite'. "
            "Install it: pip install langgraph-checkpoint-sqlite"
        ) from exc

    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / CHECKPOINT_DB
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver


async def run_graph_async(
    trades: list[TradeTask],
    settings: Settings,
    run_dir: Path,
    *,
    thread_id: str | None = None,
    resume: bool = False,
) -> dict:
    """Run (or resume) the before/after comparison via the LangGraph engine.

    thread_id defaults to the run dir name — one run dir == one resumable thread.
    On `resume=True` the graph is re-invoked with no input, replaying completed
    workers from the SQLite checkpoint; only unfinished work runs again.
    """
    thread_id = thread_id or run_dir.name
    config = {"configurable": {"thread_id": thread_id}}

    async with _sqlite_saver(run_dir) as saver:
        async with open_resources(settings, run_dir) as res:
            graph = build_graph(res, settings, run_dir, checkpointer=saver)
            inp = None if resume else {
                "trades": [t.model_dump() for t in trades],
                "results": [],
            }
            final = await graph.ainvoke(inp, config=config)
    return final["summary"]
