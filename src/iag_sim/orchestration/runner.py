"""Primary engine: plain asyncio fan-out.

For every (trade x {before, after}) it launches a worker; workers run
concurrently bounded by the semaphore inside `run_worker`. Simple, observable,
no checkpoint serialization concerns. Use the LangGraph engine (graph.py) when
you need resumable checkpointing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..config import Settings
from ..models import EnvName, TradeTask
from .postprocess import postprocess
from .resources import open_resources
from .worker import run_worker


async def run_async(trades: list[TradeTask], settings: Settings, run_dir: Path) -> dict:
    async with open_resources(settings, run_dir) as res:
        tasks = [
            run_worker(res, trade, env)
            for trade in trades
            for env in (EnvName.BEFORE, EnvName.AFTER)
        ]
        results = await asyncio.gather(*tasks)
    return postprocess(list(results), settings, run_dir)


def run(trades: list[TradeTask], settings: Settings, run_dir: Path) -> dict:
    return asyncio.run(run_async(trades, settings, run_dir))
