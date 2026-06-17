"""Per-(trade, env) worker: bounded concurrency + retry around simulate_trade.

Shared by both engines (async runner and LangGraph graph) so the unit of work
is identical regardless of orchestrator.
"""

from __future__ import annotations

from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential

from ..models import EnvName, TradeTask, WorkerResult
from ..murex.simulate import simulate_trade
from .resources import Resources


def _failed(result: WorkerResult) -> bool:
    return not result.ok


@retry(
    retry=retry_if_result(_failed),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=False,
)
async def _simulate_with_retry(res: Resources, trade: TradeTask, env: EnvName) -> WorkerResult:
    return await simulate_trade(
        harness=res.harness_for(env),
        trade=trade,
        settings=res.settings,
        backend=res.backend,
        run_dir=res.run_dir,
    )


async def run_worker(res: Resources, trade: TradeTask, env: EnvName) -> WorkerResult:
    """Run one simulation under the per-environment concurrency gate, with retry
    on failure. Each environment ("before"/"after") has its own budget so one
    side cannot starve the other."""
    async with res.semaphore_for(env):
        return await _simulate_with_retry(res, trade, env)
