"""Per-(trade, env) worker: bounded concurrency + retry around simulate_trade.

Shared by both engines (async runner and LangGraph graph) so the unit of work
is identical regardless of orchestrator.

`run_worker` ALWAYS returns a `WorkerResult` — exhausting the retries is a recorded
failure, not an exception. Tenacity signals exhaustion by raising `RetryError`,
whose message is the useless `RetryError[<Future ... returned WorkerResult>]`; that
is unwrapped back to the last attempt's result so `postprocess` can report the real
`error` per (trade, env) and the other side of the comparison still runs.
"""

from __future__ import annotations

from tenacity import (
    RetryError,
    retry,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

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


def _from_retry_error(exc: RetryError, trade: TradeTask, env: EnvName) -> WorkerResult:
    """The last attempt's `WorkerResult` (it carries the readable `error`). Falls
    back to the underlying exception when the final attempt raised instead of
    returning — `simulate_trade` shouldn't, but a result is still owed."""
    attempt = exc.last_attempt
    if attempt is not None and not attempt.failed:
        return attempt.result()
    cause = attempt.exception() if attempt is not None else exc
    return WorkerResult(
        trade_id=trade.trade_id,
        env=env,
        ok=False,
        error=f"{type(cause).__name__}: {cause}",
    )


async def run_worker(res: Resources, trade: TradeTask, env: EnvName) -> WorkerResult:
    """Run one simulation under the per-environment concurrency gate, with retry
    on failure. Each environment ("before"/"after") has its own budget so one
    side cannot starve the other.

    Never raises on automation failure: retry exhaustion comes back as the last
    `WorkerResult` (ok=False, real error message)."""
    async with res.semaphore_for(env):
        try:
            return await _simulate_with_retry(res, trade, env)
        except RetryError as exc:
            return _from_retry_error(exc, trade, env)
