"""One-at-a-time run lifecycle manager.

Enforces a single in-flight run (an `asyncio.Lock` guards the busy-check + slot
claim), launches `run_graph_async` as a detached `asyncio.Task` so it outlives the
HTTP response, and keeps an in-memory registry of run status/result for polling.

`run_graph_async` is awaited directly on the event loop — never wrapped in
`asyncio.run` (which would explode inside FastAPI's running loop). It opens its own
resources + SQLite checkpointer internally, so the manager calls nothing else.

The registry is in-memory: if the process restarts mid-run the durable on-disk
checkpoint (`<run_dir>/checkpoints.sqlite`) survives, but status does not — the
caller re-POSTs the same run id to resume from the checkpoint.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from fastapi import HTTPException, status

from ..config import Settings
from ..models import TradeTask
from ..orchestration.graph import CHECKPOINT_DB, run_graph_async
from ..runlog import setup_run_logging
from .schemas import ResultCode, RunStatus
from .service import derive_result_code, mint_run_id

log = logging.getLogger("iag_sim.api")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RunRecord:
    run_id: str
    status: RunStatus
    result_code: ResultCode | None = None
    summary: dict | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


class RunManager:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._lock = asyncio.Lock()
        self._current: RunRecord | None = None
        self._registry: dict[str, RunRecord] = {}

    @property
    def output_dir(self) -> Path:
        """Root holding every run dir — routes resolve on-disk artifacts under it."""
        return self._output_dir

    @property
    def is_busy(self) -> bool:
        return self._current is not None and self._current.status == RunStatus.RUNNING

    async def start(
        self,
        *,
        run_id: str,
        trades: list[TradeTask],
        settings: Settings,
        resume: bool,
    ) -> RunRecord:
        """Claim the single slot and launch the run in the background.

        Raises 409 if a run is already executing, 404 if `resume` is requested but
        the run dir has no checkpoint. The slot is claimed under the lock; the task
        is launched outside it."""
        async with self._lock:
            if self.is_busy:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A run is already executing; one run at a time.",
                )
            if resume:
                run_dir = self._output_dir / run_id
                ckpt = run_dir / CHECKPOINT_DB
                if not ckpt.exists():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No checkpoint found for run '{run_id}' at {ckpt}",
                    )
            else:
                run_id = mint_run_id()  # minted under the lock — no same-second clash
                run_dir = self._output_dir / run_id

            setup_run_logging(run_dir)  # CLI does this; the engine does not
            record = RunRecord(
                run_id=run_id, status=RunStatus.RUNNING, started_at=_utcnow()
            )
            self._registry[run_id] = record
            self._current = record

        log.info(
            f"{'RESUMING' if resume else 'STARTING'} run_id={run_id} "
            f"trades={len(trades)} run_dir={run_dir}"
        )
        record.task = asyncio.create_task(
            self._run(record, trades, settings, run_dir, resume),
            name=f"iag-sim-{run_id}",
        )
        return record

    async def _run(
        self,
        record: RunRecord,
        trades: list[TradeTask],
        settings: Settings,
        run_dir: Path,
        resume: bool,
    ) -> None:
        t0 = perf_counter()
        try:
            summary = await run_graph_async(trades, settings, run_dir, resume=resume)
            elapsed = round(perf_counter() - t0, 2)
            # Stamp run identity + wall-clock cost, matching the CLI's final JSON.
            summary = {
                "run_id": record.run_id,
                "total_execution_seconds": elapsed,
                **summary,
            }
            record.summary = summary
            record.result_code = derive_result_code(summary)
            record.status = RunStatus.SUCCEEDED
            log.info(
                f"run_id={record.run_id} {record.result_code.value} in {elapsed:.2f}s"
            )
        except Exception as exc:  # automation/engine failure — surfaced as FAILED
            record.error = str(exc)
            record.status = RunStatus.FAILED
            log.exception(f"run_id={record.run_id} FAILED: {exc}")
        finally:
            record.finished_at = _utcnow()
            self._current = None  # release the slot, even on failure

    def get(self, run_id: str) -> RunRecord | None:
        return self._registry.get(run_id)

    def list_all(self) -> list[RunRecord]:
        return list(self._registry.values())
