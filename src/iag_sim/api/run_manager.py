"""One-at-a-time run lifecycle manager.

Enforces a single in-flight run (an `asyncio.Lock` guards the busy-check + slot
claim), launches `run_graph_async` as a detached `asyncio.Task` so it outlives the
HTTP response, and keeps an in-memory registry of run status/result for polling.

`run_graph_async` is awaited directly on the event loop — never wrapped in
`asyncio.run` (which would explode inside FastAPI's running loop). It opens its own
resources + SQLite checkpointer internally, so the manager calls nothing else.

The registry is in-memory, but it is NOT the only copy: every status transition is
mirrored to `<run_dir>/status.json` (`run_store`), and reads fall back to disk. So a
restarted process still lists/serves past runs; a run that died mid-flight comes
back as INTERRUPTED, and the caller re-POSTs its id to resume from the durable
checkpoint (`<run_dir>/checkpoints.sqlite`).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from fastapi import HTTPException, status

from ..config import Settings
from ..models import TradeTask
from ..orchestration.graph import CHECKPOINT_DB, run_graph_async
from ..runlog import setup_run_logging
from .run_store import RunRecord, list_run_ids, read_status, write_status
from .schemas import RunStatus
from .service import derive_result_code, mint_run_id

log = logging.getLogger("iag_sim.api")

__all__ = ["RunManager", "RunRecord"]  # RunRecord re-exported from run_store


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
                run_id=run_id,
                status=RunStatus.RUNNING,
                started_at=_utcnow(),
                run_dir=run_dir,
            )
            self._registry[run_id] = record
            self._current = record
            write_status(run_dir, record)  # visible on disk from the first moment

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
            # Qualify with the type: several exception classes stringify to "" or
            # to an opaque repr, which would leave the caller with a blank error.
            record.error = f"{type(exc).__name__}: {exc}"
            record.status = RunStatus.FAILED
            log.exception(f"run_id={record.run_id} FAILED: {exc}")
        finally:
            record.finished_at = _utcnow()
            write_status(run_dir, record)  # durable outcome, survives a restart
            self._current = None  # release the slot, even on failure

    def get(self, run_id: str) -> RunRecord | None:
        """The live record if this process owns/owned the run, else the record
        rebuilt from the run dir on disk (None when there is no such run dir)."""
        record = self._registry.get(run_id)
        if record is not None:
            return record
        return read_status(self._output_dir, run_id)

    def list_all(self) -> list[RunRecord]:
        """In-memory records merged with every run dir on disk, newest first.
        Memory wins for ids present in both — it is the only place a RUNNING run is
        accurate."""
        records = dict(self._registry)
        for run_id in list_run_ids(self._output_dir):
            if run_id in records:
                continue
            record = read_status(self._output_dir, run_id)
            if record is not None:
                records[run_id] = record
        return sorted(records.values(), key=lambda r: r.run_id, reverse=True)
