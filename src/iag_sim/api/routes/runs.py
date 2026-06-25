"""Run endpoints: start/resume, status, list."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from ...models import TradeTask
from ..deps import get_run_manager
from ..run_manager import RunManager, RunRecord
from ..schemas import (
    RunListItem,
    RunRequest,
    RunStatusResponse,
    StartRunResponse,
)
from ..security import require_api_key
from ..service import build_settings_from_request

router = APIRouter(prefix="/runs", tags=["runs"])


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _to_status(rec: RunRecord) -> RunStatusResponse:
    return RunStatusResponse(
        run_id=rec.run_id,
        status=rec.status,
        result_code=rec.result_code,
        summary=rec.summary,
        error=rec.error,
        started_at=_iso(rec.started_at),
        finished_at=_iso(rec.finished_at),
    )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=StartRunResponse,
    dependencies=[Depends(require_api_key)],
)
async def start_run(
    req: RunRequest,
    manager: RunManager = Depends(get_run_manager),
) -> StartRunResponse:
    """Start a new run (empty `run_id`) or resume an existing one (folder name).

    Returns 202 immediately; the run executes in the background. Poll
    GET /runs/{run_id} for status + summary."""
    resume = bool(req.run_id)
    if not resume and not req.trades:
        raise HTTPException(
            status_code=422,
            detail="`trades` must not be empty for a new run",
        )

    settings = build_settings_from_request(req)  # 422 on invalid config
    trades = [TradeTask(trade_id=t.trade_id, extra=t.extra) for t in req.trades]

    record = await manager.start(
        run_id=req.run_id, trades=trades, settings=settings, resume=resume
    )
    return StartRunResponse(
        run_id=record.run_id,
        status=record.status,
        message=(
            f"Resuming run {record.run_id}"
            if resume
            else f"Started run {record.run_id}"
        ),
    )


@router.get(
    "",
    response_model=list[RunListItem],
    dependencies=[Depends(require_api_key)],
)
async def list_runs(
    manager: RunManager = Depends(get_run_manager),
) -> list[RunListItem]:
    return [
        RunListItem(
            run_id=r.run_id,
            status=r.status,
            result_code=r.result_code,
            started_at=_iso(r.started_at),
            finished_at=_iso(r.finished_at),
        )
        for r in manager.list_all()
    ]


@router.get(
    "/{run_id}",
    response_model=RunStatusResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_run(
    run_id: str,
    manager: RunManager = Depends(get_run_manager),
) -> RunStatusResponse:
    rec = manager.get(run_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    return _to_status(rec)
