"""Run endpoints: start/resume, status, list."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from ...models import TradeTask
from ..artifacts import (
    artifact_path,
    available_artifacts,
    media_type,
    read_summary_json,
)
from ..deps import get_run_manager
from ..run_manager import RunManager, RunRecord
from ..schemas import (
    ArtifactLink,
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


def _artifact_links(
    request: Request, output_dir: Path, run_id: str
) -> list[ArtifactLink]:
    """Download links for the artifacts that exist on disk. URLs are built with
    `url_for` so a server mounted behind a root_path still emits reachable links."""
    return [
        ArtifactLink(
            name=name,
            url=str(request.url_for("get_artifact", run_id=run_id, name=name)),
            size_bytes=size,
            media_type=media_type(name),
        )
        for name, size in available_artifacts(output_dir, run_id)
    ]


def _to_status(
    rec: RunRecord, request: Request, output_dir: Path
) -> RunStatusResponse:
    return RunStatusResponse(
        run_id=rec.run_id,
        status=rec.status,
        result_code=rec.result_code,
        summary=rec.summary,
        comparison_summary=read_summary_json(output_dir, rec.run_id),
        artifacts=_artifact_links(request, output_dir, rec.run_id),
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
    request: Request,
    manager: RunManager = Depends(get_run_manager),
) -> RunStatusResponse:
    rec = manager.get(run_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    return _to_status(rec, request, manager.output_dir)


@router.get(
    "/{run_id}/artifacts/{name}",
    name="get_artifact",
    response_class=FileResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_artifact(
    run_id: str,
    name: str,
    manager: RunManager = Depends(get_run_manager),
) -> FileResponse:
    """Download one comparison artifact (`summary.json`, `mismatches.csv`,
    `report.txt`). 404 when the name is not whitelisted or the file does not exist
    — the run may still be executing, or it produced no comparison at all."""
    path = artifact_path(manager.output_dir, run_id, name)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No artifact '{name}' for run '{run_id}'",
        )
    return FileResponse(
        path, media_type=media_type(name), filename=f"{run_id}-{name}"
    )
