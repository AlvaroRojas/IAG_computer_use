"""Durable run status: the `RunRecord` and its on-disk `status.json`.

The API's in-memory registry dies with the process, but the run dirs under
`OUTPUT_DIR` outlive it. This module makes the status durable too:

- `write_status` drops `<run_dir>/status.json` (atomic replace) when a run starts
  and again when it finishes, so the full record — engine summary, result code,
  timestamps, error — survives a restart;
- `read_status` rebuilds a `RunRecord` from disk for a run the process never
  started, falling back to INFERENCE for run dirs written before this file existed
  (or by the CLI, which writes no `status.json`): a `comparison/summary.json` on
  disk means the run finished, a lone checkpoint means it was interrupted.

A `status.json` still reading RUNNING is by definition stale — a live run always
has an in-memory record and is served from there — so it hydrates as INTERRUPTED.

Reads/writes never raise: a corrupt or unwritable status file must not take a run
(or the status endpoint) down.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..orchestration.graph import CHECKPOINT_DB
from .artifacts import read_summary_json, run_dir_for
from .schemas import ResultCode, RunStatus
from .service import derive_result_code

log = logging.getLogger("iag_sim.api")

STATUS_FILE = "status.json"

# Files/dirs that mark a directory as a run dir, so `list_run_ids` ignores
# unrelated folders that happen to sit under OUTPUT_DIR.
_RUN_MARKERS = (STATUS_FILE, CHECKPOINT_DB, "run.log", "comparison")


@dataclass
class RunRecord:
    run_id: str
    status: RunStatus
    result_code: ResultCode | None = None
    summary: dict | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    run_dir: Path | None = field(default=None, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_enum(enum_cls, value: object, default=None):
    try:
        return enum_cls(value)
    except ValueError:
        return default


def write_status(run_dir: Path, rec: RunRecord) -> None:
    """Persist `rec` to `<run_dir>/status.json`. Best effort — a failure is logged
    and swallowed, never propagated into the run."""
    payload = {
        "run_id": rec.run_id,
        "status": rec.status.value,
        "result_code": rec.result_code.value if rec.result_code else None,
        "summary": rec.summary,
        "error": rec.error,
        "started_at": _iso(rec.started_at),
        "finished_at": _iso(rec.finished_at),
    }
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        tmp = run_dir / f".{STATUS_FILE}.tmp"
        # default=str: the engine summary holds Paths etc. — never fail on those.
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(run_dir / STATUS_FILE)  # atomic — no half-written status
    except (OSError, TypeError, ValueError) as exc:
        log.warning(f"could not write {STATUS_FILE} for run_id={rec.run_id}: {exc}")


def _from_status_file(run_dir: Path, run_id: str) -> RunRecord | None:
    path = run_dir / STATUS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    status = _parse_enum(RunStatus, data.get("status"), RunStatus.UNKNOWN)
    if status is RunStatus.RUNNING:
        # Live runs are served from memory, so a RUNNING file means the process
        # died mid-run. The checkpoint is still on disk -> re-POST to resume.
        status = RunStatus.INTERRUPTED
    summary = data.get("summary")
    return RunRecord(
        run_id=run_id,
        status=status,
        result_code=_parse_enum(ResultCode, data.get("result_code")),
        summary=summary if isinstance(summary, dict) else None,
        error=data.get("error") if isinstance(data.get("error"), str) else None,
        started_at=_parse_dt(data.get("started_at")),
        finished_at=_parse_dt(data.get("finished_at")),
        run_dir=run_dir,
    )


def _infer_from_disk(output_dir: Path, run_dir: Path, run_id: str) -> RunRecord:
    """Status for a run dir with no `status.json` — a CLI run, or one from before
    status persistence existed. The comparison artifact is the completion signal;
    `derive_result_code` is reused by nesting it under the `diff` key it expects."""
    comparison = read_summary_json(output_dir, run_id)
    if comparison is not None:
        return RunRecord(
            run_id=run_id,
            status=RunStatus.SUCCEEDED,
            result_code=derive_result_code({"diff": comparison}),
            run_dir=run_dir,
        )
    status = (
        RunStatus.INTERRUPTED
        if (run_dir / CHECKPOINT_DB).exists()
        else RunStatus.UNKNOWN
    )
    return RunRecord(run_id=run_id, status=status, run_dir=run_dir)


def read_status(output_dir: Path, run_id: str) -> RunRecord | None:
    """Rebuild a `RunRecord` from the run dir, or None when `run_id` names no
    directory directly under `output_dir` (containment enforced by `run_dir_for`)."""
    run_dir = run_dir_for(output_dir, run_id)
    if run_dir is None:
        return None
    return _from_status_file(run_dir, run_id) or _infer_from_disk(
        output_dir, run_dir, run_id
    )


def list_run_ids(output_dir: Path) -> list[str]:
    """Every run-dir name under `output_dir`, newest first (run ids are UTC
    timestamps, so a reverse lexical sort is chronological)."""
    base = Path(output_dir)
    try:
        entries = list(base.iterdir())
    except OSError:
        return []
    ids = [
        p.name
        for p in entries
        if p.is_dir() and any((p / marker).exists() for marker in _RUN_MARKERS)
    ]
    return sorted(ids, reverse=True)
