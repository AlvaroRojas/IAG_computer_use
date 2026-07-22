"""Run status survives a process restart.

The in-memory registry dies with the process; `<run_dir>/status.json` does not.
A "restart" here is a SECOND app (fresh `RunManager`) over the SAME output dir —
byte-identical to what a real restart sees, since the only shared state is disk.

Also covers hydration of run dirs that carry no `status.json` at all: dirs written
by the CLI, or by a build from before status persistence existed.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from iag_sim.api.app import create_app
from iag_sim.api.run_store import STATUS_FILE, list_run_ids, read_status
from iag_sim.api.schemas import RunStatus
from iag_sim.orchestration.graph import CHECKPOINT_DB


def _wait_done(client, run_id, auth, tries=200):
    body = None
    for _ in range(tries):
        body = client.get(f"/runs/{run_id}", headers=auth).json()
        if body["status"] in ("SUCCEEDED", "FAILED"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished: {body}")


@pytest.fixture
def restarted(app, output_dir):
    """A fresh app over the same output dir — the post-restart process."""
    with TestClient(create_app(output_dir=output_dir), raise_server_exceptions=False) as c:
        yield c


def _finished_run(client, auth, payload) -> str:
    run_id = client.post("/runs", json=payload, headers=auth).json()["run_id"]
    _wait_done(client, run_id, auth)
    return run_id


def _legacy_dir(output_dir, run_id: str):
    """A run dir with no status.json, marked as a run dir by its log."""
    run_dir = output_dir / run_id
    run_dir.mkdir()
    (run_dir / "run.log").write_text("", encoding="utf-8")
    return run_dir


# --- restart ------------------------------------------------------------------

def test_status_survives_restart(client, auth, payload, recorder, output_dir, restarted):
    run_id = _finished_run(client, auth, payload)

    body = restarted.get(f"/runs/{run_id}", headers=auth).json()
    assert body["status"] == "SUCCEEDED"
    assert body["result_code"] == "MATCH"
    assert body["summary"]["run_id"] == run_id
    assert "total_execution_seconds" in body["summary"]
    assert body["started_at"] and body["finished_at"]


def test_list_survives_restart(client, auth, payload, recorder, restarted):
    run_id = _finished_run(client, auth, payload)

    listing = restarted.get("/runs", headers=auth).json()
    assert [item["run_id"] for item in listing] == [run_id]
    assert listing[0]["status"] == "SUCCEEDED"


def test_failed_run_keeps_its_error_across_restart(
    client, auth, payload, monkeypatch, restarted
):
    async def boom(trades, settings, run_dir, *, thread_id=None, resume=False):
        raise RuntimeError("docker daemon unreachable")

    import iag_sim.api.run_manager as rm

    monkeypatch.setattr(rm, "run_graph_async", boom)
    run_id = _finished_run(client, auth, payload)

    body = restarted.get(f"/runs/{run_id}", headers=auth).json()
    assert body["status"] == "FAILED"
    assert body["error"] == "RuntimeError: docker daemon unreachable"


def test_running_status_file_hydrates_as_interrupted(
    client, auth, payload, slow_engine, output_dir, restarted
):
    """A run still RUNNING on disk means the owning process died — the checkpoint
    is intact, so the caller re-POSTs the id to resume."""
    run_id = client.post("/runs", json=payload, headers=auth).json()["run_id"]
    assert json.loads(
        (output_dir / run_id / STATUS_FILE).read_text(encoding="utf-8")
    )["status"] == "RUNNING"

    body = restarted.get(f"/runs/{run_id}", headers=auth).json()
    assert body["status"] == "INTERRUPTED"


def test_live_run_beats_the_disk_copy(client, auth, payload, slow_engine):
    """Same process: the in-memory record is authoritative, so a live run still
    reads RUNNING even though disk cannot distinguish it from an interrupted one."""
    run_id = client.post("/runs", json=payload, headers=auth).json()["run_id"]
    assert client.get(f"/runs/{run_id}", headers=auth).json()["status"] == "RUNNING"


# --- hydration of dirs with no status.json ------------------------------------

@pytest.mark.parametrize(
    "matches, expected",
    [(True, "MATCH"), (False, "DIFFERENCES")],
)
def test_legacy_dir_with_comparison_is_succeeded(
    app, auth, output_dir, restarted, matches, expected
):
    run_dir = _legacy_dir(output_dir, "run-20260101-000000")
    (run_dir / "comparison").mkdir()
    (run_dir / "comparison" / "summary.json").write_text(
        json.dumps({"matches": matches, "mismatched_rows": 0 if matches else 3}),
        encoding="utf-8",
    )

    body = restarted.get("/runs/run-20260101-000000", headers=auth).json()
    assert body["status"] == "SUCCEEDED"
    assert body["result_code"] == expected
    assert body["comparison_summary"]["matches"] is matches


def test_legacy_dir_with_only_a_checkpoint_is_interrupted(app, auth, output_dir, restarted):
    run_dir = output_dir / "run-20260101-000001"
    run_dir.mkdir()
    (run_dir / CHECKPOINT_DB).write_text("", encoding="utf-8")

    body = restarted.get("/runs/run-20260101-000001", headers=auth).json()
    assert body["status"] == "INTERRUPTED"


def test_legacy_dir_with_no_trace_is_unknown(app, auth, output_dir, restarted):
    _legacy_dir(output_dir, "run-20260101-000002")

    body = restarted.get("/runs/run-20260101-000002", headers=auth).json()
    assert body["status"] == "UNKNOWN"
    assert body["result_code"] is None


def test_corrupt_status_json_falls_back_to_inference(app, auth, output_dir, restarted):
    run_dir = _legacy_dir(output_dir, "run-20260101-000003")
    (run_dir / STATUS_FILE).write_text("{not json", encoding="utf-8")
    (run_dir / CHECKPOINT_DB).write_text("", encoding="utf-8")

    body = restarted.get("/runs/run-20260101-000003", headers=auth).json()
    assert body["status"] == "INTERRUPTED"


# --- listing hygiene ----------------------------------------------------------

def test_non_run_directories_are_not_listed(app, auth, output_dir, restarted):
    (output_dir / "not-a-run").mkdir()
    _legacy_dir(output_dir, "run-20260101-000004")

    listing = restarted.get("/runs", headers=auth).json()
    assert [item["run_id"] for item in listing] == ["run-20260101-000004"]


def test_listing_is_newest_first(app, auth, output_dir, restarted):
    for name in ("run-20260101-000000", "run-20260301-000000", "run-20260201-000000"):
        _legacy_dir(output_dir, name)

    listing = restarted.get("/runs", headers=auth).json()
    assert [item["run_id"] for item in listing] == [
        "run-20260301-000000",
        "run-20260201-000000",
        "run-20260101-000000",
    ]


# --- containment --------------------------------------------------------------

@pytest.mark.parametrize("run_id", ["..", "../secrets", "sub/dir", "sub\\dir"])
def test_disk_hydration_rejects_traversal(output_dir, run_id):
    """`get` now reaches the filesystem for unknown ids — containment still comes
    from `run_dir_for`, so an id that escapes the output dir resolves to nothing."""
    assert read_status(output_dir, run_id) is None


def test_read_status_of_missing_dir_is_none(output_dir):
    assert read_status(output_dir, "run-does-not-exist") is None


def test_list_run_ids_of_missing_output_dir_is_empty(output_dir):
    assert list_run_ids(output_dir / "nope") == []


def test_write_status_failure_is_swallowed(output_dir, monkeypatch):
    """A status write must never take the run down."""
    from iag_sim.api import run_store

    def explode(*a, **kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(run_store.Path, "mkdir", explode)
    run_store.write_status(
        output_dir / "run-x",
        run_store.RunRecord(run_id="run-x", status=RunStatus.RUNNING),
    )  # no raise
