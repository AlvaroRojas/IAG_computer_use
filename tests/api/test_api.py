"""API behaviour: auth, new/resume control flow, one-at-a-time, validation.

Deterministic — `run_graph_async` is faked (see conftest). The background task runs
on the TestClient's portal loop; `_wait_done` polls the status endpoint until the
run leaves RUNNING.
"""

from __future__ import annotations

import time

from iag_sim.orchestration.graph import CHECKPOINT_DB


def _wait_done(client, run_id, auth, tries=200):
    body = None
    for _ in range(tries):
        body = client.get(f"/runs/{run_id}", headers=auth).json()
        if body["status"] in ("SUCCEEDED", "FAILED"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished: {body}")


# --- auth ---------------------------------------------------------------------

def test_health_is_unauthenticated(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_post_without_key_is_401(client, payload):
    assert client.post("/runs", json=payload).status_code == 401


def test_post_with_wrong_key_is_401(client, payload):
    r = client.post("/runs", json=payload, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_get_without_key_is_401(client):
    assert client.get("/runs/whatever").status_code == 401


# --- new run ------------------------------------------------------------------

def test_new_run_starts_and_succeeds(client, auth, payload, recorder):
    r = client.post("/runs", json=payload, headers=auth)
    assert r.status_code == 202
    body = r.json()
    run_id = body["run_id"]
    assert run_id.startswith("run-")
    assert body["status"] == "RUNNING"

    done = _wait_done(client, run_id, auth)
    assert done["status"] == "SUCCEEDED"
    assert done["result_code"] == "MATCH"
    assert done["summary"]["run_id"] == run_id
    assert "total_execution_seconds" in done["summary"]
    assert len(recorder) == 1
    assert recorder[0].resume is False
    assert [t.trade_id for t in recorder[0].trades] == ["1472107"]


# --- resume -------------------------------------------------------------------

def test_resume_existing_checkpoint(client, auth, payload, recorder, output_dir):
    run_id = "run-20260101-000000"
    (output_dir / run_id).mkdir()
    (output_dir / run_id / CHECKPOINT_DB).write_text("")  # stub checkpoint

    body = {**payload, "run_id": run_id, "trades": []}  # trades replayed on resume
    r = client.post("/runs", json=body, headers=auth)
    assert r.status_code == 202
    assert r.json()["run_id"] == run_id

    done = _wait_done(client, run_id, auth)
    assert done["status"] == "SUCCEEDED"
    assert recorder[-1].resume is True


def test_resume_missing_checkpoint_is_404(client, auth, payload, recorder):
    body = {**payload, "run_id": "run-does-not-exist", "trades": []}
    r = client.post("/runs", json=body, headers=auth)
    assert r.status_code == 404


# --- one-at-a-time ------------------------------------------------------------

def test_second_run_while_busy_is_409(client, auth, payload, slow_engine):
    first = client.post("/runs", json=payload, headers=auth)
    assert first.status_code == 202
    second = client.post("/runs", json=payload, headers=auth)
    assert second.status_code == 409


# --- validation ---------------------------------------------------------------

def test_missing_required_field_is_422(client, auth, payload):
    body = {k: v for k, v in payload.items() if k != "MUREX_BEFORE_URL"}
    assert client.post("/runs", json=body, headers=auth).status_code == 422


def test_invalid_channel_is_422(client, auth, payload):
    body = {**payload, "MUREX_CHANNEL": "citrix"}
    assert client.post("/runs", json=body, headers=auth).status_code == 422


def test_empty_trades_new_run_is_422(client, auth, payload):
    body = {**payload, "trades": []}
    assert client.post("/runs", json=body, headers=auth).status_code == 422


# --- thick channel derivation -------------------------------------------------

def test_thick_channel_forces_llm_login(client, auth, payload, recorder):
    body = {**payload, "MUREX_CHANNEL": "thick"}
    r = client.post("/runs", json=body, headers=auth)
    assert r.status_code == 202
    _wait_done(client, r.json()["run_id"], auth)
    assert recorder[0].settings.murex_llm_login is True


# --- listing ------------------------------------------------------------------

def test_list_runs(client, auth, payload, recorder):
    r = client.post("/runs", json=payload, headers=auth)
    run_id = r.json()["run_id"]
    _wait_done(client, run_id, auth)

    listing = client.get("/runs", headers=auth).json()
    assert any(item["run_id"] == run_id for item in listing)


def test_get_unknown_run_is_404(client, auth):
    assert client.get("/runs/no-such-run", headers=auth).status_code == 404
