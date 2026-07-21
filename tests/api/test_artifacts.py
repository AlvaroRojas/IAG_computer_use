"""Artifact exposure on the status endpoint + the download route.

Deterministic: the comparison files are written straight into the run dir (the
engine is faked in conftest), so these tests cover the API's view of disk only.
"""

from __future__ import annotations

import json
import time

import pytest

from iag_sim.api.artifacts import artifact_path, read_summary_json, run_dir_for


def _wait_done(client, run_id, auth, tries=200):
    body = None
    for _ in range(tries):
        body = client.get(f"/runs/{run_id}", headers=auth).json()
        if body["status"] in ("SUCCEEDED", "FAILED"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished: {body}")


SUMMARY = {"matches": False, "mismatched_rows": 2, "join_columns": ["BO origin ref"]}
MISMATCHES = "diff_kind,BO origin ref\nvalue_mismatch,1472107\n"


def _write_comparison(output_dir, run_id, *, summary=SUMMARY, mismatches=MISMATCHES):
    comparison = output_dir / run_id / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (comparison / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if mismatches is not None:
        # newline="" -> no CRLF translation, so byte size / content assertions hold.
        (comparison / "mismatches.csv").write_text(
            mismatches, encoding="utf-8", newline=""
        )
    return comparison


def _start_and_finish(client, auth, payload, output_dir, **kwargs):
    """Run the faked engine to completion, then plant comparison artifacts in its
    run dir (the fake engine writes none)."""
    run_id = client.post("/runs", json=payload, headers=auth).json()["run_id"]
    _wait_done(client, run_id, auth)
    _write_comparison(output_dir, run_id, **kwargs)
    return run_id


# --- status enrichment --------------------------------------------------------

def test_status_exposes_summary_and_links(
    client, auth, payload, recorder, output_dir
):
    run_id = _start_and_finish(client, auth, payload, output_dir)

    body = client.get(f"/runs/{run_id}", headers=auth).json()
    assert body["comparison_summary"] == SUMMARY

    by_name = {a["name"]: a for a in body["artifacts"]}
    assert set(by_name) == {"summary.json", "mismatches.csv"}  # report.txt absent
    assert by_name["mismatches.csv"]["size_bytes"] == len(MISMATCHES)
    assert by_name["mismatches.csv"]["media_type"] == "text/csv"
    assert by_name["mismatches.csv"]["url"].endswith(
        f"/runs/{run_id}/artifacts/mismatches.csv"
    )


def test_status_without_artifacts_is_empty_not_error(
    client, auth, payload, recorder
):
    """A run that produced no comparison still returns 200 with nulls/empties."""
    run_id = client.post("/runs", json=payload, headers=auth).json()["run_id"]
    body = _wait_done(client, run_id, auth)
    assert body["comparison_summary"] is None
    assert body["artifacts"] == []


def test_corrupt_summary_json_does_not_break_status(
    client, auth, payload, recorder, output_dir
):
    run_id = _start_and_finish(client, auth, payload, output_dir)
    (output_dir / run_id / "comparison" / "summary.json").write_text("{not json")

    body = client.get(f"/runs/{run_id}", headers=auth).json()
    assert body["comparison_summary"] is None
    # The file still exists, so it stays downloadable.
    assert "summary.json" in {a["name"] for a in body["artifacts"]}


# --- download route -----------------------------------------------------------

def test_download_mismatches_csv(client, auth, payload, recorder, output_dir):
    run_id = _start_and_finish(client, auth, payload, output_dir)

    r = client.get(f"/runs/{run_id}/artifacts/mismatches.csv", headers=auth)
    assert r.status_code == 200
    assert r.text == MISMATCHES
    assert r.headers["content-type"].startswith("text/csv")
    assert f"{run_id}-mismatches.csv" in r.headers["content-disposition"]


def test_download_requires_api_key(client, auth, payload, recorder, output_dir):
    run_id = _start_and_finish(client, auth, payload, output_dir)
    assert client.get(f"/runs/{run_id}/artifacts/mismatches.csv").status_code == 401


def test_download_missing_artifact_is_404(
    client, auth, payload, recorder, output_dir
):
    run_id = _start_and_finish(client, auth, payload, output_dir)
    assert client.get(f"/runs/{run_id}/artifacts/report.txt", headers=auth).status_code == 404


def test_download_unknown_name_is_404(client, auth, payload, recorder, output_dir):
    run_id = _start_and_finish(client, auth, payload, output_dir)
    r = client.get(f"/runs/{run_id}/artifacts/checkpoints.sqlite", headers=auth)
    assert r.status_code == 404


def test_download_unknown_run_is_404(client, auth):
    r = client.get("/runs/run-19700101-000000/artifacts/mismatches.csv", headers=auth)
    assert r.status_code == 404


# --- path containment ---------------------------------------------------------

@pytest.mark.parametrize(
    "run_id",
    ["..", "../..", "a/b", "..\\..", "C:\\Windows", "", "."],
)
def test_run_dir_for_rejects_traversal(output_dir, run_id):
    assert run_dir_for(output_dir, run_id) is None


def test_artifact_path_rejects_traversal(output_dir):
    outside = output_dir.parent / "secret.csv"
    outside.write_text("nope", encoding="utf-8")
    assert artifact_path(output_dir, "..", "mismatches.csv") is None


def test_read_summary_json_rejects_non_object(output_dir):
    run_id = "run-20260101-000000"
    _write_comparison(output_dir, run_id, summary=[1, 2, 3])
    assert read_summary_json(output_dir, run_id) is None
