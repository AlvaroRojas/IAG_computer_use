"""Tests for postprocess: aggregation + diff + the per-trade coverage ledger.

A zero-posting export (header-only CSV, WorkerResult.empty=True) is a TRUSTED
result, not a failure:
  * empty on BOTH sides  -> proven MATCH, listed in coverage.empty_both;
  * empty on ONE side    -> present/missing difference (datacompy rows_only_*),
                            listed in coverage.empty_one_side;
  * automation failure   -> dropped from the diff, listed in coverage.failed.
"""

from __future__ import annotations

import pytest

from iag_sim.config import Settings
from iag_sim.models import EnvName, WorkerResult
from iag_sim.orchestration.postprocess import postprocess

REQUIRED = {
    "OPENAI_API_KEY": "sk-test",
    "MUREX_BEFORE_URL": "https://before",
    "MUREX_AFTER_URL": "https://after",
    "MUREX_USER": "u",
    "MUREX_PASS": "p",
}

JOIN = ["trade_id", "gl_account", "currency"]
SEP = ";"


@pytest.fixture
def settings(monkeypatch):
    for k, v in REQUIRED.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DIFF_JOIN_COLUMNS", ",".join(JOIN))
    monkeypatch.setenv("CSV_DELIMITER", SEP)
    return Settings(_env_file=None)


def _postings(path, trade_id, gl="100", amount="10.0"):
    # Per-trade export in the RAW Murex delimiter (';'); carries the join columns.
    path.write_text(
        f"trade_id{SEP}gl_account{SEP}currency{SEP}amount\n"
        f"{trade_id}{SEP}{gl}{SEP}EUR{SEP}{amount}\n",
        encoding="utf-8",
    )
    return str(path)


def _empty(path):
    # Header-only export: the zero-posting case. Same schema, no data rows.
    path.write_text(f"trade_id{SEP}gl_account{SEP}currency{SEP}amount\n", encoding="utf-8")
    return str(path)


def _ok(trade_id, env, csv_path, *, empty=False):
    return WorkerResult(
        trade_id=trade_id, env=env, ok=True, csv_path=csv_path, empty=empty
    )


def test_empty_both_is_a_proven_match(settings, tmp_path):
    results = [
        _ok("T1", EnvName.BEFORE, _empty(tmp_path / "b1.csv"), empty=True),
        _ok("T1", EnvName.AFTER, _empty(tmp_path / "a1.csv"), empty=True),
    ]
    s = postprocess(results, settings, tmp_path)
    assert s["diff"]["matches"] is True
    assert s["comparison"] is None  # short-circuit, no datacompy on empty frames
    assert s["coverage"]["empty_both"] == ["T1"]
    assert s["coverage"]["empty_one_side"] == [] and s["coverage"]["failed"] == []


def test_empty_one_side_is_a_difference(settings, tmp_path):
    results = [
        _ok("T1", EnvName.BEFORE, _empty(tmp_path / "b1.csv"), empty=True),
        _ok("T1", EnvName.AFTER, _postings(tmp_path / "a1.csv", "T1")),
    ]
    s = postprocess(results, settings, tmp_path)
    assert s["diff"]["matches"] is False
    assert s["diff"]["rows_only_after"] == 1
    assert s["coverage"]["empty_one_side"] == ["T1"]
    assert s["coverage"]["empty_both"] == []


def test_empty_both_alongside_matching_postings(settings, tmp_path):
    # T1 has identical postings on both sides; T2 is empty on both -> overall MATCH,
    # and T2 is explicitly recorded as a proven empty-both match (not silently dropped).
    results = [
        _ok("T1", EnvName.BEFORE, _postings(tmp_path / "b1.csv", "T1")),
        _ok("T1", EnvName.AFTER, _postings(tmp_path / "a1.csv", "T1")),
        _ok("T2", EnvName.BEFORE, _empty(tmp_path / "b2.csv"), empty=True),
        _ok("T2", EnvName.AFTER, _empty(tmp_path / "a2.csv"), empty=True),
    ]
    s = postprocess(results, settings, tmp_path)
    assert s["diff"]["matches"] is True
    assert s["comparison"] is not None  # datacompy ran on T1's postings
    assert s["coverage"]["empty_both"] == ["T2"]


def test_failed_trade_is_recorded_not_compared(settings, tmp_path):
    results = [
        _ok("T1", EnvName.BEFORE, _postings(tmp_path / "b1.csv", "T1")),
        _ok("T1", EnvName.AFTER, _postings(tmp_path / "a1.csv", "T1")),
        WorkerResult(trade_id="T3", env=EnvName.BEFORE, ok=False, error="no CSV"),
        _ok("T3", EnvName.AFTER, _postings(tmp_path / "a3.csv", "T3")),
    ]
    s = postprocess(results, settings, tmp_path)
    assert "T3" in s["coverage"]["failed"]
    assert any(f["trade_id"] == "T3" for f in s["failures"])
    assert s["coverage"]["per_trade"]["T3"]["before"] == "failed"
