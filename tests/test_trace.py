"""Tracer: JSONL-per-event, flush-per-event, secret redaction, wait-duration echo."""

from __future__ import annotations

import json
import logging

from iag_sim.cua.actions import wait_duration_ms
from iag_sim.cua.trace import NullTracer, Tracer
from iag_sim.runlog import setup_run_logging


def _read_lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def test_event_writes_one_json_line_per_event(tmp_path):
    p = tmp_path / "trace.jsonl"
    t = Tracer(p, label="before:594", echo=False)
    t.event("session_start", model="gpt-5.5", max_turns=60)
    t.event("action", turn=1, action={"type": "click", "x": 10, "y": 20})
    t.close()

    recs = _read_lines(p)
    assert [r["kind"] for r in recs] == ["session_start", "action"]
    assert all(r["label"] == "before:594" for r in recs)
    assert all("ts" in r for r in recs)
    assert recs[0]["model"] == "gpt-5.5"
    assert recs[1]["action"] == {"type": "click", "x": 10, "y": 20}


def test_secret_is_redacted_in_file(tmp_path):
    p = tmp_path / "trace.jsonl"
    t = Tracer(p, label="x", secrets=["MUREXBO"], echo=False)
    t.event("action", turn=1, action={"type": "type", "text": "user MUREXBO pass"})
    t.close()

    raw = p.read_text(encoding="utf-8")
    assert "MUREXBO" not in raw
    assert "***" in raw


def test_flush_per_event_readable_before_close(tmp_path):
    p = tmp_path / "trace.jsonl"
    t = Tracer(p, label="x", echo=False)
    t.event("session_start", model="m")
    # Not closed yet — must already be on disk for live tailing.
    recs = _read_lines(p)
    assert recs and recs[0]["kind"] == "session_start"
    t.close()


def test_empty_secret_not_redacted(tmp_path):
    p = tmp_path / "trace.jsonl"
    t = Tracer(p, label="x", secrets=["", "MUREXBO"], echo=False)
    t.event("action", action={"type": "type", "text": "hello"})
    t.close()
    raw = p.read_text(encoding="utf-8")
    # An empty secret must not turn every char into ***.
    assert "hello" in raw


def test_wait_duration_ms_resolves_aliases_and_default():
    assert wait_duration_ms({"type": "wait"}) == 1000
    assert wait_duration_ms({"type": "wait", "ms": 500}) == 500
    assert wait_duration_ms({"type": "wait", "duration_ms": 2500}) == 2500


def test_wait_ms_surfaces_in_echo(tmp_path):
    # Integration: tracer echo flows through the run logger into run.log.
    log_path = setup_run_logging(tmp_path / "run-z")
    p = tmp_path / "trace.jsonl"
    t = Tracer(p, label="before:594")
    t.event("action", turn=3, action={"type": "wait"}, wait_ms=2000)
    t.close()
    for h in logging.getLogger("iag_sim").handlers:
        h.flush()
    text = log_path.read_text(encoding="utf-8")
    # wait duration shown on the echo line even though it's a top-level field.
    assert "type=wait" in text and "wait_ms=2000" in text
    # and persisted in the JSONL.
    recs = _read_lines(p)
    assert recs[0]["wait_ms"] == 2000


def test_typed_text_in_echo_but_password_redacted(tmp_path):
    # run.log echo must carry the typed text (parity with trace.jsonl) — yet a
    # typed password must still be redacted on the echo line.
    log_path = setup_run_logging(tmp_path / "run-t")
    p = tmp_path / "trace.jsonl"
    t = Tracer(p, label="before:594", secrets=["SECRETPW"])
    t.event("action", turn=2, action={"type": "type", "text": "/exports/accounting_594.csv"})
    t.event("action", turn=3, action={"type": "type", "text": "user SECRETPW"})
    t.close()
    for h in logging.getLogger("iag_sim").handlers:
        h.flush()
    text = log_path.read_text(encoding="utf-8")
    assert "/exports/accounting_594.csv" in text  # typed text now echoed
    assert "SECRETPW" not in text                 # secret redacted in echo
    assert "***" in text


def test_null_tracer_is_noop():
    t = NullTracer()
    t.event("anything", a=1)
    t.close()  # must not raise
