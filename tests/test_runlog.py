"""Per-run logging: run.log lives in the run dir (named by run id) and every
line is UTC-timestamped."""

from __future__ import annotations

import logging
import re

from iag_sim.runlog import setup_run_logging

_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z ")


def test_run_log_written_in_run_dir_with_timestamps(tmp_path):
    run_dir = tmp_path / "run-20260602-103747"
    log_path = setup_run_logging(run_dir)
    assert log_path == run_dir / "run.log"
    assert log_path.parent.name == "run-20260602-103747"  # carries the run id

    logging.getLogger("iag_sim.cli").info("hello world")
    for h in logging.getLogger("iag_sim").handlers:
        h.flush()

    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l]
    assert lines, "run.log should not be empty"
    assert _TS.match(lines[-1]), f"line not timestamped: {lines[-1]!r}"
    assert lines[-1].endswith("hello world")


def test_setup_is_idempotent_no_duplicate_handlers(tmp_path):
    run_dir = tmp_path / "run-x"
    setup_run_logging(run_dir)
    setup_run_logging(run_dir)
    # stdout + file handler only — not doubled by the second call.
    assert len(logging.getLogger("iag_sim").handlers) == 2
