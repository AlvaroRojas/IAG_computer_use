"""Per-run logging: one timestamped `run.log` inside the run directory.

Every line carries a UTC ISO-8601 timestamp, and the file lives at
`<run_dir>/run.log` so it is scoped to (and named by) the run — the same run id
as the folder. The same records are echoed to stdout for live tailing.

Both the CLI and the computer-use tracer log through the `iag_sim` logger tree
(`iag_sim.cli`, `iag_sim.trace`), so a single `run.log` interleaves orchestration
milestones and every agent's actions in real time.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_LOGGER_ROOT = "iag_sim"


def setup_run_logging(run_dir: Path) -> Path:
    """Attach stdout + `<run_dir>/run.log` handlers to the `iag_sim` logger,
    each line prefixed with a UTC timestamp. Idempotent: re-clears handlers so a
    second call (e.g. another run in-process) does not duplicate output."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    fmt.converter = time.gmtime  # UTC, matches the tracer's JSONL `ts`

    logger = logging.getLogger(_LOGGER_ROOT)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return log_path
