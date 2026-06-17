"""Command-line entrypoint.

    python -m iag_sim run --trades data/trades.csv [--engine async|langgraph] [--headed]

Reads the trade list, runs the chosen engine across both Murex environments, and
prints the comparison summary (and where the artifacts were written).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings
from .models import TradeTask
from .runlog import setup_run_logging

log = logging.getLogger("iag_sim.cli")


def load_trades(path: Path) -> list[TradeTask]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "trade_id" not in reader.fieldnames:
            raise ValueError(f"{path} must have a 'trade_id' header column")
        trades: list[TradeTask] = []
        for row in reader:
            trade_id = (row.get("trade_id") or "").strip()
            if not trade_id:
                continue
            extra = {
                k: v for k, v in row.items() if k != "trade_id" and v not in (None, "")
            }
            trades.append(TradeTask(trade_id=trade_id, extra=extra))
    if not trades:
        raise ValueError(f"no trades found in {path}")
    return trades


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")


def _cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.headed:
        settings.headless = False
    if args.max_concurrency:
        settings.max_concurrency = args.max_concurrency

    resume = bool(args.resume)
    if resume:
        # Resume is langgraph-only (durable SQLite checkpoint lives in the run dir).
        from .orchestration.graph import CHECKPOINT_DB

        run_dir = Path(args.resume)
        ckpt = run_dir / CHECKPOINT_DB
        if not ckpt.exists():
            print(f"cannot resume: no checkpoint at {ckpt}", file=sys.stderr)
            return 1
        engine = "langgraph"
        trades: list[TradeTask] = []  # restored from the checkpoint, not the CSV
    else:
        run_dir = settings.output_dir / _run_id()
        trades = load_trades(Path(args.trades))
        engine = args.engine

    log_path = setup_run_logging(run_dir)  # run_dir/run.log, timestamped, named by run id
    run_id = run_dir.name
    if resume:
        log.info(
            f"RESUMING run_id: {run_id}  engine: {engine}  run_dir: {run_dir}  log: {log_path}"
        )
        if args.engine != "langgraph":
            log.info("(resume forces engine=langgraph)")
    else:
        log.info(
            f"run_id: {run_id}  trades: {len(trades)}  engine: {engine}  "
            f"run_dir: {run_dir}  log: {log_path}"
        )

    start = time.perf_counter()
    if engine == "langgraph":
        from .orchestration.graph import run_graph_async

        summary = asyncio.run(run_graph_async(trades, settings, run_dir, resume=resume))
    else:
        from .orchestration.runner import run_async

        summary = asyncio.run(run_async(trades, settings, run_dir))
    elapsed = round(time.perf_counter() - start, 2)

    # Stamp run identity + wall-clock cost onto the final results JSON.
    summary = {"run_id": run_id, "total_execution_seconds": elapsed, **summary}
    log.info("final results:\n" + json.dumps(summary, indent=2))
    log.info(f"total execution time: {elapsed:.2f}s")

    diff = summary.get("diff")
    if diff is None:
        log.info("No comparison produced (one or both environments yielded no data).")
        return 1
    if diff["matches"]:
        log.info("RESULT: before and after MATCH (no accounting differences).")
        return 0
    log.info(
        "RESULT: DIFFERENCES FOUND — "
        f"{diff['mismatched_rows']} mismatched rows, "
        f"{diff['rows_only_before']} only-before, {diff['rows_only_after']} only-after. "
        f"See {summary['comparison']['report']}"
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iag-sim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the before/after accounting comparison")
    run.add_argument("--trades", default="data/trades.csv", help="path to trades CSV")
    run.add_argument(
        "--engine", choices=["async", "langgraph"], default="async", help="orchestration engine"
    )
    run.add_argument("--headed", action="store_true", help="show the browser (debug)")
    run.add_argument("--max-concurrency", type=int, default=0, help="override MAX_CONCURRENCY")
    run.add_argument(
        "--resume",
        metavar="RUN_DIR",
        default="",
        help="resume a crashed run from its dir's SQLite checkpoint "
        "(langgraph engine; replays completed workers, re-runs only the rest)",
    )
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
