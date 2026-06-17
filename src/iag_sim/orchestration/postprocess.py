"""Deterministic post-processing shared by both engines: aggregate the per-trade
CSVs into before/after aggregates, diff them, write the comparison artifacts.
No LLM involvement here by design.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..aggregate import write_aggregated
from ..config import Settings
from ..diff import compare, write_comparison
from ..models import WorkerResult


def postprocess(results: list[WorkerResult], settings: Settings, run_dir: Path) -> dict:
    """Aggregate + diff. Returns a JSON-serializable run summary."""
    by_env: dict[str, list[tuple[str, Path]]] = {"before": [], "after": []}
    failures: list[dict] = []
    for r in results:
        if r.ok and r.csv_path:
            by_env[r.env.value].append((r.trade_id, Path(r.csv_path)))
        else:
            failures.append({"trade_id": r.trade_id, "env": r.env.value, "error": r.error})

    sep = settings.csv_delimiter
    before_csv = write_aggregated(by_env["before"], run_dir / "before_aggregated.csv", sep=sep)
    after_csv = write_aggregated(by_env["after"], run_dir / "after_aggregated.csv", sep=sep)

    summary: dict = {
        "run_dir": str(run_dir),
        "before_csv": str(before_csv),
        "after_csv": str(after_csv),
        "trades_ok_before": len(by_env["before"]),
        "trades_ok_after": len(by_env["after"]),
        "failures": failures,
    }

    # Only diff when both sides produced data, else there is nothing to compare.
    if by_env["before"] and by_env["after"]:
        result = compare(
            pd.read_csv(before_csv),
            pd.read_csv(after_csv),
            join_columns=settings.diff_join_columns,
            abs_tol=settings.diff_abs_tol,
            rel_tol=settings.diff_rel_tol,
        )
        paths = write_comparison(result, run_dir / "comparison")
        summary["comparison"] = {k: str(v) for k, v in paths.items()}
        summary["diff"] = result.summary()
    else:
        summary["comparison"] = None
        summary["diff"] = None

    return summary
