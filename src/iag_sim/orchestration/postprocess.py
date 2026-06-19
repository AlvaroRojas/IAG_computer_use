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


def _empty_match_summary(join_columns: list[str]) -> dict:
    """A match-shaped diff summary (same keys as CompareResult.summary) for the
    case where every successful trade was empty on both sides: nothing for
    datacompy to compare, but a PROVEN match — both sides ran and produced zero
    postings. Built without calling datacompy on empty frames."""
    return {
        "matches": True,
        "rows_before": 0,
        "rows_after": 0,
        "rows_only_before": 0,
        "rows_only_after": 0,
        "mismatched_rows": 0,
        "join_columns": list(join_columns),
    }


def _build_coverage(status: dict[str, dict[str, str]]) -> dict:
    """Per-trade coverage ledger so empty-both is an EXPLICIT proven match and
    coverage is auditable. `status[env][trade_id]` is one of
    'postings' | 'empty' | 'failed'. Classifies each trade across both envs:
      - empty_both:     empty on before AND after -> proven match (else invisible
                        to datacompy, since a 0-row trade contributes no rows);
      - empty_one_side: empty on exactly one side -> a real present/missing
                        difference (also surfaced by datacompy's rows_only_*);
      - failed:         any side failed / never produced a result.
    Trades with postings on both sides are compared by datacompy and not listed.
    """
    all_ids = sorted(set(status["before"]) | set(status["after"]))
    empty_both: list[str] = []
    empty_one_side: list[str] = []
    failed: list[str] = []
    per_trade: dict[str, dict[str, str]] = {}
    for tid in all_ids:
        b = status["before"].get(tid, "missing")
        a = status["after"].get(tid, "missing")
        per_trade[tid] = {"before": b, "after": a}
        if b in ("failed", "missing") or a in ("failed", "missing"):
            failed.append(tid)
        elif b == "empty" and a == "empty":
            empty_both.append(tid)
        elif (b == "empty") != (a == "empty"):
            empty_one_side.append(tid)
        # else: postings on both sides -> handled by the datacompy diff.
    return {
        "empty_both": empty_both,
        "empty_one_side": empty_one_side,
        "failed": failed,
        "per_trade": per_trade,
    }


def postprocess(results: list[WorkerResult], settings: Settings, run_dir: Path) -> dict:
    """Aggregate + diff. Returns a JSON-serializable run summary."""
    by_env: dict[str, list[tuple[str, Path]]] = {"before": [], "after": []}
    status: dict[str, dict[str, str]] = {"before": {}, "after": {}}
    failures: list[dict] = []
    for r in results:
        env = r.env.value
        if r.ok and r.csv_path:
            by_env[env].append((r.trade_id, Path(r.csv_path)))
            status[env][r.trade_id] = "empty" if r.empty else "postings"
        else:
            failures.append({"trade_id": r.trade_id, "env": env, "error": r.error})
            status[env][r.trade_id] = "failed"

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
        "coverage": _build_coverage(status),
    }

    # Only diff when both sides produced data, else there is nothing to compare.
    if by_env["before"] and by_env["after"]:
        before_df = pd.read_csv(before_csv)
        after_df = pd.read_csv(after_csv)
        if len(before_df) == 0 and len(after_df) == 0:
            # Every successful trade was empty on both sides: a PROVEN match,
            # without leaning on datacompy's empty-frame edge behavior.
            summary["comparison"] = None
            summary["diff"] = _empty_match_summary(settings.diff_join_columns)
        else:
            result = compare(
                before_df,
                after_df,
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
