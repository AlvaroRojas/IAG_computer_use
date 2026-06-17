"""Deterministic before/after comparison of aggregated accounting CSVs.

The LLM never touches this step. Comparison uses datacompy (built for financial
reconciliation): join on key columns, tolerance on numeric amounts, fully
reproducible output suitable for audit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import datacompy
import pandas as pd


@dataclass(frozen=True)
class CompareResult:
    matches: bool
    rows_before: int
    rows_after: int
    rows_only_before: int
    rows_only_after: int
    mismatched_rows: int
    join_columns: list[str]
    report: str
    mismatches: pd.DataFrame

    def summary(self) -> dict:
        return {
            "matches": self.matches,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_only_before": self.rows_only_before,
            "rows_only_after": self.rows_only_after,
            "mismatched_rows": self.mismatched_rows,
            "join_columns": self.join_columns,
        }


def compare(
    before: pd.DataFrame,
    after: pd.DataFrame,
    join_columns: list[str],
    abs_tol: float = 0.01,
    rel_tol: float = 0.0,
) -> CompareResult:
    """Compare two aggregated frames on `join_columns` with numeric tolerance."""
    missing_before = [c for c in join_columns if c not in before.columns]
    missing_after = [c for c in join_columns if c not in after.columns]
    if missing_before or missing_after:
        raise ValueError(
            f"join columns missing — before:{missing_before} after:{missing_after}. "
            f"before has {list(before.columns)}, after has {list(after.columns)}"
        )

    # datacompy >=1.0 renamed the pandas class to PandasCompare; fall back to
    # the legacy `Compare` name for older installs.
    compare_cls = getattr(datacompy, "PandasCompare", None) or datacompy.Compare
    cmp = compare_cls(
        before,
        after,
        join_columns=join_columns,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        df1_name="before",
        df2_name="after",
    )

    return CompareResult(
        matches=bool(cmp.matches()),
        rows_before=int(len(before)),
        rows_after=int(len(after)),
        rows_only_before=int(len(cmp.df1_unq_rows)),
        rows_only_after=int(len(cmp.df2_unq_rows)),
        mismatched_rows=int(len(cmp.all_mismatch())),
        join_columns=list(join_columns),
        report=cmp.report(),
        mismatches=cmp.all_mismatch(),
    )


def write_comparison(result: CompareResult, out_dir: Path) -> dict[str, Path]:
    """Write report.txt, mismatches.csv and summary.json. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": out_dir / "report.txt",
        "mismatches": out_dir / "mismatches.csv",
        "summary": out_dir / "summary.json",
    }
    paths["report"].write_text(result.report, encoding="utf-8")
    result.mismatches.to_csv(paths["mismatches"], index=False)
    paths["summary"].write_text(
        json.dumps(result.summary(), indent=2), encoding="utf-8"
    )
    return paths
