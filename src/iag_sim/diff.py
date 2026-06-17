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
    # Value mismatches: rows whose key exists in BOTH envs but a non-key value
    # differs beyond tolerance (datacompy's paired <col>_before/<col>_after shape).
    mismatches: pd.DataFrame
    # Full rows whose key exists in only ONE env (a posting removed/added by the
    # change) — datacompy's df1_unq_rows / df2_unq_rows, original column shape.
    only_before: pd.DataFrame
    only_after: pd.DataFrame

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

    def combined_differences(self) -> pd.DataFrame:
        """All three difference kinds in ONE frame for mismatches.csv, tagged by a
        leading `diff_kind` column: 'value_mismatch' | 'only_before' | 'only_after'.

        One uniform schema — join keys + datacompy's `<col>_before`/`<col>_after`
        pairs — so no extra plain columns:
          * value_mismatch: both sides filled (datacompy `all_mismatch` shape);
          * only_before: the removed row's compared values go in `<col>_before`,
            `<col>_after` left blank;
          * only_after: the added row's values go in `<col>_after`, `<col>_before`
            blank.
        The column union is NaN-filled where a row has no value for a column. Empty
        (header only, just `diff_kind`) when the envs match exactly. (datacompy
        lowercases column names in its diff frames, so headers read lowercase.)"""
        # join keys, lowercased to match datacompy's diff-frame column casing.
        key_lower = {c.lower() for c in self.join_columns}

        def _to_paired(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
            # Lowercase to align with datacompy's all_mismatch frame, then suffix
            # every NON-key (compared) column to `<col>_before` / `<col>_after` so
            # a one-sided row lands in the same pair the value_mismatch rows use.
            out = frame.copy()
            out.columns = [str(c).lower() for c in out.columns]
            rename = {c: f"{c}_{suffix}" for c in out.columns if c not in key_lower}
            return out.rename(columns=rename)

        frames: list[pd.DataFrame] = []
        if self.mismatches is not None and not self.mismatches.empty:
            m = self.mismatches.copy()
            m.insert(0, "diff_kind", "value_mismatch")
            frames.append(m)
        if self.only_before is not None and not self.only_before.empty:
            b = _to_paired(self.only_before, "before")
            b.insert(0, "diff_kind", "only_before")
            frames.append(b)
        if self.only_after is not None and not self.only_after.empty:
            a = _to_paired(self.only_after, "after")
            a.insert(0, "diff_kind", "only_after")
            frames.append(a)
        if not frames:
            return pd.DataFrame(columns=["diff_kind"])
        return pd.concat(frames, ignore_index=True, sort=False)


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
        only_before=cmp.df1_unq_rows,
        only_after=cmp.df2_unq_rows,
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
    # mismatches.csv carries ALL differences (value mismatches + rows only in one
    # env), discriminated by the `diff_kind` column — not just value mismatches.
    result.combined_differences().to_csv(paths["mismatches"], index=False)
    paths["summary"].write_text(
        json.dumps(result.summary(), indent=2), encoding="utf-8"
    )
    return paths
