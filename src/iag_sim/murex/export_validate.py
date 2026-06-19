"""Validate a collected Murex accounting-simulation export before the pipeline
trusts it. The computer-use model's "DONE" reply is never a success signal — the
only proof an export is real is a non-empty, parseable CSV on disk whose postings
reference THIS trade. `simulate_trade` runs `validate_export` right after
`collect_export`; a failed check returns `WorkerResult(ok=False, ...)`, which the
existing tenacity retry in `worker.py` re-drives.

Pure (one file read, no other I/O) so it is unit-testable with `tmp_path`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Murex exports a numeric trade ref as a float-formatted string whose decimal
# separator follows the server locale: the "BO origin ref" cell reads
# "4572.000000000000" (period locale) OR "4572,000000000000" (comma locale, the
# real on-prem output) — not "4572". Strip a trailing decimal separator (. or ,)
# + zeros so the ref matches the integer trade id. Done by regex, NOT float()
# round-trip, to avoid precision loss on trade ids beyond 2**53. The cell's
# delimiter is ';' so an in-cell comma is unambiguously the decimal mark. Non-
# numeric or fractional refs (e.g. "4572,5") are left untouched and so still fail
# an integer-id match, as intended.
_TRAILING_ZERO_DECIMALS = re.compile(r"[.,]0+$")


def _norm_ref(value: object) -> str:
    return _TRAILING_ZERO_DECIMALS.sub("", str(value).strip())


@dataclass(frozen=True)
class ExportCheck:
    """Result of validating one export. `reason` is the human-readable failure
    (used verbatim as `WorkerResult.error`); `rows` is the data-row count on pass;
    `empty` marks a trusted zero-posting export (passed with 0 data rows)."""

    ok: bool
    reason: str | None = None
    rows: int = 0
    empty: bool = False


def validate_export(
    path: Path,
    *,
    trade_id: str,
    sep: str = ";",
    min_rows: int = 1,
    require_trade_id: bool = True,
    trade_id_columns: Sequence[str] = ("BO origin ref",),
) -> ExportCheck:
    """Return ExportCheck(ok=True, rows=n) iff `path` is a real export for `trade_id`.

    Checks in order (first failure wins):
      1. file exists and is non-empty;
      2. parses as CSV with the pipeline's delimiter (`sep` = the raw Murex
         delimiter `aggregate.concat_trade_csvs` later reads);
      3. has at least `min_rows` data rows;
      4. (if `require_trade_id`) every row matches `trade_id` in AT LEAST ONE of
         `trade_id_columns`. The queried id lands in "Trade nb" for a normal trade
         but in "Origin Trade nb" for an origin/novated trade (whose "Trade nb" is the
         resolved trade), so matching ANY listed column avoids wrongly rejecting
         legitimate origin trades while still catching a truly wrong export (matches
         none). At least one listed column must be present.

    With the default `min_rows=0`, a header-only CSV (0 data rows) is a VALID
    empty result — a zero-posting accounting simulation that still exported its
    header — and returns `ExportCheck(ok=True, rows=0, empty=True)`. The per-row
    trade-id match is vacuous on 0 rows, but the `trade_id_column` must still be
    present (when `require_trade_id`) as proof the right export schema was emitted.
    A truly broken export (missing file / 0 bytes / unparseable) still fails.
    """
    if not path.exists() or path.stat().st_size == 0:
        return ExportCheck(ok=False, reason="export file empty/missing")

    try:
        # dtype=str: compare the ref column as text, no numeric coercion; keep_default_na
        # off so an empty cell stays "" (not NaN) and fails the ref match explicitly.
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    except Exception as exc:  # pandas raises a variety of parse errors
        return ExportCheck(ok=False, reason=f"export not valid CSV: {type(exc).__name__}: {exc}")

    rows = len(df)
    if rows < min_rows:
        return ExportCheck(ok=False, reason=f"export has {rows} rows, need >= {min_rows}")

    if require_trade_id:
        present = [c for c in trade_id_columns if c in df.columns]
        if not present:
            return ExportCheck(
                ok=False,
                reason=f"export missing trade-id column(s); looked for "
                f"{list(trade_id_columns)}, have {list(df.columns)}",
            )
        want = _norm_ref(trade_id)
        # A row is good if the wanted id appears in ANY present column.
        matches_any = None
        for c in present:
            col_match = df[c].map(_norm_ref) == want
            matches_any = col_match if matches_any is None else (matches_any | col_match)
        bad = ~matches_any
        if bad.any():
            # Report the RAW offending cells (not the normalized form) for debugging.
            idx = bad.idxmax()  # first offending row
            shown = ", ".join(f"{c}={df.loc[idx, c]!r}" for c in present)
            return ExportCheck(
                ok=False,
                reason=f"export row references {shown}; expected {want!r} in one of {present}",
            )

    return ExportCheck(ok=True, rows=rows, empty=(rows == 0))
