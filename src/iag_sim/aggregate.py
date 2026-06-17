"""Aggregate per-trade CSV exports into one CSV per environment.

Pure logic (`tag_and_concat`) is split from file IO (`concat_trade_csvs`) so the
core is unit-testable with in-memory DataFrames and no disk access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRADE_ID_COLUMN = "trade_id"


def tag_and_concat(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Tag each per-trade frame with its trade_id and concatenate.

    - Ensures a `trade_id` column exists and equals the supplied id (the
      pipeline's source of truth), overwriting any value already in the CSV.
    - Produces a stable column order: trade_id first, then remaining columns
      sorted, so before/after aggregates always line up.
    - Empty input -> empty frame with just the trade_id column.
    """
    tagged: list[pd.DataFrame] = []
    for trade_id, df in frames:
        copy = df.copy()
        copy[TRADE_ID_COLUMN] = trade_id
        tagged.append(copy)

    if not tagged:
        return pd.DataFrame({TRADE_ID_COLUMN: pd.Series(dtype="object")})

    combined = pd.concat(tagged, ignore_index=True)
    other_cols = sorted(c for c in combined.columns if c != TRADE_ID_COLUMN)
    return combined[[TRADE_ID_COLUMN, *other_cols]]


def concat_trade_csvs(
    items: list[tuple[str, Path]], sep: str = ";"
) -> pd.DataFrame:
    """Read each (trade_id, csv_path) and aggregate via `tag_and_concat`.

    `sep` is the RAW Murex export delimiter — Mx.3 "Download as CSV" emits
    SEMICOLON-separated values, not commas. The double-quoted Comment column
    (which contains embedded ';') is parsed correctly by pandas' default
    quotechar.
    """
    frames = [(trade_id, pd.read_csv(path, sep=sep)) for trade_id, path in items]
    return tag_and_concat(frames)


def write_aggregated(
    items: list[tuple[str, Path]], out_path: Path, sep: str = ";"
) -> Path:
    """Aggregate the given per-trade CSVs and write the result to `out_path`.

    Input is read with `sep` (Murex raw delimiter); the aggregate is written
    comma-separated (pandas default) so downstream readers use plain read_csv.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = concat_trade_csvs(items, sep=sep)
    df.to_csv(out_path, index=False)
    return out_path
