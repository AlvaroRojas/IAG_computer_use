"""Tests for per-trade CSV aggregation."""

from __future__ import annotations

import pandas as pd

from iag_sim.aggregate import TRADE_ID_COLUMN, concat_trade_csvs, tag_and_concat


def test_tag_and_concat_tags_trade_id_and_orders_columns():
    a = pd.DataFrame({"gl_account": ["100"], "amount": [10.0], "currency": ["EUR"]})
    b = pd.DataFrame({"gl_account": ["200"], "amount": [20.0], "currency": ["USD"]})

    out = tag_and_concat([("TRD-1", a), ("TRD-2", b)])

    assert list(out.columns) == [TRADE_ID_COLUMN, "amount", "currency", "gl_account"]
    assert out[TRADE_ID_COLUMN].tolist() == ["TRD-1", "TRD-2"]
    assert len(out) == 2


def test_tag_overwrites_existing_trade_id():
    a = pd.DataFrame({"trade_id": ["WRONG"], "amount": [10.0]})
    out = tag_and_concat([("TRD-1", a)])
    assert out[TRADE_ID_COLUMN].tolist() == ["TRD-1"]


def test_empty_input_yields_empty_frame():
    out = tag_and_concat([])
    assert list(out.columns) == [TRADE_ID_COLUMN]
    assert len(out) == 0


def test_concat_trade_csvs_reads_files(tmp_path):
    # Murex "Download as CSV" emits SEMICOLON-separated values — concat must read
    # with sep=";" (the default) or the whole row collapses into one column.
    p1 = tmp_path / "t1.csv"
    p2 = tmp_path / "t2.csv"
    pd.DataFrame({"gl_account": ["100"], "amount": [1.0]}).to_csv(p1, index=False, sep=";")
    pd.DataFrame({"gl_account": ["200"], "amount": [2.0]}).to_csv(p2, index=False, sep=";")

    out = concat_trade_csvs([("TRD-1", p1), ("TRD-2", p2)])
    assert len(out) == 2
    assert set(out[TRADE_ID_COLUMN]) == {"TRD-1", "TRD-2"}
    # Columns must be split, not lumped into one "gl_account;amount" field.
    assert {"gl_account", "amount"} <= set(out.columns)
    assert set(out["gl_account"]) == {100, 200}


def test_concat_trade_csvs_respects_quoted_embedded_delimiter(tmp_path):
    # The real Comment column is double-quoted around embedded ';' — pandas must
    # keep it as ONE field, mirroring the live trade-594 export.
    p = tmp_path / "t.csv"
    p.write_text(
        "Rule nb;Comment\n134;\"PTF_WF;CTP_BNKISS;0;\"\n", encoding="utf-8"
    )
    out = concat_trade_csvs([("594", p)])
    assert len(out) == 1
    assert out["Comment"].iloc[0] == "PTF_WF;CTP_BNKISS;0;"
