"""Tests for the deterministic before/after diff. This is the audit-critical
component: identical -> no diff; real delta -> flagged; within tolerance -> not
flagged. Must be 100% reproducible."""

from __future__ import annotations

import pandas as pd

from iag_sim.diff import compare, write_comparison

JOIN = ["trade_id", "gl_account", "currency"]


def _frame(amount_100: float, amount_200: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": ["TRD-1", "TRD-1"],
            "gl_account": ["100", "200"],
            "currency": ["EUR", "EUR"],
            "amount": [amount_100, amount_200],
        }
    )


def test_identical_frames_match():
    before = _frame(10.0, 20.0)
    after = _frame(10.0, 20.0)
    result = compare(before, after, JOIN, abs_tol=0.01)
    assert result.matches is True
    assert result.mismatched_rows == 0
    assert result.rows_only_before == 0
    assert result.rows_only_after == 0


def test_amount_delta_beyond_tolerance_is_flagged():
    before = _frame(10.0, 20.0)
    after = _frame(10.0, 25.0)  # +5 on account 200
    result = compare(before, after, JOIN, abs_tol=0.01)
    assert result.matches is False
    assert result.mismatched_rows == 1


def test_delta_within_tolerance_is_not_flagged():
    before = _frame(10.0, 20.0)
    after = _frame(10.005, 20.0)  # +0.005, under abs_tol=0.01
    result = compare(before, after, JOIN, abs_tol=0.01)
    assert result.matches is True
    assert result.mismatched_rows == 0


def test_row_only_in_one_env_is_reported():
    before = _frame(10.0, 20.0)
    after = pd.concat(
        [
            _frame(10.0, 20.0),
            pd.DataFrame(
                {
                    "trade_id": ["TRD-1"],
                    "gl_account": ["300"],
                    "currency": ["EUR"],
                    "amount": [5.0],
                }
            ),
        ],
        ignore_index=True,
    )
    result = compare(before, after, JOIN, abs_tol=0.01)
    assert result.matches is False
    assert result.rows_only_after == 1
    assert result.rows_only_before == 0


def test_missing_join_column_raises():
    before = _frame(10.0, 20.0).drop(columns=["currency"])
    after = _frame(10.0, 20.0)
    try:
        compare(before, after, JOIN)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_determinism_repeated_runs_identical():
    before = _frame(10.0, 20.0)
    after = _frame(10.0, 25.0)
    r1 = compare(before, after, JOIN, abs_tol=0.01)
    r2 = compare(before, after, JOIN, abs_tol=0.01)
    assert r1.summary() == r2.summary()


def test_write_comparison_outputs_files(tmp_path):
    before = _frame(10.0, 20.0)
    after = _frame(10.0, 25.0)
    result = compare(before, after, JOIN, abs_tol=0.01)
    paths = write_comparison(result, tmp_path / "comparison")
    assert paths["report"].exists()
    assert paths["mismatches"].exists()
    assert paths["summary"].exists()


def _three_kinds():
    # before: A matches, B mismatches, key 900 only-before
    before = pd.DataFrame(
        {
            "trade_id": ["TRD-1", "TRD-1", "TRD-1"],
            "gl_account": ["100", "200", "900"],
            "currency": ["EUR", "EUR", "EUR"],
            "amount": [10.0, 20.0, 9.0],
        }
    )
    # after: A matches, B changed (+5), key 800 only-after
    after = pd.DataFrame(
        {
            "trade_id": ["TRD-1", "TRD-1", "TRD-1"],
            "gl_account": ["100", "200", "800"],
            "currency": ["EUR", "EUR", "EUR"],
            "amount": [10.0, 25.0, 8.0],
        }
    )
    return before, after


def test_combined_differences_tags_all_three_kinds():
    before, after = _three_kinds()
    result = compare(before, after, JOIN, abs_tol=0.01)
    assert result.matches is False
    combined = result.combined_differences()
    assert set(combined["diff_kind"]) == {"value_mismatch", "only_before", "only_after"}
    ob = combined[combined["diff_kind"] == "only_before"]
    oa = combined[combined["diff_kind"] == "only_after"]
    assert (ob["gl_account"] == "900").any()  # removed posting present in full
    assert (oa["gl_account"] == "800").any()  # added posting present in full


def test_one_sided_rows_use_the_before_after_pair_only():
    # No extra plain compared column: removed -> amount_before only; added ->
    # amount_after only; the opposite side stays NaN. Uniform schema with mismatches.
    before, after = _three_kinds()
    combined = compare(before, after, JOIN, abs_tol=0.01).combined_differences()
    assert "amount" not in combined.columns
    assert {"amount_before", "amount_after"} <= set(combined.columns)

    ob = combined[combined["diff_kind"] == "only_before"].iloc[0]
    assert ob["amount_before"] == 9.0 and pd.isna(ob["amount_after"])

    oa = combined[combined["diff_kind"] == "only_after"].iloc[0]
    assert oa["amount_after"] == 8.0 and pd.isna(oa["amount_before"])

    vm = combined[combined["diff_kind"] == "value_mismatch"].iloc[0]
    assert vm["amount_before"] == 20.0 and vm["amount_after"] == 25.0


def test_mismatches_csv_contains_added_and_removed(tmp_path):
    before, after = _three_kinds()
    result = compare(before, after, JOIN, abs_tol=0.01)
    paths = write_comparison(result, tmp_path / "comparison")
    df = pd.read_csv(paths["mismatches"], dtype=str)
    assert "diff_kind" in df.columns
    assert {"value_mismatch", "only_before", "only_after"} <= set(df["diff_kind"])
    assert (df["gl_account"] == "900").any() and (df["gl_account"] == "800").any()
    assert "amount" not in df.columns and "amount_before" in df.columns


def test_combined_differences_empty_on_match():
    before = _frame(10.0, 20.0)
    result = compare(before, before.copy(), JOIN, abs_tol=0.01)
    combined = result.combined_differences()
    # Empty (no diff rows) but carries the full STABLE schema, identical to the
    # populated case: diff_kind + join keys + <col>_before/<col>_after pairs.
    assert combined.empty
    assert list(combined.columns) == [
        "diff_kind", "trade_id", "gl_account", "currency",
        "amount_before", "amount_after",
    ]
