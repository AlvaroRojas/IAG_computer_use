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
