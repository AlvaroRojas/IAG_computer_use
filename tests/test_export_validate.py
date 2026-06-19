"""Reality-gate validator: an export is trusted only when it is a real, non-empty,
parseable CSV whose every posting references THIS trade. Pure — fixtures are
synthetic semicolon CSVs under tmp_path."""

from __future__ import annotations

from pathlib import Path

from iag_sim.murex.export_validate import validate_export

HEADER = "Value date;Rule nb;BO origin ref;Amount;Cur."


def _write(p: Path, *lines: str) -> Path:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _valid_csv(p: Path, trade_id: str = "594", n: int = 2) -> Path:
    rows = [HEADER]
    for i in range(n):
        rows.append(f"2026-06-17;{1000 + i};{trade_id};123.45;EUR")
    return _write(p, *rows)


def test_missing_file_fails(tmp_path):
    c = validate_export(tmp_path / "nope.csv", trade_id="594")
    assert not c.ok and "empty/missing" in c.reason


def test_empty_file_fails(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("", encoding="utf-8")
    c = validate_export(p, trade_id="594")
    assert not c.ok and "empty/missing" in c.reason


def test_unparseable_fails(tmp_path):
    # whitespace-only content is non-empty on disk but has no columns -> parse error
    p = _write(tmp_path / "bad.csv", "", "")
    c = validate_export(p, trade_id="594")
    assert not c.ok and "not valid CSV" in c.reason


def test_header_only_fails_min_rows(tmp_path):
    p = _write(tmp_path / "h.csv", HEADER)
    c = validate_export(p, trade_id="594", min_rows=1)
    assert not c.ok and "rows" in c.reason


def test_header_only_valid_when_min_rows_zero(tmp_path):
    # Default gate (min_rows=0): a header-only CSV is a TRUSTED zero-posting result.
    p = _write(tmp_path / "h0.csv", HEADER)
    c = validate_export(p, trade_id="594", min_rows=0)
    assert c.ok and c.rows == 0 and c.empty is True and c.reason is None


def test_empty_export_still_requires_trade_id_column(tmp_path):
    # Empty but missing the trade-id column -> schema proof fails even at min_rows=0.
    p = _write(tmp_path / "hm.csv", "Value date;Amount")
    c = validate_export(p, trade_id="594", min_rows=0)
    assert not c.ok and "missing trade-id column" in c.reason


def test_nonempty_export_is_not_marked_empty(tmp_path):
    p = _valid_csv(tmp_path / "ne.csv", "594", 2)
    c = validate_export(p, trade_id="594", min_rows=0)
    assert c.ok and c.rows == 2 and c.empty is False


def test_valid_all_rows_match(tmp_path):
    p = _valid_csv(tmp_path / "ok.csv", "594", 2)
    c = validate_export(p, trade_id="594")
    assert c.ok and c.rows == 2 and c.reason is None


def test_missing_trade_id_column(tmp_path):
    p = _write(tmp_path / "m.csv", "Value date;Amount", "2026-06-17;1.0")
    c = validate_export(p, trade_id="594")
    assert not c.ok and "missing trade-id column" in c.reason


def test_row_references_other_trade(tmp_path):
    p = _write(
        tmp_path / "x.csv", HEADER,
        "2026-06-17;1000;594;1;EUR",
        "2026-06-17;1001;999;1;EUR",
    )
    c = validate_export(p, trade_id="594")
    assert not c.ok and "999" in c.reason and "594" in c.reason


def test_require_trade_id_false_skips_check(tmp_path):
    # No matching column, but the trade-id gate is off -> still valid.
    p = _write(tmp_path / "off.csv", "Value date;Amount", "2026-06-17;1.0")
    c = validate_export(p, trade_id="594", require_trade_id=False)
    assert c.ok and c.rows == 1


def test_min_rows_boundary(tmp_path):
    p = _valid_csv(tmp_path / "b.csv", "594", 2)
    assert validate_export(p, trade_id="594", min_rows=2).ok
    assert not validate_export(p, trade_id="594", min_rows=3).ok


def test_trade_id_compared_as_text_no_coercion(tmp_path):
    # Leading-zero ids must survive (dtype=str): "00594" != float 594.
    p = _valid_csv(tmp_path / "z.csv", "00594", 1)
    assert validate_export(p, trade_id="00594").ok
    assert not validate_export(p, trade_id="594").ok


def test_trade_id_whitespace_trimmed(tmp_path):
    p = _write(tmp_path / "w.csv", HEADER, "2026-06-17;1000; 594 ;1;EUR")
    assert validate_export(p, trade_id="594").ok


def test_custom_trade_id_column(tmp_path):
    p = _write(tmp_path / "c.csv", "Trade ref;Amount", "594;1.0")
    assert validate_export(p, trade_id="594", trade_id_columns=["Trade ref"]).ok


def test_matches_id_in_any_of_multiple_columns(tmp_path):
    # Origin/novated trade: queried id 594 lands in "Origin Trade nb", while
    # "Trade nb" holds the resolved trade 777. Matching ANY listed column passes.
    cols = ["Origin Trade nb", "Trade nb", "Amount"]
    p = _write(
        tmp_path / "multi.csv",
        ";".join(cols),
        "594;777;1.0",
        "594;777;2.0",
    )
    c = validate_export(
        p, trade_id="594", trade_id_columns=["Trade nb", "Origin Trade nb"]
    )
    assert c.ok and c.rows == 2
    # Normal trade: id in "Trade nb" also passes.
    p2 = _write(tmp_path / "multi2.csv", ";".join(cols), "111;594;3.0")
    assert validate_export(
        p2, trade_id="594", trade_id_columns=["Trade nb", "Origin Trade nb"]
    ).ok


def test_wrong_trade_fails_when_no_column_matches(tmp_path):
    # A row whose id is in NEITHER listed column -> rejected (anti-hallucination).
    cols = ["Origin Trade nb", "Trade nb", "Amount"]
    p = _write(tmp_path / "wrong.csv", ";".join(cols), "888;777;1.0")
    c = validate_export(
        p, trade_id="594", trade_id_columns=["Trade nb", "Origin Trade nb"]
    )
    assert not c.ok and "594" in c.reason and ("888" in c.reason or "777" in c.reason)


def test_float_formatted_ref_matches_integer_trade_id(tmp_path):
    # Murex writes the ref as a float string ("4572.000000000000"); it must match
    # the integer trade id "4572" after trailing-zero-decimal normalization.
    p = _write(tmp_path / "f.csv", HEADER, "2026-06-17;1000;4572.000000000000;1;EUR")
    assert validate_export(p, trade_id="4572").ok


def test_comma_decimal_ref_matches_integer_trade_id(tmp_path):
    # Real on-prem Murex output uses a COMMA decimal separator
    # ("4572,000000000000"). The cell delimiter is ';' so the comma is
    # unambiguously the decimal mark and must normalize like a period.
    p = _write(tmp_path / "c.csv", HEADER, "2026-06-17;301;4572,000000000000;1000;EUR")
    assert validate_export(p, trade_id="4572").ok


def test_comma_fractional_ref_still_fails(tmp_path):
    # A genuinely fractional comma ref is NOT an integer-id match.
    p = _write(tmp_path / "cf.csv", HEADER, "2026-06-17;301;4572,5;1000;EUR")
    c = validate_export(p, trade_id="4572")
    assert not c.ok and "4572,5" in c.reason


def test_trade_id_given_with_trailing_zeros_normalized(tmp_path):
    # Symmetry: a ".0" on the WANT side normalizes too.
    p = _write(tmp_path / "g.csv", HEADER, "2026-06-17;1000;4572;1;EUR")
    assert validate_export(p, trade_id="4572.0").ok


def test_fractional_ref_still_fails(tmp_path):
    # A genuinely fractional ref is NOT an integer-id match.
    p = _write(tmp_path / "h.csv", HEADER, "2026-06-17;1000;4572.5;1;EUR")
    c = validate_export(p, trade_id="4572")
    assert not c.ok and "4572.5" in c.reason
