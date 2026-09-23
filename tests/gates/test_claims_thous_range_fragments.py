"""Regression test:
claims.py's `tokenise()` ran its range-pair extraction loop over the ORIGINAL text, not the
THOUS-substituted text the BARE step just below it already uses. A genuine numeric range between
two thousands-separated numbers (e.g. "12,345-67,890 dollars") had its internal comma boundary
read as a second range hyphen ("345-67"), producing spurious fragment tokens ('345', '67')
alongside the real, correct ones ('12,345', '67,890').

Confirmed dormant against the live corpus at the time (no genuine numeric range between two
thousands-separated numbers existed), so this is a "would have failed" reproduction, not a
real-corpus one -- fixed by running the range loop over the same THOUS-substituted text the BARE
step already computes.
"""
import claims as C


def test_thousands_separated_range_produces_no_spurious_fragment_tokens():
    toks = C.tokenise("12,345-67,890 dollars")
    assert "12,345" in toks
    assert "67,890" in toks
    # the bug: the internal comma boundary used to be misread as a second range hyphen
    assert "345" not in toks, "spurious fragment token from the comma boundary (the bug)"
    assert "67" not in toks, "spurious fragment token from the comma boundary (the bug)"


def test_ordinary_non_thousands_range_still_works():
    """The fix must not disturb tokenise()'s existing, correct handling of a plain range."""
    toks = C.tokenise("the estimate is 290-430 units")
    assert "290" in toks
    assert "430" in toks


def test_hyphenated_compound_adjective_after_a_thousands_number_still_benign():
    """The one comma-then-hyphen shape that DOES occur in the live corpus today: a
    thousands-separated count immediately followed by a hyphenated compound adjective
    (e.g. "2,949-model"), not a numeric range. Must not produce a spurious second numeric token."""
    toks = C.tokenise("a panel of 2,949-model forecasts")
    assert "2,949" in toks
    assert "949" not in toks
