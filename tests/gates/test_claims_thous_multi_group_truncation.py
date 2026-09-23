"""Regression test: claims.py's THOUS
regex (`(?<![\\w.])\\d{1,3}(?:,\\d{3})+(?![\\w.])`) truncated a 2+-comma-group thousands number
immediately followed by a word character or a literal "." to a shorter, wrong fragment token.

Mechanism: the trailing lookahead `(?![\\w.])` only constrains what follows the LAST comma-group
the engine has matched so far -- it says nothing about whether another comma-group follows. Python's
`re` backtracks `(?:,\\d{3})+` down to the shortest repetition count that still satisfies the
lookahead, so "1,234,567users" matched as '1,234' (stopping after the first group, since "," is
not a word character or ".") instead of the correct 7-digit token or no match at all. Confirmed live
in isolation before this fix: `re.findall(THOUS, "1,234,567users") == ['1,234']`.

This exact defect was already flagged (but explicitly left unfixed, "out of scope") by
`test_claims_pp_suffix_tokenization.py::test_thousands_separated_pp_value_tokenises_as_one_token_not_fragments`'s
docstring, which noted it "reproduces with a plain trailing '.'/word char, no 'pp' involved" and was
left for a later fix. This is that fix.

Fix: extend the trailing lookahead to `(?![\\w.]|,\\d{3})`, so the engine may not stop at a
comma-group boundary that is itself followed by another comma-group -- it must consume every group
or match none. Verified in isolation this yields NO match at all for the trigger shapes (a
same-direction "too strict" failure, not a new false negative) while leaving the legitimate
single-group case, the correctly-terminated multi-group case, and thousands-separated numeric
ranges unaffected. Confirmed dormant against the live corpus: no 7+-digit, 2+-comma-group number
exists in paper/draft_ssrn.md or paper/note_ssrn.md today, so this fix is a pure no-op on the live gate run
(pinned separately by the live UNRESOLVED/MISPLACED counts, unchanged by this fix).
"""
import claims as C


def test_multi_comma_group_number_followed_by_word_char_does_not_truncate():
    """The bug's exact repro: before the fix, THOUS itself matched and tokenise() returned
    ['1,234'] -- a wrong, shorter fragment silently dressed up as a valid whole thousands-count
    token (and, via the BENIGN "thousands-separated count, matched whole" rule, silently waved
    through with no ORPHAN raised at all). After the fix, THOUS no longer matches this shape, so
    the wrong '1,234' fragment must never appear; the digit groups instead fall through to the
    BARE alternatives as separate, individually-suspicious tokens ('1', '234', ...) -- surfacing
    the malformed shape for scrutiny instead of silently absorbing it as a clean thousands count."""
    toks = C.tokenise("total of 1,234,567users registered")
    assert "1,234" not in toks, "REGRESSION: truncated to the wrong shorter fragment (the bug)"
    assert "1,234,567" not in toks  # THOUS never produces the full number for this shape either
    assert toks == ["1", "234"], f"expected the bare per-group fallback, got {toks}"


def test_multi_comma_group_number_followed_by_literal_period_does_not_truncate():
    """The broader trigger condition: a literal '.' also triggers the truncation, not
    just a word character (e.g. a sentence-final thousands number). Same fallback behavior as the
    word-char case above."""
    toks = C.tokenise("the total reached 1,234,567.")
    assert "1,234" not in toks, "REGRESSION: truncated to the wrong shorter fragment (the bug)"
    assert "1,234,567" not in toks
    assert toks == ["1", "234", "567"], f"expected the bare per-group fallback, got {toks}"


def test_multi_comma_group_number_followed_by_space_is_unaffected():
    """Negative control: the legitimate case (whitespace/EOL after the number, no adjacent word
    char or period) must tokenise correctly both before and after the fix."""
    assert C.tokenise("total of 1,234,567 users registered") == ["1,234,567"]
    assert C.tokenise("a value of 12,345,678 was recorded") == ["12,345,678"]


def test_single_comma_group_thousands_numbers_unaffected():
    """Negative control: the ordinary single-group thousands case (this project's actual live
    usage, e.g. '148,169 claims') must be completely unaffected by widening the lookahead."""
    assert C.tokenise("148,169 claims filed") == ["148,169"]
    assert C.tokenise("a panel of 2,949-model forecasts") == ["2,949"]
    assert C.tokenise("1,200 dollars") == ["1,200"]
    assert C.tokenise("a rate of 1,200%") == ["1,200"]
    assert C.tokenise("the SD is 0.45pp today") == ["0.45"]


def test_legitimate_numeric_ranges_unaffected():
    """Negative control: genuine numeric ranges, including a thousands-separated range, must
    still tokenise both endpoints correctly -- the widened lookahead must not interfere with the
    separate range-pair extraction loop in tokenise()."""
    assert set(C.tokenise("the estimate is 290-430 units")) == {"290", "430"}
    toks = C.tokenise("12,345-67,890 dollars")
    assert "12,345" in toks and "67,890" in toks
    assert "345" not in toks and "67" not in toks


def test_thous_regex_directly_on_the_bug_repro():
    """Pin the regex itself, independent of the rest of tokenise()'s pipeline."""
    import re
    assert re.findall(C.THOUS, "1,234,567users") in ([], ["1,234,567"])
    assert re.findall(C.THOUS, "1,234,567.") in ([], ["1,234,567"])
    assert re.findall(C.THOUS, "1,234,567 users") == ["1,234,567"]
    assert re.findall(C.THOUS, "148,169 claims") == ["148,169"]
