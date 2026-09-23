"""Regression tests for claims.py's `misplaced()` categorical-semicolon-list fix, added
while wiring make_figures.py's output into paper/draft_ssrn.md's prose. Before this fix, a semicolon-joined
value only matched if the WHOLE joined phrase (or its "a, b and c" English rendering) appeared as one
contiguous substring -- built for numeric/date lists like "1988;2009;2010", it never matched a
categorical list of model/series codes ("ar1;dhr;ets;sarima;seasonal_naive;theta"), because prose
enumerates those individually in natural language ("AR1... ETS... the Theta method... DHR"), not as
one joined phrase. Demonstrated real false positives this closed: data.curated_models,
data.curated_series, data.curated_metrics, data.flagship_contrast_models,
data.flagship_scoreboard_models -- all 5 were MISPLACED on the live draft before this fix, and are 0
after it (verified directly: this fix closed 6 of the MISPLACED entries standing on the live draft at
the time, leaving one survivor among them -- a bare bool with no textual rendering at all, an inherent
limit, not a citation gap. The accepted MISPLACED set today is whatever `code/gates/claims.py`'s
`N_MISPLACED` ratchet pins; it has moved since for unrelated reasons, so no absolute count is
restated here).

These tests lock in the MECHANISM on synthetic fixtures, not the live draft's current content, so they
keep passing if draft_ssrn.md's prose changes.
"""
import claims as C


def _entry(value, cites):
    return {"value": value, "producer": "test", "cites": cites, "cross_check": None}


# --------------------------- the real bug: categorical lists scattered in prose ---------------------------

def test_categorical_semicolon_list_found_when_parts_are_scattered_in_prose():
    """The actual false positive this fix closed: each part present individually, in its own prose
    form, nowhere near each other -- not as one joined 'a, b and c' phrase."""
    canon = {"data.curated_models": _entry("ar1;dhr;ets;sarima;theta", ["S3.1"])}
    sections = {
        "S3.1": ("first-order autoregression (AR1), exponential smoothing (ETS), the Theta method, "
                 "seasonal ARIMA (SARIMA), and dynamic harmonic regression (DHR)."),
    }
    bad, unknown, checked, uncited = C.misplaced(canon, sections)
    assert bad == [], f"expected all parts found, got: {bad}"
    assert checked == 1


def test_categorical_list_case_insensitive_match():
    canon = {"data.curated_metrics": _entry("mase;rmsse", ["S3.1"])}
    sections = {"S3.1": "the MASE/RMSSE scale denominator is computed from the fully-revised series."}
    bad, _u, _c, _uc = C.misplaced(canon, sections)
    assert bad == []


def test_categorical_list_underscore_matches_hyphen_or_space_in_prose():
    """The one known code<->prose separator mismatch: 'seasonal_naive' (code) vs 'seasonal-naive'
    (prose). Underscore must also accept a plain space, not just a hyphen."""
    canon = {
        "k1": _entry("seasonal_naive;ar1", ["S1"]),
        "k2": _entry("seasonal_naive;ar1", ["S2"]),
    }
    sections = {
        "S1": "the pool includes seasonal-naive and AR1.",
        "S2": "the pool includes seasonal naive and AR1.",
    }
    bad, _u, _c, _uc = C.misplaced(canon, sections)
    assert bad == [], f"expected hyphen AND space variants to both match, got: {bad}"


# --------------------------- the fix must not become a rubber stamp ---------------------------

def test_categorical_list_still_reports_bad_when_one_part_is_genuinely_missing():
    """THE REGRESSION GUARD: a category the fix must still catch -- one part of the list simply
    never appears in the section at all. If this test fails, the fix has been made too permissive."""
    canon = {"data.curated_models": _entry("ar1;dhr;ets;sarima;theta", ["S3.1"])}
    sections = {
        "S3.1": ("first-order autoregression (AR1), exponential smoothing (ETS), the Theta method, "
                 "and dynamic harmonic regression (DHR)."),   # SARIMA genuinely absent
    }
    bad, _u, _c, _uc = C.misplaced(canon, sections)
    assert len(bad) == 1
    assert "data.curated_models" in bad[0]


def test_categorical_list_still_reports_bad_when_declared_location_is_wrong_section():
    """A category list that's fully present, but in the WRONG section -- the whole point of
    MISPLACED. The fix must not make every categorical value pass regardless of location."""
    canon = {"data.curated_models": _entry("ar1;theta", ["S9"])}
    sections = {
        "S3.1": "the pool includes AR1 and the Theta method.",
        "S9": "an unrelated paragraph about the M4 competition, no model codes here.",
    }
    bad, _u, _c, _uc = C.misplaced(canon, sections)
    assert len(bad) == 1


def test_numeric_semicolon_list_still_requires_contiguous_or_english_join_not_scattered_parts():
    """The categorical-list leniency is scoped OUT of numeric/date lists on purpose (they already
    have a correct, narrower rendering: the "a, b and c" English join). A numeric list must NOT pass
    just because its digits appear scattered unrelated to each other in the text -- that would be a
    real weakening of the date-list check's precision."""
    canon = {"m4.flip_years": _entry("1988;2009;2010", ["S9"])}
    sections = {
        # each individual number present, but scattered, not as "1988, 2009 and 2010" nor as
        # "1988;2009;2010" verbatim -- must NOT be accepted for a purely numeric list.
        "S9": "in 1988 something happened. Separately, 2009 was also a year. So was 2010, eventually.",
    }
    bad, _u, _c, _uc = C.misplaced(canon, sections)
    assert len(bad) == 1, "a numeric list must not gain the categorical leniency"


def test_numeric_semicolon_list_english_join_still_works():
    canon = {"m4.flip_years": _entry("1988;2009;2010", ["S9"])}
    sections = {"S9": "deleting three years (1988, 2009 and 2010) flips the pooled verdict."}
    bad, _u, _c, _uc = C.misplaced(canon, sections)
    assert bad == []


# --------------------------- bool documentation (no rendering possible) ---------------------------

def test_bool_value_has_no_renderings_documented_limit():
    assert C.renderings(True) == set()
    assert C.renderings(False) == set()
