"""`_calendar_filled()` (state_eb_decision.py:67-79) bridges a single-month gap in the state UR
panel for `nowcast()`'s training array, and the fill it uses has to be strictly causal.

The hazard, concretely: bridging the single-month Oct-2025 gap with `.interpolate(limit=1)` would
be a TWO-SIDED fill, building the imputed value from BOTH the month before AND the month after the
gap. Since `nowcast()`'s training array for target month `t` is `u_filled[u_filled.index < t]`, the
training row for the first month after the gap would then incorporate that same month's own real
value -- a genuine look-ahead leak, contradicting nowcast()'s own docstring ("refit on all data
strictly before it", state_eb_decision.py:83). `ffill(limit=1)`, what the function actually uses,
is strictly causal. These tests pin that property and the function's other documented contract:
fail loud on a wider gap.
"""
import numpy as np
import pandas as pd
import pytest

from state_eb_decision import _calendar_filled


def _monthly_series(values, start="2020-01-01", drop=None):
    """A float Series on a monthly DatetimeIndex, with `drop` (an ISO date string, e.g.
    "2020-06-01") removed from the index entirely -- not NaN, ABSENT, matching how the real
    Oct-2025 gap reaches this function, which is why it reindexes onto a full monthly grid before
    filling (state_eb_decision.py:70-71)."""
    idx = pd.date_range(start, periods=len(values), freq="MS")
    s = pd.Series(values, index=idx, dtype=float)
    if drop is not None:
        s = s.drop(pd.Timestamp(drop))
    return s


def test_no_gap_reproduces_the_original_series_unchanged():
    s = _monthly_series(range(12))
    filled = _calendar_filled(s)
    pd.testing.assert_series_equal(filled, s, check_names=False)


def test_single_month_gap_is_filled_with_the_prior_months_value_not_an_average():
    """The load-bearing assertion: ffill(limit=1) carries the PRIOR month's value forward, it does
    NOT average the neighboring months. A value of exactly `before` (not `(before+after)/2`) at the
    bridged month is the direct, checkable signature of a causal (not two-sided) fill."""
    s = _monthly_series([10.0, 10.0, 10.0, 10.0, 10.0, 40.0], drop="2020-05-01")
    # index is Jan,Feb,Mar,Apr,Jun; the gap is May (idx 4 of the full 6-month grid)
    filled = _calendar_filled(s)
    may = pd.Timestamp("2020-05-01")
    assert may in filled.index
    assert filled.loc[may] == 10.0, (
        "the bridged month must equal the PRIOR month's value (10.0), not an average toward the "
        "following month's value (40.0) -- an average would leak the following month's information "
        "into a training array built from dates strictly before it"
    )


def test_filled_value_at_the_gap_does_not_depend_on_the_value_immediately_after_it():
    """Direct no-look-ahead check: change ONLY the value of the month immediately after the gap,
    holding everything up to and including the gap fixed, and confirm the imputed gap value is
    unchanged. A two-sided interpolation would fail this (the imputed value would shift toward
    the new following-month value); a causal ffill cannot, by construction."""
    s1 = _monthly_series([10.0, 10.0, 10.0, 10.0, 10.0, 40.0], drop="2020-05-01")
    s2 = _monthly_series([10.0, 10.0, 10.0, 10.0, 10.0, 999.0], drop="2020-05-01")
    may = pd.Timestamp("2020-05-01")
    assert _calendar_filled(s1).loc[may] == _calendar_filled(s2).loc[may]


def test_a_gap_wider_than_one_month_raises_instead_of_silently_filling():
    """The function's own documented fail-loud branch (never exercised by any test before this
    file) -- ffill(limit=1) leaves a 2-month gap only
    partially filled (the second missing month stays NaN), which must raise, not silently pass a
    partially-imputed series downstream."""
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    s = pd.Series([10.0, 10.0, 10.0, np.nan, np.nan, 10.0], index=idx, dtype=float).dropna()
    with pytest.raises(ValueError, match="un-bridgeable gap"):
        _calendar_filled(s)


def test_filled_series_has_no_remaining_gaps_in_its_calendar_grid():
    s = _monthly_series([1.0, 2.0, 3.0, 4.0, 5.0], drop="2020-03-01")
    filled = _calendar_filled(s)
    full_grid = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    assert list(filled.index) == list(full_grid)
    assert not filled.isna().any()
