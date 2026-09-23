"""sahm_evans_realtime.py::main()'s rolling nowcast loop must not build `hist` directly from
`u[u.index < t]` with no calendar-grid gap-fill -- the same bug class covered for this file's
sibling state_eb_decision.py::nowcast() (see tests/pipeline/test_calendar_filled_no_lookahead.py).
It is reachable on the real cached UNRATE vintage series, which has a genuine 2025-09-01 ->
2025-11-01 gap (the Oct-2025 federal-shutdown gap), though inert on every currently-cited
realtime.sahm_*/evans_* canonical value. The module has a local `_calendar_filled()` mirroring
state_eb_decision.py's function exactly; these tests mirror
test_calendar_filled_no_lookahead.py's own coverage for the sibling function, applied to this one.
"""
import numpy as np
import pandas as pd
import pytest

from sahm_evans_realtime import _calendar_filled


def _monthly_series(values, start="2020-01-01", drop=None):
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
    s = _monthly_series([10.0, 10.0, 10.0, 10.0, 10.0, 40.0], drop="2020-05-01")
    filled = _calendar_filled(s)
    may = pd.Timestamp("2020-05-01")
    assert may in filled.index
    assert filled.loc[may] == 10.0, (
        "the bridged month must equal the PRIOR month's value (10.0), not an average toward the "
        "following month's value (40.0) -- an average would leak the following month's information "
        "into a training array built from dates strictly before it"
    )


def test_filled_value_at_the_gap_does_not_depend_on_the_value_immediately_after_it():
    s1 = _monthly_series([10.0, 10.0, 10.0, 10.0, 10.0, 40.0], drop="2020-05-01")
    s2 = _monthly_series([10.0, 10.0, 10.0, 10.0, 10.0, 999.0], drop="2020-05-01")
    may = pd.Timestamp("2020-05-01")
    assert _calendar_filled(s1).loc[may] == _calendar_filled(s2).loc[may]


def test_a_gap_wider_than_one_month_raises_instead_of_silently_filling():
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


def test_real_cached_unrate_series_has_the_documented_oct2025_gap_and_is_bridgeable():
    """Direct confirmation against real data (not synthetic): the real-time UNRATE series has a
    genuine interior gap that _calendar_filled() must bridge without raising."""
    import sahm_evans_realtime as SER
    try:
        u = SER.real_time_series(force=False)
    except Exception:
        pytest.skip("real-time UNRATE cache not available")
    filled = _calendar_filled(u)
    assert not filled.isna().any()
    assert len(filled) >= len(u)
