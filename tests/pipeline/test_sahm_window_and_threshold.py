"""Regression tests for two findings in
sahm_evans_realtime.py, both inert on every cited canonical value but visible in the shipped
onsets CSV:

1. `sahm()` computed the reference low as `rolling(12).min()` over t-11..t (month t-12 dropped,
   month t included). Sahm (2019) and FRED's SAHMCURRENT use the minimum over the PREVIOUS 12
   months, t-12..t-1 -- reproducing SAHMCURRENT to 0/919 mismatches confirmed
   that window. The shipped file recorded the realized July-2024 trigger as 2024-08.
2. `onsets()` compared `sig >= thr` on raw floats; an exact 0.50 gap under the "published"
   3-term summation evaluates to 0.4999999999999991 and read as not-hot (2008-04 vs 2008-05 for one
   and the same realized series). Now rounded to 9 dp before the comparison.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import sahm_evans_realtime as SER  # noqa: E402


def _monthly(values, start="2000-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_sahm_reference_low_is_the_previous_twelve_months_not_including_t():
    """avg3 dips to 4.0 exactly 12 months before t, is 5.0 in between, and at t itself is 3.0.
    Under t-12..t-1 the reference low at t is 4.0 (the dip is inside the window) so sig = -1.0;
    under the old t-11..t window the dip is excluded and t's own 3.0 becomes the low, giving 0.0."""
    vals = [5.0] * 5 + [4.0] + [5.0] * 11 + [3.0]   # index 5 is the dip; index 17 = t
    avg3 = _monthly(vals)
    sig = SER.sahm(avg3)
    t = avg3.index[17]
    assert sig.loc[t] == -1.0, f"reference low must include t-12 and exclude t: got {sig.loc[t]}"


def test_sahm_reference_low_excludes_month_t_itself():
    """A fresh low AT t must not be its own reference: sig at t must be negative, not zero."""
    vals = [5.0] * 13 + [2.0]
    sig = SER.sahm(_monthly(vals))
    assert sig.iloc[-1] < 0


def test_sahm_reference_low_is_nan_for_the_first_twelve_months():
    """t-12..t-1 needs 12 prior months, so months 0..11 (12 of them) have no reference low and
    month 12 is the first with one."""
    sig = SER.sahm(_monthly([5.0] * 20))
    assert sig.iloc[:12].isna().all()
    assert sig.iloc[12:].notna().all()


def test_onsets_treats_an_exact_half_point_gap_as_hot_despite_binary_noise():
    """Same mechanism as the shipped 2008-04 case: in exact decimal arithmetic (3.0+3.0+3.6)/3 is
    3.2 and 3.2 - 2.7 is 0.5, so the gap is exactly at the threshold. In IEEE754 the intermediate
    is 3.1999999999999997 and the 'published' summation (a + b + c)/3.0 minus the reference low
    lands at 0.49999999999999956 -- just below thr, hence the rounding in onsets()."""
    sig = pd.Series([0.0, 0.0, (3.0 + 3.0 + 3.6) / 3.0 - 2.7],
                    index=pd.date_range("2008-02-01", periods=3, freq="MS"))
    assert sig.iloc[-1] < 0.5, "fixture must exhibit the binary-representation shortfall"
    assert abs(sig.iloc[-1] - 0.5) < 1e-12
    assert SER.onsets(sig, 0.5) == [sig.index[-1]]


def test_onsets_does_not_promote_a_genuinely_smaller_gap():
    sig = pd.Series([0.0, 0.0, 0.49], index=pd.date_range("2008-02-01", periods=3, freq="MS"))
    assert SER.onsets(sig, 0.5) == []
