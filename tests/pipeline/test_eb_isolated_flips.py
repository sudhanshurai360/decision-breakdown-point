"""Regression tests: S7 asserted that an isolated
flip month "does not by itself create or remove a benefits period" -- unbacked, and inverted under
20 CFR 615.11(c) (one "on" determination starts a >=13-week EB period). `_isolated_flips()` now
computes, per flip month, whether a commonly-on month sits at t-1 or t+1; a flip with no such
neighbour is an entire spell that turns on the tied model, not a timing shift.
"""
import pandas as pd

from state_eb_decision import _isolated_flips


def _s(pattern, start="2010-01-01"):
    idx = pd.date_range(start, periods=len(pattern), freq="MS")
    return pd.Series([c == "1" for c in pattern], index=idx)


def test_flip_adjacent_to_a_commonly_on_month_is_a_timing_shift_not_isolated():
    flip = _s("0100")      # flip at month 2
    all_on = _s("0011")    # everyone ON from month 3: the flip is the leading edge of an agreed spell
    iso = _isolated_flips(flip, all_on)
    assert int(iso.sum()) == 0


def test_flip_with_no_commonly_on_neighbour_is_isolated():
    flip = _s("0100")
    all_on = _s("0000")
    iso = _isolated_flips(flip, all_on)
    assert iso.tolist() == [False, True, False, False]


def test_neighbour_outside_the_scored_window_still_counts():
    """The scored window may start AFTER a commonly-on month: all_on is passed from the full
    record, so a flip at the window's first month sees the neighbour just outside it."""
    flip = _s("100", start="2010-02-01")                 # window starts Feb; flip in Feb
    all_on = _s("1000", start="2010-01-01")              # Jan (outside the window) is commonly ON
    iso = _isolated_flips(flip, all_on)
    assert int(iso.sum()) == 0


def test_calendar_gap_is_respected_not_positional():
    """A missing month between the flip and the commonly-on month means they are NOT adjacent."""
    idx = pd.to_datetime(["2010-01-01", "2010-02-01", "2010-04-01"])   # March absent
    flip = pd.Series([False, True, False], index=idx)
    all_on = pd.Series([False, False, True], index=idx)                # April on, but March is a gap
    iso = _isolated_flips(flip, all_on)
    assert bool(iso.loc["2010-02-01"]) is True


def test_non_flip_months_are_never_flagged():
    flip = _s("0000"); all_on = _s("0000")
    assert int(_isolated_flips(flip, all_on).sum()) == 0
