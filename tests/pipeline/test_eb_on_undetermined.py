"""eb_on()'s NaN propagation (state_eb_decision.py + decision_protocol.py).

THE HAZARD: the real state-UR panel has a genuine 1-month gap (Oct 2025, all 51 states, likely the
federal shutdown). `three_month_average()`/`reference_average()` correctly return NaN across it, but
a plain `flat_ok = a >= 6.5` in `eb_on()`, or a plain `>=` inside `decision_protocol.py`'s
`statutory_lookback_ok()`, evaluates NaN to False -- indistinguishable from a genuine "below
threshold" reading. A genuine EB trigger that was True on both sides of the gap would be silently
forced to False at exactly the gap-adjacent months.

THE RULE: both return pandas' nullable "boolean" dtype, so `&`/`|` implement Kleene (three-valued)
logic automatically: NA & False = False, NA & True = NA, NA | True = True, NA | False = NA. No
downstream caller needs a code change to consume this correctly -- scan()'s
`E_all.reindex(common).dropna()` and dmcs.py's `year_mcs()`'s `.dropna()` are already there as
safety nets, and against bool-dtype output they would be silent no-ops (a bool column cannot hold
NaN).

These tests use synthetic data with an explicit, controlled gap (not the real panel -- that's covered
by tests/pipeline/test_dmcs.py's real-data fixture) so the exact expected NA/True/False pattern is
known in advance and independent of what's in the real data on any given day.
"""
import numpy as np
import pandas as pd

import decision_protocol as DP
import state_eb_decision as SEB


def _series_with_gap(values_by_year, gap_month="2020-06-01"):
    """A monthly series covering `values_by_year` (year -> constant monthly value), with one month
    dropped entirely (simulating the real panel's Oct-2025 gap)."""
    idx = pd.date_range(f"{min(values_by_year)}-01-01", f"{max(values_by_year)}-12-01", freq="MS")
    s = pd.Series([values_by_year[d.year] for d in idx], index=idx, dtype=float)
    return s.drop(pd.Timestamp(gap_month))


# ---------------------------------------------------------------------------
# statutory_lookback_ok() -- the (e)(1)(ii) ratio comparison, in isolation.
# ---------------------------------------------------------------------------
def test_statutory_lookback_ok_returns_nullable_boolean_dtype():
    current = pd.Series([9.0, 9.0, np.nan])
    prior = pd.Series([7.5, 7.5, 7.5])
    result = DP.statutory_lookback_ok(current, prior)
    assert result.dtype == "boolean"


def test_statutory_lookback_ok_nan_input_gives_na_not_false():
    current = pd.Series([9.0, np.nan])
    prior = pd.Series([7.5, 7.5])
    result = DP.statutory_lookback_ok(current, prior)
    assert result.iloc[0] == True   # noqa: E712 -- nullable boolean, `is True` doesn't apply
    assert pd.isna(result.iloc[1])


def test_statutory_lookback_ok_determined_values_unaffected():
    """Non-NaN inputs give exactly the same True/False verdict as the old plain comparison --
    this fix changes NaN handling only, not the statutory math itself."""
    current = pd.Series([9.0, 5.0])   # 9.0/7.5=120% >= 110% -> True; 5.0/7.5=66.7% -> False
    prior = pd.Series([7.5, 7.5])
    result = DP.statutory_lookback_ok(current, prior)
    assert list(result) == [True, False]


# ---------------------------------------------------------------------------
# eb_on() end-to-end, on a controlled synthetic gap (real math, real gap mechanism, known answer).
# ---------------------------------------------------------------------------
def test_eb_on_returns_nullable_boolean_dtype():
    u = _series_with_gap({2018: 7.5, 2019: 7.5, 2020: 9.0})
    result = SEB.eb_on(u, u)
    assert result.dtype == "boolean"


def test_eb_on_gap_adjacent_months_are_na_not_false():
    """A genuine EB trigger (9.0 vs a 7.5 prior-year reference, 120% >= 110%, true on both sides of
    the gap) must show <NA> -- not a fabricated False -- at exactly the two months whose
    3-month-average current leg spans the missing month (t-1 and t-2 of the gap)."""
    u = _series_with_gap({2018: 7.5, 2019: 7.5, 2020: 9.0}, gap_month="2020-06-01")
    result = SEB.eb_on(u, u)
    assert pd.isna(result.loc["2020-07-01"])
    assert pd.isna(result.loc["2020-08-01"])


def test_eb_on_non_gap_adjacent_months_stay_determined_true():
    """The same genuine trigger, at months whose window does NOT touch the gap, must remain the
    same determined True it would be without any gap at all -- the fix must not spread
    undeterminedness beyond where the missing month actually reaches."""
    u_gapped = _series_with_gap({2018: 7.5, 2019: 7.5, 2020: 9.0}, gap_month="2020-06-01")
    u_full = pd.Series(
        [7.5] * 24 + [9.0] * 12,
        index=pd.date_range("2018-01-01", "2020-12-01", freq="MS"),
    )
    gapped_result = SEB.eb_on(u_gapped, u_gapped)
    full_result = SEB.eb_on(u_full, u_full)
    unaffected_months = ["2020-03-01", "2020-04-01", "2020-05-01", "2020-09-01", "2020-10-01"]
    for m in unaffected_months:
        assert gapped_result.loc[m] == full_result.loc[m] == True   # noqa: E712


def test_eb_on_flat_leg_itself_undetermined_when_its_own_window_touches_the_gap():
    """`a` (the CURRENT-leg 3-month average) is itself NaN at any month whose t-1/t-2 legs include
    the gap month -- confirmed directly, since three_month_average()'s own published-protocol legs
    are realized[t-1]/realized[t-2]. So `flat_ok` is undetermined there regardless of the
    underlying UR level; this is not a bug, it correctly reflects that the statute's own current-leg
    quantity cannot be computed for that month at all from the published record."""
    u = _series_with_gap({2018: 3.0, 2019: 3.0, 2020: 3.0}, gap_month="2020-06-01")  # well under 6.5 throughout
    a = DP.statutory_round_tur(DP.three_month_average(u, u, "published"))
    assert pd.isna(a.loc["2020-07-01"]) and pd.isna(a.loc["2020-08-01"])
    result = SEB.eb_on(u, u)
    assert pd.isna(result.loc["2020-07-01"]) and pd.isna(result.loc["2020-08-01"])


def test_kleene_and_a_determined_false_leg_is_dispositive_regardless_of_the_other_legs_na():
    """The property eb_on()'s `flat_ok & lookback_ok` relies on: pandas' nullable "boolean" `&`
    implements Kleene logic, where a determined False on ONE operand is dispositive even if the
    OTHER operand is undetermined (NA & False = False, not NA) -- exactly the real-world case of a
    month whose CURRENT leg is determined but whose REFERENCE leg (a different date, shifted +12/+24
    months) independently touches the gap. Tested directly on the dtype/operator eb_on() actually
    uses, not re-derived by hand, since that is the load-bearing mechanism the fix depends on."""
    det_false = pd.array([False], dtype="boolean")
    na = pd.array([pd.NA], dtype="boolean")
    det_true = pd.array([True], dtype="boolean")
    assert (det_false & na)[0] == False    # noqa: E712 -- False dominates AND regardless of the unknown side
    assert pd.isna((det_true & na)[0])     # True & NA genuinely can't tell -> NA
    assert (det_true | na)[0] == True      # noqa: E712 -- True dominates OR regardless of the unknown side
    assert pd.isna((det_false | na)[0])    # False | NA genuinely can't tell -> NA
