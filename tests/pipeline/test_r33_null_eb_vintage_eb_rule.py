"""Mutation-closing tests: 11 of 15 mutations of the
EB trigger rule and of scan()'s window / near / knife / isolated lines survived the suite because
no test evaluated the rule at a boundary against a hand-computed answer. Every test here was
verified to FAIL on the named mutant(s) and PASS on the frozen modules.

The reference implementation below is INDEPENDENT of decision_protocol.py: explicit month
offsets (t-1, t-2 for the current leg; t-12..t-14 and t-24..t-26 for the two reference legs),
exact rational arithmetic (fractions.Fraction) with round-half-up at each statutory stage
(20 CFR 615.12(e)(3) tenths; (e)(1)(ii) hundredths-then-integer), explicit three-valued logic.
All fixture values are on the tenths grid in [3.0, 20.0), where the verdict of the float cascade
and of exact arithmetic agree.
"""
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
for _sub in ("pipeline", "instrument"):
    _p = str(ROOT / "code" / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

import state_eb_decision as SED  # noqa: E402


# ---------------------------------------------------------------------------------------------
# Independent reference: 20 CFR 615.12(e) on a gap-free monthly calendar, exact arithmetic
# ---------------------------------------------------------------------------------------------

def _rhu(x: Fraction, decimals: int) -> Fraction:
    """Round half up (= half away from zero for the non-negative rates here) at `decimals`."""
    f = Fraction(10) ** decimals
    return Fraction(math.floor(x * f + Fraction(1, 2))) / f


def _ref_lookback_ok(a: Fraction, r: Fraction) -> bool:
    """(e)(1)(ii): ratio of the two tenths-rounded trigger values, rounded to hundredths, times
    100, rounded to an integer, at or above 110."""
    return _rhu(_rhu(a / r, 2) * 100, 0) >= 110


def ref_eb_on(fc_t: list, u_t: list) -> list:
    """fc_t, u_t: integer TENTHS (65 == 6.5%) on consecutive calendar months. Returns, per month,
    True / False / None (undetermined). Current leg: (fc[t] + u[t-1] + u[t-2]) / 3 rounded to
    tenths; reference legs: mean of u over t-12..t-14 and t-24..t-26, each rounded to tenths;
    ON iff a >= 6.5 AND (110% of either reference leg). Three-valued: a missing reference leg is
    unknown, a False floor is dispositive, False|unknown is unknown."""
    n = len(u_t)
    out = []
    for i in range(n):
        if i < 2:
            out.append(None); continue
        a = _rhu(Fraction(fc_t[i] + u_t[i - 1] + u_t[i - 2], 30), 1)
        flat = a >= Fraction(65, 10)
        legs = []
        for lag in (12, 24):
            if i - lag - 2 < 0:
                legs.append(None); continue
            r = _rhu(Fraction(u_t[i - lag] + u_t[i - lag - 1] + u_t[i - lag - 2], 30), 1)
            legs.append(_ref_lookback_ok(a, r))
        if True in legs:
            look = True
        elif None in legs:
            look = None
        else:
            look = False
        if flat is False:
            out.append(False)
        elif look is None:
            out.append(None)
        else:
            out.append(bool(look))
    return out


def _series(tenths, start="1990-01-01"):
    idx = pd.date_range(start, periods=len(tenths), freq="MS")
    return pd.Series(np.asarray(tenths, float) / 10.0, index=idx)


def _assert_matches_reference(fc_t, u_t):
    code = SED.eb_on(_series(fc_t), _series(u_t))
    ref = ref_eb_on(fc_t, u_t)
    assert len(code) == len(ref)
    for d, c, r in zip(code.index, code.tolist(), ref):
        if r is None:
            assert pd.isna(c), f"{d.date()}: code={c}, reference=undetermined"
        else:
            assert not pd.isna(c) and bool(c) is r, f"{d.date()}: code={c}, reference={r}"


# ---------------------------------------------------------------------------------------------
# eb_on() at the statutory boundaries
# ---------------------------------------------------------------------------------------------

def _three_year_panel(l1, l2, l3):
    """36 months: year 1 flat at l1, year 2 at l2, year 3 at l3 (tenths), with fc == u. At months
    26..35 the legs are clean: current = (fc, l3, l3), reference-1 = l2, reference-2 = l1."""
    u = [l1] * 12 + [l2] * 12 + [l3] * 12
    return list(u), u


@pytest.mark.parametrize("l1,l2,l3,expected,why", [
    (59, 59, 65, True,  "a = 6.5 EXACTLY (floor is >=), 6.5/5.9 = 1.1017 -> 110 -> ON"),
    (60, 60, 66, True,  "6.6/6.0 = 1.10 EXACTLY -> 110 (threshold is >=) -> ON"),
    (50, 65, 70, True,  "prior year 1 fails (7.0/6.5 -> 108), prior year 2 passes (140): EITHER -> ON"),
    (65, 50, 70, True,  "prior year 1 passes (140), prior year 2 fails (108): EITHER -> ON"),
    (50, 50, 64, False, "a = 6.4 < 6.5 floor -> OFF although 128% of both prior years"),
    (60, 60, 65, False, "floor met at 6.5 but 6.5/6.0 = 1.0833 -> 108 < 110 -> OFF"),
    (61, 61, 66, False, "6.6/6.1 = 1.0820 -> 108 -> OFF (one tenth away from the exact-110 case)"),
    (66, 66, 72, False, "7.2/6.6 = 1.0909 -> 109 -> OFF (one integer point below 110)"),
])
def test_eb_on_boundary_rows_hand_verified_against_20_cfr_615_12(l1, l2, l3, expected, why):
    """kills: #3 `(a >= 6.5)` -> `(a > 6.5)` (rows with a == 6.5 exactly: DC 2008-08, OR 2001-09 /
    2008-09, GA 2008-09 on the real panel), #4 `ratio_pct_int >= 110` -> `> 110` (the exact-110
    row), #11 `ok(ref1) | ok(ref2)` -> `&` (the either-year rows). Each row's verdict is
    hand-derived in `why`, not recomputed from the code."""
    fc, u = _three_year_panel(l1, l2, l3)
    out = SED.eb_on(_series(fc), _series(u))
    t = out.index[30]
    assert not pd.isna(out.loc[t]), why
    assert bool(out.loc[t]) is expected, why
    assert ref_eb_on(fc, u)[30] is expected, "reference implementation disagrees with the hand verdict"


def test_eb_on_reference_legs_are_twelve_and_twenty_four_months_back():
    """kills: #1 `date_shift(ar, 12)` -> 11, #2 `date_shift(ar, 24)` -> 23.

    u is flat 6.0 with one-month spikes to 9.0 at index 5 and index 17; the forecast makes the
    current leg a = round1((7.8 + 6.0 + 6.0)/3) = 6.6 at every t >= 26. A reference leg that
    contains a spike is round1((9.0 + 6.0 + 6.0)/3) = 7.0 -> 6.6/7.0 = 0.943 -> 94 -> that leg
    fails; a spike-free leg is 6.0 -> 6.6/6.0 = 1.10 -> 110 -> passes. The spike at 17 sits in
    t-12..t-14 exactly for t in {29, 30, 31} and the spike at 5 sits in t-24..t-26 for the same
    three months, so those are the only OFF months. Under an 11-month (resp. 23-month) look-back
    the spiked leg moves to t in {28, 29, 30} while the other leg stays at {29, 30, 31}, so month
    31 (resp. 31) turns ON -- the pattern changes."""
    u = [60] * 36
    u[5] = 90
    u[17] = 90
    fc = [60] * 26 + [78] * 10
    out = SED.eb_on(_series(fc), _series(u))
    got = {i: bool(out.iloc[i]) for i in range(26, 36)}
    expected = {i: (i not in (29, 30, 31)) for i in range(26, 36)}
    assert got == expected
    assert [ref_eb_on(fc, u)[i] for i in range(26, 36)] == [expected[i] for i in range(26, 36)]


def test_eb_on_matches_the_exact_reference_on_a_random_tenths_series():
    """kills: #1, #2, #3, #4, #11 on a 240-month cyclical series (seed 35: a 6-year sine wave
    plus a random walk, on the tenths grid in [4.0, 9.5]) with tenths-grid forecast noise --
    every determined month's verdict and every undetermined month agree with the independent
    exact-arithmetic reference (77 ON, 145 OFF, 18 undetermined). The fixture is checked to
    contain both exact-6.5 current legs and exact-110 ratios so the boundaries are exercised,
    not just the interior."""
    rng = np.random.default_rng(35)
    n = 240
    base = 62 + 22 * np.sin(2 * np.pi * np.arange(n) / 72) + np.cumsum(rng.integers(-2, 3, n)) * 0.3
    u = [int(np.clip(round(b), 40, 95)) for b in base]
    fc = [int(np.clip(v + rng.integers(-5, 6), 40, 95)) for v in u]
    # fixture self-checks: boundary cases present
    ref = ref_eb_on(fc, u)
    a_t = [_rhu(Fraction(fc[i] + u[i - 1] + u[i - 2], 30), 1) for i in range(2, n)]
    assert sum(1 for i in range(2, n) if a_t[i - 2] == Fraction(65, 10) and ref[i] is True) >= 3, (
        "fixture lacks ON months whose current leg is exactly 6.5")
    ratios = []
    for i in range(26, n):
        a = _rhu(Fraction(fc[i] + u[i - 1] + u[i - 2], 30), 1)
        for lag in (12, 24):
            r = _rhu(Fraction(u[i - lag] + u[i - lag - 1] + u[i - lag - 2], 30), 1)
            ratios.append(_rhu(_rhu(a / r, 2) * 100, 0))
    assert ratios.count(110) >= 3, "fixture lacks exact-110 ratios"
    assert ref.count(True) >= 50 and ref.count(False) >= 50 and ref.count(None) >= 10, (
        "fixture is not informative")
    _assert_matches_reference(fc, u)


# ---------------------------------------------------------------------------------------------
# scan(): window inclusion, near-trigger ANY, knife-edge strictness, isolated on the full record
# ---------------------------------------------------------------------------------------------

def _scan_fixture():
    """One state, 1980-01..2009-12 (360 months), realized UR flat at 5.0 so that every model's
    trigger value is A[t] = round1((fc[t] + 10)/3) and the reference legs are always 5.0 (ON iff
    A >= 6.5, since 6.5/5.0 -> 130). Two tied models ar1/theta with default forecast 5.0 and the
    following designed months (A_ar1, A_theta -> status):
      1996-12  9.5/9.5 -> 6.5/6.5  both ON (commonly ON, just BEFORE the 1997-01 window start)
      1997-01  9.5/9.3 -> 6.5/6.4  flip (ar1 only), spread 0.1 -> knife; near; neighbour 1996-12
                                    commonly ON on the FULL record -> not isolated
      1998-06  9.5/8.6 -> 6.5/6.2  flip, spread 0.3 exactly -> NOT knife (rule is < 0.3); isolated
      1999-09  8.0/5.0 -> 6.0/5.0  no flip; near-trigger because ANY model is in [6, 7]
      2000-03  9.5/9.5 -> 6.5/6.5  commonly ON, no flip; near
      2001-05  5.0/9.5 -> 5.0/6.5  flip (theta only), spread 1.5; near; isolated
    Hand-computed statutory-window (>= 1997-01) totals: evaluated 156, flips 3, near 5,
    near-flips 3, knife 1, isolated 2, episodes 3, max episode 1, divergent states 1."""
    idx = pd.date_range("1980-01-01", "2009-12-01", freq="MS")
    u = pd.Series(5.0, index=idx, name="ZZ")
    fc = {"ar1": pd.Series(5.0, index=idx), "theta": pd.Series(5.0, index=idx)}
    design = {"1996-12-01": (9.5, 9.5), "1997-01-01": (9.5, 9.3), "1998-06-01": (9.5, 8.6),
              "1999-09-01": (8.0, 5.0), "2000-03-01": (9.5, 9.5), "2001-05-01": (5.0, 9.5)}
    for d, (a, b) in design.items():
        fc["ar1"].loc[pd.Timestamp(d)] = a
        fc["theta"].loc[pd.Timestamp(d)] = b
    return u, fc


def _run_scan(monkeypatch, u, fc, start):
    monkeypatch.setattr(SED, "nowcast", lambda s: {m: v.copy() for m, v in fc.items()})
    monkeypatch.setattr(SED, "_tied_models",
                        lambda s, nc: (["ar1", "theta"], s.index[s.index >= "1995-01-01"]))
    df = pd.DataFrame({"ZZ": u})
    return SED.scan(df, protocol="published", windows={"win": pd.Timestamp(start)})["win"]


def test_scan_window_near_knife_and_isolated_counts_match_hand_computed_fixture(monkeypatch):
    """kills: #8 scored window `index >= start` -> `>` (the 1997-01 flip and one evaluated month
    vanish), #9 `near = (...).any(axis=1)` -> `.all(axis=1)` (1999-09 leaves the denominator),
    #10 `knife = spread < 0.3` -> `<= 0.3` (1998-06 becomes knife-edge), #13
    `_isolated_flips(flip, E_all.all(...))` -> `E.all(...)` (1997-01's commonly-on neighbour is
    outside the window, so it would be counted isolated)."""
    u, fc = _scan_fixture()
    acc = _run_scan(monkeypatch, u, fc, "1997-01-01")
    assert acc["tot_eval"] == 156
    assert acc["tot_flip"] == 3
    assert acc["tot_near"] == 5
    assert acc["tot_near_flip"] == 3
    assert acc["tot_knife"] == 1
    assert acc["tot_isolated"] == 2
    assert acc["tot_ep"] == 3
    assert acc["states_div"] == 1
    per = acc["per"][0]
    assert per["flip_months"] == 3 and per["knife_edge"] == 1 and per["isolated_flip_months"] == 2
    assert per["max_episode_months"] == 1 and per["flip_episodes"] == 3
    dates = dict(acc["flip_dates"])["ZZ"]
    assert [d for d, _ in dates] == ["1997-01-01", "1998-06-01", "2001-05-01"]
    assert [s for _, s in dates] == [0.1, 0.3, 1.5]
    # the net-on attribution: ar1 is ON at two of the three flips, theta at one
    assert acc["trigger_counts"] == {"ar1": 1}
    assert acc["trigger_counts_clean"] == {"ar1": 1}


def test_scan_fixture_verdicts_agree_with_the_exact_reference(monkeypatch):
    """The scan fixture's per-model ON/OFF pattern, re-derived by the exact reference on the same
    integer-tenths inputs: exactly the designed months are ON, nothing else."""
    u, fc = _scan_fixture()
    u_t = [int(round(v * 10)) for v in u.tolist()]
    for m, on_dates in (("ar1", ["1996-12-01", "1997-01-01", "1998-06-01", "2000-03-01"]),
                        ("theta", ["1996-12-01", "2000-03-01", "2001-05-01"])):
        fc_t = [int(round(v * 10)) for v in fc[m].tolist()]
        ref = ref_eb_on(fc_t, u_t)
        on = [str(d.date()) for d, r in zip(u.index, ref) if r is True]
        assert on == on_dates, (m, on)


def test_tied_models_common_index_starts_exactly_fifteen_years_after_the_panel_start():
    """kills: #12 `_tied_models` warm-up `year >= min + WARMUP_YEARS` -> `>` (common would start
    a year later, shrinking every scored window by twelve months per state). Panel 1980-01..
    2009-12 -> common must be exactly 1995-01..2009-12 (180 months)."""
    idx = pd.date_range("1980-01-01", "2009-12-01", freq="MS")
    rng = np.random.default_rng(5)
    u = pd.Series(5.0 + np.cumsum(rng.normal(0, 0.1, len(idx))), index=idx, name="ZZ")
    nc = {m: u + rng.normal(0, 0.2 + 0.05 * k, len(idx)) for k, m in enumerate(SED.MODELS)}
    tied, common = SED._tied_models(u, nc)
    assert common[0] == pd.Timestamp("1995-01-01")
    assert common[-1] == pd.Timestamp("2009-12-01")
    assert len(common) == 180
    assert isinstance(tied, list) and all(m in SED.MODELS for m in tied)
