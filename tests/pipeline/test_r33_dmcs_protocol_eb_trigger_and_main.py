"""Mutation-verified tests for decision_protocol.py
(via state_eb_decision.eb_on, the statutory trigger) and dmcs.py (year_mcs and main()).

Number-changing mutations in the trigger arithmetic (reference window 3 -> 2, `>= 110` -> `> 110`,
published -> all-forecast current leg) and in dmcs.main()'s own aggregation (near band `<= 7.0` ->
`< 7.0`, decision loss `!=` -> `==`) survive the entire fast suite, because dmcs.main() is never
executed under test and the only trigger-level cases with a hand-known answer sit far from the
6.5 / 110 boundaries. Each test names the mutation it kills ("kills: ...") and was verified to
FAIL on a copy carrying that mutation and PASS on the real module.

Part 1 is a 20-row hand-verified eb_on() ON/OFF table against 20 CFR 615.12(e): the (e)(3)
tenths-rounded 3-month trigger value must be >= 6.5 AND its (e)(1)(ii)-rounded ratio to the same
3-month value one OR two years earlier must "equal or exceed" 110 percent. Part 2 runs dmcs.main()
end to end on a synthetic three-state UR panel with monkeypatched nowcasts, where every dmcs.csv
cell and every dmcs.* emit has a hand-derived value (constant per-year losses make every MCS step a
certain rejection or an exact tie, so no bootstrap arithmetic is involved).
"""
import ast

import numpy as np
import pandas as pd
import pytest

import dmcs
import state_eb_decision as SEB


# ============================================================================================
# Part 1: eb_on() boundary table (decision_protocol.reference_average / three_month_average /
#         statutory_round_tur / statutory_lookback_ok, through SEB.eb_on)
# ============================================================================================

_T = pd.Timestamp("2020-06-01")


def _eb_case(cur, ref1, ref2=(9.0, 9.0, 9.0), fc_2020=None):
    """A 36-month realized series (2018-01 .. 2020-12) evaluated at t = 2020-06.

    cur  = (u[t-2], u[t-1], u[t])  -> the CURRENT leg, placed at 2020-04..06
    ref1 = (u[t-14], u[t-13], u[t-12]) -> the ONE-year-back reference window, 2019-04..06
    ref2 = (u[t-26], u[t-25], u[t-24]) -> the TWO-year-back reference window, 2018-04..06;
           defaults to 9.0 so the t-24 leg can never pass on its own (a/9.0 < 1.10 for a < 9.9)
    Every other month is 5.0. fc_2020, if given, overrides the FORECAST series for every month of
    2020 (the realized series is unchanged), so the published-protocol current leg is
    (fc[t] + u[t-1] + u[t-2]) / 3 while an all-forecast leg would be fc alone.
    Returns eb_on(fc, u).loc[t] as a Python bool."""
    idx = pd.date_range("2018-01-01", "2020-12-01", freq="MS")
    u = pd.Series(5.0, index=idx)
    u.loc["2020-04-01":"2020-06-01"] = list(cur)
    u.loc["2019-04-01":"2019-06-01"] = list(ref1)
    u.loc["2018-04-01":"2018-06-01"] = list(ref2)
    fc = u.copy()
    if fc_2020 is not None:
        fc.loc["2020-01-01":"2020-12-01"] = fc_2020
    out = SEB.eb_on(fc, u).loc[_T]
    assert not pd.isna(out), "fixture error: the trigger must be determined at t"
    return bool(out)


# (label, cur, ref1, ref2, fc_2020, expected). Each row's arithmetic is written out by hand.
_EB_TABLE = [
    # --- the 6.5 flat-leg boundary -------------------------------------------------------------
    ("6.5-exact flat leg, ratio 6.5/5.9 = 110.17 -> rounds to 110 -> ON",
     (6.5, 6.5, 6.5), (5.9, 5.9, 5.9), None, None, True),
    ("6.5 after (e)(3) rounding: (6.4+6.5+6.5)/3 = 6.4667 -> 6.5; vs 5.9 -> ON",
     (6.4, 6.5, 6.5), (5.9, 5.9, 5.9), None, None, True),
    ("6.4 after (e)(3) rounding: (6.4+6.4+6.5)/3 = 6.4333 -> 6.4 < 6.5 -> OFF",
     (6.4, 6.4, 6.5), (5.9, 5.9, 5.9), None, None, False),
    ("flat leg fails at 6.4 even though the look-back is 128 -> OFF",
     (6.4, 6.4, 6.4), (5.0, 5.0, 5.0), None, None, False),
    # --- the 110 look-back boundary ("equals or exceeds") ----------------------------------------
    ("110-exact: 6.6/6.0 = 1.10 -> 110 -> ON (float ratio is 109.99999999999999; the cascade rounds it)",
     (6.6, 6.6, 6.6), (6.0, 6.0, 6.0), None, None, True),
    ("110-exact from above: 7.7/7.0 = 110.00000000000001 -> 110 -> ON",
     (7.7, 7.7, 7.7), (7.0, 7.0, 7.0), None, None, True),
    ("6.5/6.0 = 108.33 -> 108 -> OFF",
     (6.5, 6.5, 6.5), (6.0, 6.0, 6.0), None, None, False),
    ("7.0/6.4 = 109.375 -> (e)(1)(ii) rounds to 109, not 110 -> OFF",
     (7.0, 7.0, 7.0), (6.4, 6.4, 6.4), None, None, False),
    ("8.0/7.5 = 106.67 -> 107 -> OFF",
     (8.0, 8.0, 8.0), (7.5, 7.5, 7.5), None, None, False),
    # --- the reference is (e)(3)-rounded BEFORE the ratio --------------------------------------
    ("reference (5.9+5.9+6.0)/3 = 5.9333 -> 5.9; 6.5/5.9 -> 110 -> ON",
     (6.5, 6.5, 6.5), (5.9, 5.9, 6.0), None, None, True),
    ("reference (5.9+6.0+6.0)/3 = 5.9667 -> 6.0; 6.5/6.0 -> 108 -> OFF",
     (6.5, 6.5, 6.5), (5.9, 6.0, 6.0), None, None, False),
    # --- the reference is a THREE-month average (a 2-month window gives the opposite call) ------
    ("ref (5.0,5.0,8.0): 3-month 6.0 -> 6.6/6.0 = 110 -> ON  [2-month (5.0,8.0) = 6.5 -> 101.5 -> OFF]",
     (6.6, 6.6, 6.6), (5.0, 5.0, 8.0), None, None, True),
    ("ref (5.0,8.0,5.0): 3-month 6.0 -> 110 -> ON  [2-month (8.0,5.0) = 6.5 -> OFF]",
     (6.6, 6.6, 6.6), (5.0, 8.0, 5.0), None, None, True),
    ("ref (8.0,5.0,5.0): 3-month 6.0 -> 110 -> ON  [2-month (5.0,5.0) = 5.0 -> ON too; non-discriminating]",
     (6.6, 6.6, 6.6), (8.0, 5.0, 5.0), None, None, True),
    # --- the current leg is also a three-month average -----------------------------------------
    ("current (5.0+5.0+9.5)/3 = 6.5 -> 6.5; vs 5.9 -> ON",
     (5.0, 5.0, 9.5), (5.9, 5.9, 5.9), None, None, True),
    # --- the published protocol: forecast at t, PUBLISHED at t-1 and t-2 -----------------------
    ("published leg: fc 9.0 all 2020, realized (5,5,5): (9+5+5)/3 = 6.33 -> 6.3 -> OFF  [all-forecast 9.0/5.0 -> ON]",
     (5.0, 5.0, 5.0), (5.0, 5.0, 5.0), None, 9.0, False),
    ("published leg: fc 5.0 all 2020, realized (8,8,8): (5+8+8)/3 = 7.0; 7.0/5.0 = 140 -> ON  [all-forecast 5.0 -> OFF]",
     (8.0, 8.0, 8.0), (5.0, 5.0, 5.0), None, 5.0, True),
    # --- either prior year: the t-24 leg ------------------------------------------------------
    ("t-12 fails (6.6/8.0 = 82.5) but t-24 passes (6.6/5.0 = 132) -> ON",
     (6.6, 6.6, 6.6), (8.0, 8.0, 8.0), (5.0, 5.0, 5.0), None, True),
    ("t-12 gives 108 (6.5/6.0) but t-24 gives 110 (6.5/5.9) -> ON via the second year",
     (6.5, 6.5, 6.5), (6.0, 6.0, 6.0), (5.9, 5.9, 5.9), None, True),
    ("both prior years fail (6.6/7.0 = 94) -> OFF",
     (6.6, 6.6, 6.6), (7.0, 7.0, 7.0), (7.0, 7.0, 7.0), None, False),
]


def _row_id(i, label):
    slug = "".join(ch if ch.isalnum() else "_" for ch in label.split(":")[0][:36]).strip("_")
    return f"row{i + 1:02d}_{slug}"


@pytest.mark.parametrize("label,cur,ref1,ref2,fc_2020,expected", _EB_TABLE,
                         ids=[_row_id(i, r[0]) for i, r in enumerate(_EB_TABLE)])
def test_eb_on_hand_verified_boundary_table(label, cur, ref1, ref2, fc_2020, expected):
    """kills: M6 (`>= 110` -> `> 110`; rows '110-exact', '6.5-exact', 'reference ... 5.9',
    'ref (5.0,5.0,8.0)', 'ref (5.0,8.0,5.0)', 'ref (8.0,5.0,5.0)', 'current ... 9.5', 't-24 gives
    110'), M5 (reference window 3 -> 2; rows 'ref (5.0,5.0,8.0)' and 'ref (5.0,8.0,5.0)'), M4
    (published current leg -> all-forecast; the two 'published leg' rows). Every expected value is
    hand-computed from 20 CFR 615.12(e) as written in the row label."""
    kwargs = {"cur": cur, "ref1": ref1, "fc_2020": fc_2020}
    if ref2 is not None:
        kwargs["ref2"] = ref2
    assert _eb_case(**kwargs) is expected, label


def test_eb_table_rows_cover_both_boundaries_and_both_directions():
    """Guard on the table itself: it must contain a 6.5-exact flat row, a 110-exact look-back row,
    and both ON and OFF outcomes, so a future edit cannot hollow it out."""
    labels = [r[0] for r in _EB_TABLE]
    assert any(l.startswith("6.5-exact") for l in labels)
    assert any(l.startswith("110-exact") for l in labels)
    outcomes = {r[-1] for r in _EB_TABLE}
    assert outcomes == {True, False}
    assert len(_EB_TABLE) == 20


# ============================================================================================
# Part 2: dmcs.year_mcs() call contract and dmcs.main() end to end on a synthetic panel
# ============================================================================================

def test_year_mcs_calls_mcs_with_alpha_010_and_the_given_seed(monkeypatch):
    """kills: M1 (`alpha=0.10` -> 0.25), M2 (`seed=seed` -> `seed+1`).

    The paper commits S7's decision-MCS to the same tuning as the forecast-error MCS (alpha=0.10,
    block 3, B=2000, seed 0). year_mcs() must forward exactly alpha=0.10 and the seed it was
    given, and nothing else (block/B/elimination stay at mcs_test.mcs()'s defaults)."""
    seen = {}

    def spy(L, **kw):
        seen.update(kw)
        return [0, 1], 1.0

    monkeypatch.setattr(dmcs.MCS, "mcs", spy)
    common = pd.date_range("2000-01-01", periods=36, freq="MS")
    dmcs.year_mcs({"A": pd.Series(1.0, index=common), "B": pd.Series(2.0, index=common)}, common, seed=5)
    assert seen == {"alpha": 0.10, "seed": 5}


def _synthetic_state_panel():
    """Three synthetic states on 1978-01 .. 2005-12 (336 months >= MIN_STATE_PANEL_MONTHS=300).
    WARMUP_YEARS=15 puts the scored window at 1993-01 .. 2005-12 (13 full calendar years);
    EB_TUR_EFFECTIVE (1993-03) trims the near-trigger window to 1993-03 onwards.

    Realized UR is a flat 5.0 (AA, BB) or 4.9 (CC), so the realized EB call is OFF every month
    and the reference legs are 5.0 / 4.9. Forecasts are 5.0 except in designated months:

      forecast 9.0  -> trigger value round((9 + 5 + 5)/3, 1) = 6.3  : near-trigger, EB OFF (correct)
      forecast 11.0 -> trigger value round((11 + 5 + 5)/3, 1) = 7.0 : near-trigger ONLY through the
                       band's upper edge (A <= 7.0), and EB ON (7.0 >= 6.5, 7.0/5.0 = 140) -> a
                       misclassification of the realized OFF call

    AA: good = 9.0 in Apr-Sep;  edge = 11.0 in Oct-Dec;  bad = 11.0 in Apr-Sep
    BB: good = 9.0 in Apr-Sep;  edge = 9.0 in Oct-Dec;   bad = 11.0 in Apr-Sep
    CC: good = 9.0 in April only; edge, bad = 5.0 always (13 near months < 30 -> dropped)

    Hand-derived per-year values (identical in every scored year, so every MCS step is a certain
    rejection or an exact tie -- see mcs_test.mcs()'s se == 0 branches):
      near months per year: Apr-Dec = 9 (AA, BB)  -> 13 x 9 = 117
      AA decision loss on near months: good 0/9, edge 3/9, bad 6/9   -> dMCS = {good}
      AA stat loss (|fc - u|, full 12-month years): good 6x4/12 = 2.0, edge 3x6/12 = 1.5,
         bad 6x6/12 = 3.0 -> MCS_stat = {edge}; on near months: good 24/9, edge 18/9, bad 36/9
         -> MCS_stat_near = {edge}
      BB decision loss: good 0, edge 0 (exact tie), bad 6/9 -> dMCS = {edge, good} (size 2)
      BB stat loss: good 2.0, edge 3x4/12 = 1.0, bad 3.0 -> {edge}; near: 24/9, 12/9, 36/9 -> {edge}
    """
    idx = pd.date_range("1978-01-01", "2005-12-01", freq="MS")
    month = idx.month

    def fc(hi_months, hi_val):
        s = pd.Series(5.0, index=idx)
        s[np.isin(month, hi_months)] = hi_val
        return s

    apr_sep, oct_dec = [4, 5, 6, 7, 8, 9], [10, 11, 12]
    nowcasts = {
        "AA": {"good": fc(apr_sep, 9.0), "edge": fc(oct_dec, 11.0), "bad": fc(apr_sep, 11.0)},
        "BB": {"good": fc(apr_sep, 9.0), "edge": fc(oct_dec, 9.0), "bad": fc(apr_sep, 11.0)},
        "CC": {"good": fc([4], 9.0), "edge": fc([], 5.0), "bad": fc([], 5.0)},
    }
    df = pd.DataFrame({"AA": pd.Series(5.0, index=idx), "BB": pd.Series(5.0, index=idx),
                       "CC": pd.Series(4.9, index=idx)})
    return df, nowcasts


def _run_dmcs_main(monkeypatch, tmp_path):
    df, nowcasts = _synthetic_state_panel()
    monkeypatch.setattr(SEB, "load_state_ur", lambda: df)
    monkeypatch.setattr(SEB, "nowcast", lambda u: nowcasts[u.name])
    monkeypatch.setattr(SEB, "MODELS", {"good": None, "edge": None, "bad": None})
    store = {}
    monkeypatch.setattr(dmcs, "emit", lambda key, value, **kw: store.__setitem__(key, (value, kw)))
    monkeypatch.setattr(dmcs, "flush", lambda: None)
    monkeypatch.setattr(dmcs, "RES", tmp_path)
    dmcs.main()
    R = pd.read_csv(tmp_path / "dmcs.csv").set_index("state")
    for col in ("stat_set", "decision_set", "stat_set_near"):
        R[col] = R[col].apply(ast.literal_eval)
    return R, store


def test_dmcs_main_near_trigger_band_includes_its_upper_edge_and_decision_loss_is_misclassification(monkeypatch, tmp_path):
    """kills: M9 (near band `(A <= 7.0)` -> `(A < 7.0)`), M10 (`!= realized_eb` -> `==`).

    Under M9 the Oct-Dec months (near only through edge's exact 7.0) drop out: AA's near count
    falls from 117 to 78 and, with edge's misclassifications no longer scored, good and edge tie
    under the decision loss (dMCS_decision 2, not 1). Under M10 the decision loss becomes an
    AGREEMENT score, so AA's decision set flips from {good} to {bad} and BB's from {edge, good}
    to {bad}. Every value below is hand-derived in _synthetic_state_panel's docstring."""
    R, _store = _run_dmcs_main(monkeypatch, tmp_path)

    assert list(R.index) == ["AA", "BB"]                       # CC dropped by the 30-near-month floor
    for s in ("AA", "BB"):
        assert R.loc[s, "near_trigger_months"] == 117
        assert R.loc[s, "near_trigger_years"] == 13 and R.loc[s, "full_years"] == 13
        assert R.loc[s, "MCS_stat"] == 1 and R.loc[s, "stat_set"] == ["edge"]
        assert R.loc[s, "MCS_stat_near"] == 1 and R.loc[s, "stat_set_near"] == ["edge"]
        assert not R.loc[s, "separates"] and not R.loc[s, "separates_symmetric"]
    assert R.loc["AA", "dMCS_decision"] == 1 and R.loc["AA", "decision_set"] == ["good"]
    assert R.loc["BB", "dMCS_decision"] == 2 and R.loc["BB", "decision_set"] == ["edge", "good"]


def test_dmcs_main_emits_hand_derived_counts_for_the_synthetic_panel(monkeypatch, tmp_path):
    """kills: M9, M10 (through the emitted counts), and pins every dmcs.* aggregate against the
    hand-derived two-state outcome: AA equal (1 = 1), BB wider from a clean singleton (2 > 1, and
    also a singleton under the full-sample statistical leg); CC eligible but dropped."""
    _, store = _run_dmcs_main(monkeypatch, tmp_path)
    v = {k: val for k, (val, _) in store.items()}

    assert v["dmcs.n_eligible"] == 3 and v["dmcs.n_states"] == 2
    assert v["dmcs.mean_set_statistical"] == pytest.approx(1.0)
    assert v["dmcs.mean_set_statistical_near"] == pytest.approx(1.0)
    assert v["dmcs.mean_set_decision"] == pytest.approx(1.5)
    assert v["dmcs.separates"] == 0 and v["dmcs.separates_symmetric"] == 0
    assert v["dmcs.single_decision"] == 1
    assert v["dmcs.tied_both"] == 1 and v["dmcs.tied_both_symmetric"] == 1
    assert v["dmcs.equal_symmetric"] == 1 and v["dmcs.wider_symmetric"] == 1
    assert v["dmcs.wider_from_singleton"] == 1
    assert store["dmcs.wider_from_singleton"][1]["of"] == 1
    assert store["dmcs.wider_from_singleton"][1]["states"] == "BB"
    assert v["dmcs.wider_from_singleton_also_full_sample"] == 1
    assert store["dmcs.wider_from_singleton_also_full_sample"][1]["states"] == "BB"
    assert v["dmcs.dropped_near_trigger_states"] == 1
    assert store["dmcs.dropped_near_trigger_states"][1]["states"] == "CC"
    assert store["dmcs.lowest_mean_ur_states"][1]["states"] == "AA;BB;CC"
    assert v["dmcs.dropped_lowest_ur_overlap"] == 1
    assert store["dmcs.dropped_lowest_ur_overlap"][1]["states"] == "CC"
