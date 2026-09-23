"""Hand-verified coverage of score_multiseries.py's rescore_series() arithmetic.

score_multiseries.py IS touched by an existing test (test_scoring_loop_parity.py), but that test is
a PARITY check: it feeds run_multiseries.run_series() and score_multiseries.rescore_series() the
SAME synthetic detail panel and asserts the two functions' outputs agree with EACH OTHER on 13
shared fields. That is real coverage for the specific duplication-drift risk it targets, but it
never independently hand-verifies rescore_series()'s own arithmetic against a value computed outside
the module -- if BOTH functions shared the identical bug (e.g. both scaling by the wrong column, or
both re-applying an extra sqrt to an already-RMSSE'd value -- a "double-square-root" defect: RMSSE's
sqrt is applied per-year inside walkforward.agg_metric(), so a second `** 0.5` anywhere downstream
would silently double it), the parity test would stay green while the actual numbers were wrong.

This file hand-verifies rescore_series() end-to-end against independently-derived expected values:
sae/msse computed from the correct (not swapped) scale_abs/scale_sq, RMSSE's sqrt-per-year-BEFORE-
pooling order of operations (walkforward.agg_metric()'s actual computation), fragility()'s
pooled_winner/k_star on a hand-derivable constant-dominance panel, and the NaN-scale row-drop
behavior.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import score_multiseries as SM


def _pred_rows(series, model, years, y_true, y_pred, rows_per_year=12):
    """`rows_per_year` identical (year, y_true, y_pred) rows per year -- rescore_series() only reads
    series/model/year/y_true/y_pred; target is included for realism but unused by the function."""
    rows = []
    for yr in years:
        for k in range(rows_per_year):
            rows.append(dict(series=series, model=model, year=yr,
                              target=pd.Timestamp(f"{yr}-01-01") + pd.DateOffset(months=k),
                              y_true=y_true, y_pred=y_pred))
    return rows


def _fake_year_scales(scale_abs, scale_sq):
    """Returns a WF.year_scales replacement that hands back a CONSTANT scale per requested year,
    distinguishing the abs/sq legs by the `squared` kwarg -- exactly the two legs rescore_series()
    calls separately (`sc_abs = WF.year_scales(..., squared=False, ...)` /
    `sc_sq = WF.year_scales(..., squared=True, ...)`). Deliberately DISTINCT values so a scale-swap
    bug (abs<->sq mixed up in the sae/msse computation) is numerically detectable."""
    def fn(truth, years, m=12, squared=False, trail=None):
        val = scale_sq if squared else scale_abs
        return {y: val for y in years}
    return fn


def _patch_common(monkeypatch, scale_abs=2.0, scale_sq=4.0, n_boot=50):
    monkeypatch.setattr(SM, "N_BOOT", n_boot)   # speed only -- seed=0 is fixed at the call site
    monkeypatch.setattr(SM.WF, "load_vintages", lambda sid: None)
    monkeypatch.setattr(SM.WF, "realized", lambda v, sid="", guard_before=None: None)   # mirrors WF.realized, including guard_before
    monkeypatch.setattr(SM.WF, "year_scales", _fake_year_scales(scale_abs, scale_sq))


# --- sae/msse must use the CORRECT (not swapped) scale, pooled_winner hand-verified --------------

def test_rescore_series_sae_msse_correct_scale_and_pooled_winner(monkeypatch):
    """8 years, 12 rows/year. Model A: constant error 2.0 every row. Model B: constant error 6.0.
    scale_abs=2.0, scale_sq=4.0 (deliberately DIFFERENT so a scale-swap bug is detectable):
      sae_A = 2/2 = 1.0 every row  -> MASE pooled loss for A = 1.0
      sae_B = 6/2 = 3.0 every row  -> MASE pooled loss for B = 3.0
      msse_A = 2^2/4 = 1.0 every row -> per-year mean=1.0 -> sqrt=1.0 -> RMSSE pooled loss for A = 1.0
      msse_B = 6^2/4 = 9.0 every row -> per-year mean=9.0 -> sqrt=3.0 -> RMSSE pooled loss for B = 3.0
    A must be pooled_winner under BOTH metrics, with winner_pooled_loss == 1.0 (not 3.0, and not the
    scale-swapped alternative 2.0/2.83.. that a scale_abs<->scale_sq mixup would produce).
    """
    _patch_common(monkeypatch, scale_abs=2.0, scale_sq=4.0)
    years = list(range(2015, 2023))   # 8 years
    pred = pd.DataFrame(_pred_rows("S1", "A", years, y_true=100.0, y_pred=102.0)
                         + _pred_rows("S1", "B", years, y_true=100.0, y_pred=106.0))

    rows = SM.rescore_series("S1", pred)
    by_metric = {r["metric"]: r for r in rows}
    assert set(by_metric) == {"mase", "rmsse"}

    tag = f"trail{SM.TRAIL}" if SM.TRAIL else "expanding"
    for metric in ("mase", "rmsse"):
        r = by_metric[metric]
        assert r["series"] == "S1"
        assert r["scale"] == tag
        assert r["pooled_winner"] == "A", f"{metric}: expected A (lower loss) to win, got {r['pooled_winner']}"
        assert r["winner_pooled_loss"] == pytest.approx(1.0, abs=1e-6), (
            f"{metric}: winner_pooled_loss={r['winner_pooled_loss']!r} != 1.0 -- "
            f"suggests scale_abs/scale_sq got swapped in the sae/msse computation"
        )
        assert r["n_years"] == 8


# --- fragility()'s k_star/reversal, hand-verified on the SAME constant-dominance panel -----------

def test_rescore_series_k_star_and_reversal_hand_verified(monkeypatch):
    """Same construction as the previous test: A constant-beats B in every single one of 8 years.
    Every period is then a-favoring (c[i] > 0), so breakdown_number()'s greedy loop
    (fragility.py:114-121) never reaches its `c[i] <= 0` break and removes all T of them --
    k* = T, the maximally robust case. Hand-verified: k_star == n_years == 8,
    k_over_T == 1.0, reversal == False (A wins every period, so pooled winner == per-period winner),
    fragile == False (k_over_T is nowhere near the FRAGILE_KSTAR_RATIO_THRESH=0.25 floor, and A's
    dominance is deterministic so winner_stability must be ~1.0, well above FRAGILE_STABILITY_THRESH
    =0.60)."""
    _patch_common(monkeypatch, scale_abs=2.0, scale_sq=4.0)
    years = list(range(2015, 2023))
    pred = pd.DataFrame(_pred_rows("S1", "A", years, y_true=100.0, y_pred=102.0)
                         + _pred_rows("S1", "B", years, y_true=100.0, y_pred=106.0))

    rows = SM.rescore_series("S1", pred)
    for r in rows:
        assert r["k_star"] == 8, f"{r['metric']}: expected k_star==n_years==8 (maximal robustness), got {r['k_star']}"
        assert r["k_over_T"] == pytest.approx(1.0)
        assert r["reversal"] is False
        assert r["fragile"] is False
        assert r["winner_stability"] == pytest.approx(1.0, abs=1e-6)


# --- RMSSE: sqrt applied PER YEAR before pooling, not to the pooled mean (no double-sqrt) --------

def test_rescore_series_rmsse_sqrt_applied_per_year_not_to_pooled_mean(monkeypatch):
    """Locks in the exact historical defect class this guards against: walkforward.agg_metric()
    applies sqrt to the per-year mean(msse) itself (`out["loss"] = np.sqrt(out["loss"])` when
    metric=="rmsse"), so any downstream code that applied a second `** 0.5` to that already-square-
    rooted loss would silently double-square-root it. Model A: err=2.0 in
    year 1, err=6.0 in year 2 (scale_sq=4.0 constant, 1 row/year so the per-year mean is just that
    single value):
      msse_2015 = 4/4 = 1.0  -> sqrt = 1.0
      msse_2016 = 36/4 = 9.0 -> sqrt = 3.0
      correct pooled RMSSE loss for A (mean of the two PER-YEAR sqrts, uniform weight) = (1.0+3.0)/2 = 2.0
    A WRONG implementation that pools msse first then sqrts once (sqrt(mean(1.0, 9.0)) = sqrt(5) =
    2.236) or that double-square-roots (sqrt(2.0) = 1.414) would both fail this exact assertion.
    Model B is a large constant-error decoy (always loses) purely so fragility() has >=2 models.
    """
    _patch_common(monkeypatch, scale_abs=2.0, scale_sq=4.0)
    pred = pd.DataFrame(
        [dict(series="S1", model="A", year=2015, target=pd.Timestamp("2015-06-01"), y_true=100.0, y_pred=102.0),
         dict(series="S1", model="A", year=2016, target=pd.Timestamp("2016-06-01"), y_true=100.0, y_pred=106.0),
         dict(series="S1", model="B", year=2015, target=pd.Timestamp("2015-06-01"), y_true=100.0, y_pred=200.0),
         dict(series="S1", model="B", year=2016, target=pd.Timestamp("2016-06-01"), y_true=100.0, y_pred=200.0)]
    )
    rows = SM.rescore_series("S1", pred)
    rmsse_row = next(r for r in rows if r["metric"] == "rmsse")
    assert rmsse_row["pooled_winner"] == "A"
    assert rmsse_row["winner_pooled_loss"] == pytest.approx(2.0, abs=1e-6), (
        f"got {rmsse_row['winner_pooled_loss']!r}; expected 2.0 (mean of per-year sqrts 1.0 and 3.0). "
        f"2.236 would mean sqrt is applied to the POOLED mean instead of per-year; 1.414 would mean "
        f"an extra sqrt is applied on top of agg_metric()'s own per-year sqrt (double-square-root)"
    )


# --- rows whose year has no valid scale (NaN) must be dropped, not silently coerced --------------

def test_rescore_series_drops_rows_with_nan_scale_for_a_year(monkeypatch):
    """3 years of data, but WF.year_scales() reports NaN for the middle year (as real year_scales()
    does whenever a year has too few pre-period observations) -- rescore_series()'s own
    `g.dropna(subset=["scale_abs", "scale_sq"])` must exclude every row from that year entirely, so
    n_years reflects only the 2 valid years, not 3, and the excluded year's rows must not silently
    leak in as sae=inf/NaN."""
    monkeypatch.setattr(SM, "N_BOOT", 50)
    monkeypatch.setattr(SM.WF, "load_vintages", lambda sid: None)
    monkeypatch.setattr(SM.WF, "realized", lambda v, sid="", guard_before=None: None)   # mirrors WF.realized, including guard_before

    def fake_year_scales(truth, years, m=12, squared=False, trail=None):
        val = 4.0 if squared else 2.0
        return {y: (np.nan if y == 2016 else val) for y in years}
    monkeypatch.setattr(SM.WF, "year_scales", fake_year_scales)

    years = [2015, 2016, 2017]
    pred = pd.DataFrame(_pred_rows("S1", "A", years, y_true=100.0, y_pred=102.0, rows_per_year=4)
                         + _pred_rows("S1", "B", years, y_true=100.0, y_pred=106.0, rows_per_year=4))

    rows = SM.rescore_series("S1", pred)
    for r in rows:
        assert r["n_years"] == 2, f"{r['metric']}: expected 2016 (NaN scale) dropped, n_years={r['n_years']}"
