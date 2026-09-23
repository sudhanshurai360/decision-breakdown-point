"""density_audit.py is touched by existing tests
(test_density_audit_fetches_data.py checks main() calls MA.fetch() not a hardcoded path;
test_warn_fallback_contract.py checks the fallback-warning print-once contract;
test_classical_forecaster_formulas_agree.py checks f_ses()/f_damped()'s minimum-length GATE
boundary agrees with sibling files) -- but nothing anywhere hand-verifies the actual VALUES these
functions compute, and nothing hand-verifies the file's core mechanism: the MAE-vs-RMSE
winner-FLIP rate over random k-subsets of the leaderboard, which is the entire "leaderboard-density"
law this producer exists to measure (its module docstring calls it "the LEADERBOARD-DENSITY curve
on M4-Monthly"). f_drift() and f_mean() in particular are referenced nowhere else in tests/.

This file closes that gap two ways:
  1. Direct hand-verified value tests for f_drift/f_mean/f_ses/f_damped (real production functions,
     called directly -- no monkeypatching needed).
  2. An end-to-end test of main()'s flip-rate computation at the FULL-model-pool boundary
     (k == n_models), where rng.choice(names, k, replace=False) has only one possible subset
     CONTENT regardless of the random draw (only its internal ORDER is randomized) -- engineered so
     that the flip outcome is deterministic and hand-derivable *despite* that residual randomness
     (see the docstring on that test for the argument), letting this test run the real main()
     end-to-end against tiny synthetic data and assert an exact expected flip_rate.
"""
import sys

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES

import density_audit as DA
import m_audit as MA


# =================================================================================================
# f_drift -- linear extrapolation from the first to the last observation. ZERO prior test coverage.
# =================================================================================================

def test_f_drift_hand_computed_linear_extrapolation():
    y = np.array([10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0])  # len=8, y[0]=10, y[-1]=18
    # Hand-derive independently of the function: slope = (last - first) / (n - 1), then
    # forecast = last + slope * step, for step = 1..h.
    slope = (18.0 - 10.0) / (8 - 1)
    expected = np.array([18.0 + slope * 1, 18.0 + slope * 2, 18.0 + slope * 3])
    got = DA.f_drift(y, m=12, h=3)
    assert got == pytest.approx(expected)
    # Sanity-check the hand arithmetic against an independently-computed decimal too.
    assert expected == pytest.approx([134 / 7, 142 / 7, 150 / 7])


def test_f_drift_len_below_2_falls_back_to_flat_last_value():
    """Guard branch: `if len(y) < 2: return np.repeat(y[-1], h)` -- a single-observation series has
    no defined slope, so drift must degrade to flat persistence, not divide by zero."""
    y = np.array([5.0])
    got = DA.f_drift(y, m=12, h=4)
    assert got == pytest.approx([5.0, 5.0, 5.0, 5.0])


# =================================================================================================
# f_mean -- mean of a trailing window (2*m if m>1 else 6, capped at len(y)). ZERO prior coverage.
# =================================================================================================

def test_f_mean_hand_computed_full_window_when_series_shorter_than_2m():
    """m=12 (monthly) -> window = min(len(y), 24). len(y)=8 < 24, so the window is the WHOLE
    series -- hand-computed mean of all 8 values."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    expected_mean = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) / 8  # = 4.5
    got = DA.f_mean(y, m=12, h=2)
    assert got == pytest.approx([expected_mean, expected_mean])


def test_f_mean_hand_computed_trailing_window_of_6_when_m_equals_1():
    """m=1 (non-seasonal branch) -> window = min(len(y), 6): the LAST 6 values only, not the full
    10-observation series."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])  # len=10
    last6 = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    expected_mean = sum(last6) / 6  # = 7.5
    got = DA.f_mean(y, m=1, h=2)
    assert got == pytest.approx([expected_mean, expected_mean])


# =================================================================================================
# f_ses / f_damped -- ZERO prior VALUE coverage (only the flat-fallback GATE boundary, a bool, was
# tested elsewhere). Two regimes: len(y) < 10 (documented flat fallback) and len(y) >= 10 (real
# statsmodels fit).
# =================================================================================================

def test_f_ses_and_f_damped_below_length_10_return_the_exact_flat_value_not_just_a_flat_shape():
    """Both functions' own length gate (`len(y) < 10`) must degrade to np.repeat(y[-1], h) EXACTLY
    -- not merely "a constant array" but the specific last observed value."""
    y = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 8.0])  # len=9 < 10
    assert DA.f_ses(y, m=12, h=3) == pytest.approx([8.0, 8.0, 8.0])
    assert DA.f_damped(y, m=12, h=3) == pytest.approx([8.0, 8.0, 8.0])


def test_f_ses_and_f_damped_real_fit_on_a_constant_series_forecasts_the_constant():
    """len(y)=12 >= 10 -> both take the REAL statsmodels-fit branch (not the flat fallback). For a
    perfectly constant input series (no trend, no noise), simple *and* damped-trend exponential
    smoothing must both converge to level == that constant and forecast it flat for every horizon
    -- a value derivable from the method's definition (no information exists in a constant series to
    fit a nonzero trend or move the level away from it), independent of calling the function itself.
    Cross-checked directly against statsmodels below so this isn't just "trust the module under
    test": it fits the identical statsmodels call by hand and compares."""
    y = np.repeat(100.0, 12)
    ses_got = DA.f_ses(y, m=12, h=3)
    damped_got = DA.f_damped(y, m=12, h=3)
    assert ses_got == pytest.approx([100.0, 100.0, 100.0], abs=1e-6)
    assert damped_got == pytest.approx([100.0, 100.0, 100.0], abs=1e-6)
    # Independent re-derivation: fit the same statsmodels calls directly in the test.
    ses_direct = np.asarray(ES(y, initialization_method="heuristic").fit().forecast(3), float)
    damped_direct = np.asarray(
        ES(y, trend="add", damped_trend=True, initialization_method="heuristic").fit().forecast(3),
        float,
    )
    assert ses_got == pytest.approx(ses_direct)
    assert damped_got == pytest.approx(damped_direct)


# =================================================================================================
# main()'s core mechanism: the MAE-vs-RMSE winner-flip rate over random k-subsets of the leaderboard.
# Never previously hand-verified anywhere (existing coverage only checks main() reads data via
# MA.fetch(), and separately that arena_crowding.py's OWN, DIFFERENT crowd()/arena() functions
# produce values in [0, 1] -- neither touches density_audit.main()'s own flip-rate loop at all).
# =================================================================================================

def test_main_flip_rate_at_full_model_pool_matches_hand_derived_value(tmp_path, monkeypatch):
    """Runs the REAL main() end-to-end against tiny synthetic M4-shaped CSVs, with MA.MODELS
    monkeypatched to {} so the only models in play are density_audit's own f_drift/f_mean/f_ses/
    f_damped (all real production code, all exactly hand-computable for a length-9 train series,
    since length 9 < 10 forces f_ses/f_damped's flat-fallback branch).

    Why this is exact despite `rng.choice(names, k, replace=False)` being random: at k == n_models
    (here 4), every draw's SUBSET CONTENT is forced (there is only one 4-of-4 subset) -- only the
    internal ORDER of `sub` is randomized. The two synthetic series below are engineered so the
    flip outcome does not depend on that order:
      * series "s1": ses and damped produce IDENTICAL forecasts (both fall back to flat y[-1]), so
        they are EXACTLY tied under both MAE and RMSE. That tied value (2.033) is the unique MAE
        minimum (below drift's 2.633 and mean's 6.033) but NOT the RMSE minimum (drift's 2.910 is
        lower than the ses/damped tie's 2.944) -- so MAE's winner is always one of {ses, damped}
        (never drift) while RMSE's winner is always drift, a flip REGARDLESS of which of the tied
        pair `min()` happens to pick first.
      * series "s2": drift's forecast exactly matches the actual (0 error), a strict, untied minimum
        under both MAE and RMSE -- never a flip, regardless of subset order.
    With 6 repeats per series (main()'s own fixed inner loop), the k=4 row's flips list is exactly
    [True]*6 (s1) + [False]*6 (s2) -> flip_rate == 6/12 == 0.5 exactly, every run, every seed.
    """
    # Train: two series, identical 9-point ramp 10..18 (len=9 -> f_ses/f_damped's flat-fallback
    # branch, so all 4 models are simple closed-form functions of y).
    tr_df = pd.DataFrame({
        "id": ["s1", "s2"],
        "t1": [10.0, 10.0], "t2": [11.0, 11.0], "t3": [12.0, 12.0], "t4": [13.0, 13.0],
        "t5": [14.0, 14.0], "t6": [15.0, 15.0], "t7": [16.0, 16.0], "t8": [17.0, 17.0],
        "t9": [18.0, 18.0],
    })
    # Test/actual: h=3 each, engineered per the docstring above.
    te_df = pd.DataFrame({
        "id": ["s1", "s2"],
        "t1": [23.0, 19.0], "t2": [19.0, 20.0], "t3": [18.1, 21.0],
    })
    tr_f = tmp_path / "train.csv"
    te_f = tmp_path / "test.csv"
    tr_df.to_csv(tr_f, index=False)
    te_df.to_csv(te_f, index=False)

    monkeypatch.setattr(MA, "MODELS", {})  # only drift/mean/ses/damped remain -> 4 models total
    monkeypatch.setattr(MA, "fetch", lambda freq: (tr_f, te_f))
    monkeypatch.setattr(MA, "RESULTS", tmp_path)
    # n=2 (!= the documented default 1500) so main() takes its own "skip canonical emit" branch --
    # avoids needing to also stub canonical_emit's registry machinery for this test.
    monkeypatch.setattr(sys, "argv", ["density_audit.py", "2"])

    DA.main()

    out = pd.read_csv(tmp_path / "m4_density_curve.csv")
    row = out.loc[out.n_models == 4]
    assert len(row) == 1, f"expected exactly one n_models==4 row, got: {out}"
    assert row.flip_rate.iloc[0] == pytest.approx(0.5), (
        f"hand-derived expected flip_rate at the full 4-model pool is exactly 0.5 (6/12 -- see "
        f"this test's docstring); got {row.flip_rate.iloc[0]} -- full curve:\n{out}"
    )
    # Also pin the two hand-derivable per-model forecast/error facts the flip argument depends on,
    # so a change to f_drift/f_mean/f_ses/f_damped that happens to preserve flip_rate==0.5 by
    # accident (e.g. a compensating bug) still gets caught here.
    y_train = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
    assert DA.f_drift(y_train, 12, 3) == pytest.approx([19.0, 20.0, 21.0])
    assert DA.f_mean(y_train, 12, 3) == pytest.approx([14.0, 14.0, 14.0])
    assert DA.f_ses(y_train, 12, 3) == pytest.approx([18.0, 18.0, 18.0])
    assert DA.f_damped(y_train, 12, 3) == pytest.approx([18.0, 18.0, 18.0])
