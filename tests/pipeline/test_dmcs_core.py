"""dmcs.py IS touched by existing tests (test_dmcs.py, test_dmcs_near_trigger.py,
test_dmcs_symmetric_robustness_cut.py), but every one of them either re-derives SEB.eb_on()'s call
shape against real state-UR data, or asserts structural invariants on the COMMITTED dmcs.csv/
canonical.json artifacts. None of them hand-verifies dmcs.py's own actual computational core --
`year_mcs()`, the function that turns a {model: monthly-loss-Series} dict + a `common` DatetimeIndex
into a per-year-mean loss matrix and hands it to mcs_test.mcs() -- against an independently
hand-computed expected answer. A bug in year_mcs()'s reindex/dropna/groupby-year-mean plumbing
(e.g. reindexing against the wrong index, dropping rows column-wise instead of row-wise, or grouping
by the wrong index level) could silently corrupt every dMCS_decision/MCS_stat cell in dmcs.csv while
every existing test stayed green (they only check the eb_on() call site and the CSV's own internal
consistency, never re-derive year_mcs()'s output from first principles).

This file closes that gap with synthetic, hand-computable inputs (mcs_test.mcs() itself already has
real hand-verified coverage elsewhere -- test_mcs_test_degenerate_se.py, test_mcs_size_level.py --
so this file targets year_mcs()'s OWN plumbing, not mcs()'s statistics).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import dmcs


# --- length floor: `if len(M) < 30: return None`, evaluated AFTER reindex+dropna -----------------

def test_year_mcs_returns_none_below_thirty_row_floor_but_not_at_exactly_thirty():
    common_29 = pd.date_range("2000-01-01", periods=29, freq="MS")
    loss_by_model = {"A": pd.Series(1.0, index=common_29), "B": pd.Series(2.0, index=common_29)}
    assert dmcs.year_mcs(loss_by_model, common_29) is None

    # exactly 30 rows, no NaNs to drop -> must NOT hit the None branch (the floor is `< 30`, not
    # `<= 30`). Models made IDENTICAL so the statistical result itself is a trivial tie (both
    # survive) -- this assertion is purely about the length gate, not mcs()'s own statistics.
    common_30 = pd.date_range("2000-01-01", periods=30, freq="MS")
    loss_by_model_30 = {"A": pd.Series(np.arange(30, dtype=float), index=common_30),
                         "B": pd.Series(np.arange(30, dtype=float), index=common_30)}
    result = dmcs.year_mcs(loss_by_model_30, common_30)
    assert result is not None
    assert result == {"A", "B"}


# --- reindex(common) + row-wise dropna() + groupby(year).mean(), hand-verified via a spy ---------

def test_year_mcs_reindex_drops_out_of_common_dates_and_dropna_is_row_wise(monkeypatch):
    """40 consecutive months (2000-01 .. 2003-04, spanning years 2000/2001/2002/2003). A has an
    EXTRA date (1999-12, outside `common`) that reindex(common) must drop entirely -- if it leaked
    through, mean-by-year would show a spurious 5th year (1999) instead of 4. B is NaN at index 3
    (2000-04) -- pandas DataFrame.dropna() with no subset= drops that ROW for BOTH columns, so A's
    real value at that date must ALSO be excluded from the 2000 mean, not just B's.

    Spies on mcs_test.MCS.mcs() to capture the exact (years x models) matrix year_mcs() builds and
    hands it, and compares every cell against an independently hand-computed expected value -- the
    thing no existing dmcs.py test does."""
    common = pd.date_range("2000-01-01", periods=40, freq="MS")   # 2000:12, 2001:12, 2002:12, 2003:4
    a_index = pd.DatetimeIndex(["1999-12-01"]).append(common)      # extra out-of-common date, must be dropped
    a_vals = np.concatenate([[999.0], np.arange(1, 41, dtype=float)])   # A[i] = i+1 for i=0..39
    a = pd.Series(a_vals, index=a_index)

    b_vals = 1000.0 + np.arange(40, dtype=float)                   # B[i] = 1000+i for i=0..39
    b_vals[3] = np.nan                                              # NaN at 2000-04 (index 3)
    b = pd.Series(b_vals, index=common)

    loss_by_model = {"A": a, "B": b}   # dict order -> M.columns order = ["A", "B"]

    captured = {}
    def spy_mcs(L, alpha=0.10, seed=0):
        captured["L"] = np.asarray(L, dtype=float).copy()
        return [0, 1], 1.0   # both "survive" -- the return value isn't under test here
    monkeypatch.setattr(dmcs.MCS, "mcs", spy_mcs)

    result = dmcs.year_mcs(loss_by_model, common, seed=0)
    assert result == {"A", "B"}
    assert "L" in captured, "year_mcs() never reached MCS.mcs() -- did the length floor fire unexpectedly?"

    yr = captured["L"]
    assert yr.shape == (4, 2), (
        "expected exactly 4 year-rows (2000-2003) -- a 5th row would mean the out-of-common "
        "1999-12 date leaked through reindex(common)"
    )
    # Year 2000: indices 0..11 EXCLUDING index 3 (dropped row) -> A values {1,2,3,5,...,12}, 11 obs.
    expected_2000_a = (sum(range(1, 13)) - 4) / 11
    expected_2000_b = (11 * 1000 + (sum(range(0, 12)) - 3)) / 11
    # Year 2001: indices 12..23, all present -> A values 13..24, B values 1012..1023.
    expected_2001_a = (13 + 24) / 2
    expected_2001_b = 1000 + (12 + 23) / 2
    # Year 2002: indices 24..35, all present -> A values 25..36, B values 1024..1035.
    expected_2002_a = (25 + 36) / 2
    expected_2002_b = 1000 + (24 + 35) / 2
    # Year 2003: indices 36..39 (4 months only), all present -> A values 37..40, B values 1036..1039.
    expected_2003_a = (37 + 40) / 2
    expected_2003_b = 1000 + (36 + 39) / 2

    expected = np.array([
        [expected_2000_a, expected_2000_b],
        [expected_2001_a, expected_2001_b],
        [expected_2002_a, expected_2002_b],
        [expected_2003_a, expected_2003_b],
    ])
    np.testing.assert_allclose(yr, expected, rtol=1e-12,
        err_msg="year_mcs()'s per-year-mean matrix does not match the hand-computed expectation -- "
                "either the out-of-common date leaked through, or dropna() is not dropping the "
                "NaN row for BOTH columns (row-wise), or the groupby-year-mean itself is wrong")


# --- full-stack: a real, certainly-dominant model must be the sole survivor ----------------------

def test_year_mcs_survivor_set_hand_verified_constant_dominant_model():
    """8 distinct years (>= mcs_test.MIN_T_FOR_ZERO_SE=8), A constant loss 0.0 vs B constant loss
    5.0 every single month. A CONSTANT non-zero per-period differential across >= MIN_T_FOR_ZERO_SE
    periods is mcs()'s own documented 'certain rejection' case (t_obs -> inf) -- B must be
    eliminated and A alone must survive. Runs the REAL mcs_test.mcs(), not a spy: this is the
    full-stack sanity check that year_mcs()'s plumbing feeds mcs() a matrix that actually produces
    the textbook-obvious answer, not just a plausible-shaped one."""
    common = pd.date_range("2010-01-01", periods=96, freq="MS")   # 8 years x 12 months
    loss_by_model = {"A": pd.Series(0.0, index=common), "B": pd.Series(5.0, index=common)}
    result = dmcs.year_mcs(loss_by_model, common, seed=0)
    assert result == {"A"}, f"expected the certainly-better model A alone to survive, got {result}"


def test_year_mcs_ties_are_preserved_when_models_identical():
    """A and B have IDENTICAL (non-constant, so this isn't the degenerate-se floor test above)
    losses every period -> the equal-predictive-ability null is never rejected (obs_diff==0 always)
    -> both models must survive as a genuine tie, per mcs()'s own docstring ('a genuine exact tie...
    is correctly uninformative and is skipped')."""
    common = pd.date_range("2010-01-01", periods=40, freq="MS")
    vals = np.sin(np.arange(40, dtype=float)) + 3.0   # non-constant but identical across models
    loss_by_model = {"A": pd.Series(vals, index=common), "B": pd.Series(vals, index=common)}
    result = dmcs.year_mcs(loss_by_model, common, seed=0)
    assert result == {"A", "B"}
