"""kstar_persistence.py's own module docstring poses the question it exists to answer: "does k*
predict anything a practitioner cares about?" -- via split-half chronological persistence testing.
`main()` itself does file I/O (reads breadth_distinct_predictions.csv, calls fred_md_sweep.py's loaders, calls emit())
and is exercised end-to-end by a full producer rerun; this file instead unit-tests the two pure,
independently-testable computations -- `split_half()` (the chronological midpoint split) and
`analyze_series()` (the first-half k*/second-half persistence arithmetic) -- against hand-derived
expected values and the file's own documented MIN_T_PER_HALF exclusion guard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code" / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code" / "instrument"))

import numpy as np
import pandas as pd
import pytest

import kstar_persistence as KP
from kstar_persistence import split_half, analyze_series, MIN_T_PER_HALF


# --- split_half(): the chronological midpoint boundary -------------------------------------
# The off-by-one boundary itself (not just "it runs") is what these lock in.

def test_split_half_odd_T_puts_the_extra_year_in_the_first_half():
    """T=15 -> ceil(15/2)=8 in the first half, 7 in the second (not 7/8) -- the exact convention
    split_half()'s own docstring states ('first ceil(T/2) years vs the rest')."""
    T = 15
    yrs = list(range(2000, 2000 + T))
    La = np.arange(T * 2, dtype=float).reshape(T, 2)  # 2 dummy models, distinct values per row
    La1, La2, yrs1, yrs2 = split_half(La, yrs)
    assert len(yrs1) == 8 and len(yrs2) == 7
    assert yrs1 == yrs[:8] and yrs2 == yrs[8:]
    assert La1.shape == (8, 2) and La2.shape == (7, 2)
    np.testing.assert_array_equal(La1, La[:8])
    np.testing.assert_array_equal(La2, La[8:])


def test_split_half_even_T_splits_exactly_in_half():
    T = 14
    yrs = list(range(2010, 2010 + T))
    La = np.arange(T * 3, dtype=float).reshape(T, 3)
    La1, La2, yrs1, yrs2 = split_half(La, yrs)
    assert len(yrs1) == 7 and len(yrs2) == 7
    np.testing.assert_array_equal(La1, La[:7])
    np.testing.assert_array_equal(La2, La[7:])


def test_split_half_minimum_T_boundary_matches_MIN_T_PER_HALF_convention():
    """T = 2*MIN_T_PER_HALF (currently 16) is the smallest T that clears the >=8-per-half gate on
    BOTH halves for an even split; confirm split_half itself gives exactly MIN_T_PER_HALF years on
    each side here (analyze_series's own inclusion boundary, exercised below, depends on this)."""
    T = 2 * MIN_T_PER_HALF
    yrs = list(range(1990, 1990 + T))
    La = np.zeros((T, 1))
    La1, La2, yrs1, yrs2 = split_half(La, yrs)
    assert len(yrs1) == MIN_T_PER_HALF
    assert len(yrs2) == MIN_T_PER_HALF


# --- analyze_series(): the MIN_T_PER_HALF exclusion guard -----------------------------------

def test_analyze_series_excludes_when_either_half_is_below_min_T():
    """T=10 -> T1=5, T2=5, both below MIN_T_PER_HALF=8. Confirm the exact excluded-row contract
    coded in kstar_persistence.py's own analyze_series() (all derived fields None/NaN, not merely
    'excluded is truthy')."""
    T = 10
    yrs = list(range(2000, 2000 + T))
    models = ["A", "B"]
    La = np.column_stack([np.arange(T, dtype=float), np.arange(T, dtype=float) + 1.0])
    row = analyze_series("synthetic_short", La, models, yrs, "synthetic")
    assert row["excluded"] is True
    assert row["T"] == 10 and row["T1"] == 5 and row["T2"] == 5
    assert row["winner1"] is None and row["winner2"] is None and row["persists"] is None
    for k in ("k1", "k1_over_T1", "winner_stability1", "rank1_in_2", "rank_percentile", "loss_gap"):
        assert np.isnan(row[k]), f"{k} should be NaN for an excluded row, got {row[k]!r}"


def test_analyze_series_included_right_at_the_min_T_boundary():
    """T1=T2=MIN_T_PER_HALF (8) each is the smallest INCLUDED case -- confirm it is NOT excluded
    (off-by-one check on the >= vs > boundary the guard uses)."""
    T = 2 * MIN_T_PER_HALF
    yrs = list(range(2000, 2000 + T))
    models = ["A", "B"]
    # A strictly beats B every single period on both halves -> unambiguous, no ties anywhere.
    La = np.column_stack([np.zeros(T), np.ones(T)])
    row = analyze_series("synthetic_boundary", La, models, yrs, "synthetic")
    assert row["excluded"] is False
    assert row["T1"] == MIN_T_PER_HALF and row["T2"] == MIN_T_PER_HALF


# --- analyze_series(): hand-computed persistence arithmetic ----------------------------------

def test_analyze_series_dominant_winner_persists_with_full_kstar_and_zero_gap():
    """Model A has strictly lower loss than B in EVERY period of both halves. Hand-derived:
    winner1=A, winner2=A, persists=True. k1 = decision_breakdown's pairwise breakdown_number(A,B):
    every period is A-favoring, so that function's greedy loop (fragility.py:114-121) never reaches
    its `c[i] <= 0` break and removes all of them -- k1 must equal T1 exactly (not merely
    'large'). rank1_in_2=0 (A is
    still the best on the second half) so rank_percentile=1.0, and loss_gap=0.0 (no second-half
    winner other than A)."""
    T1, T2 = 8, 8
    T = T1 + T2
    yrs = list(range(2000, 2000 + T))
    models = ["A", "B"]
    lossA = np.concatenate([np.full(T1, 1.0), np.full(T2, 1.0)])
    lossB = np.concatenate([np.full(T1, 2.0), np.full(T2, 2.0)])
    La = np.column_stack([lossA, lossB])
    row = analyze_series("synthetic_dominant", La, models, yrs, "synthetic")
    assert row["excluded"] is False
    assert row["winner1"] == "A" and row["winner2"] == "A"
    assert row["persists"] is True
    assert row["k1"] == T1  # exact per breakdown_number's own documented "beats every period" case
    assert row["k1_over_T1"] == pytest.approx(1.0)
    assert row["rank1_in_2"] == 0
    assert row["rank_percentile"] == pytest.approx(1.0)
    assert row["loss_gap"] == pytest.approx(0.0)


def test_analyze_series_narrow_first_half_winner_flips_in_second_half():
    """Constructed so the first-half winner is decided by only 2 of 8 'anomalous' periods (the
    other 6 are an exact tie between A and B), and the second half reverses which model is better
    outright. Hand-derived expectations:
      First half: A=[1,1,1,1,1,1,0,0], B=[1,1,1,1,1,1,1,1] -> mean_A=0.75 < mean_B=1.0 -> winner1=A.
      k1 = breakdown_number(A,B): margin contributions c_t = w*(lb-la); the 6 tied periods
      contribute 0, the 2 anomalous periods contribute 1 each (M=2). Greedy removes the two c=1
      periods (both needed to drive the margin <=0): k1=2 exactly.
      Second half: A=2.0 flat, B=1.0 flat for all 8 periods -> winner2=B, so persists=False.
      order2 sorted ascending = [B, A]; winner1='A' is at index 1 of 2 models ->
      rank_percentile = 1 - 1/(2-1) = 0.0. loss_gap = means2[A] - means2[B] = 2.0 - 1.0 = 1.0.
    """
    T1, T2 = 8, 8
    T = T1 + T2
    yrs = list(range(2000, 2000 + T))
    models = ["A", "B"]
    lossA1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    lossB1 = np.full(T1, 1.0)
    lossA2 = np.full(T2, 2.0)
    lossB2 = np.full(T2, 1.0)
    La = np.column_stack([
        np.concatenate([lossA1, lossA2]),
        np.concatenate([lossB1, lossB2]),
    ])
    row = analyze_series("synthetic_flip", La, models, yrs, "synthetic")
    assert row["excluded"] is False
    assert row["winner1"] == "A"
    assert row["k1"] == 2
    assert row["k1_over_T1"] == pytest.approx(2 / 8)
    assert row["winner2"] == "B"
    assert row["persists"] is False
    assert row["rank1_in_2"] == 1
    assert row["rank_percentile"] == pytest.approx(0.0)
    assert row["loss_gap"] == pytest.approx(1.0)
    # winner_stability1 is a block-bootstrap statistic (not exactly hand-derivable here) --
    # sanity-bound it only, real coverage of its own arithmetic lives in fragility.py's test suite.
    assert 0.0 <= row["winner_stability1"] <= 1.0


def test_analyze_series_rank_percentile_with_three_models():
    """Same flip shape as above but with a third model C sandwiched between A and B on the second
    half, to confirm rank_percentile's (Kmods-1) denominator (not a hard-coded 2) is exercised
    correctly. Second half means: B=1.0, C=1.5, A=2.0 -> order2=[B,C,A], winner1='A' is at index 2
    of 3 models -> rank_percentile = 1 - 2/(3-1) = 0.0."""
    T1, T2 = 8, 8
    T = T1 + T2
    yrs = list(range(2000, 2000 + T))
    models = ["A", "B", "C"]
    lossA1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    lossB1 = np.full(T1, 1.0)
    lossC1 = np.full(T1, 1.0)
    lossA2 = np.full(T2, 2.0)
    lossB2 = np.full(T2, 1.0)
    lossC2 = np.full(T2, 1.5)
    La = np.column_stack([
        np.concatenate([lossA1, lossA2]),
        np.concatenate([lossB1, lossB2]),
        np.concatenate([lossC1, lossC2]),
    ])
    row = analyze_series("synthetic_three_model", La, models, yrs, "synthetic")
    assert row["winner1"] == "A"
    assert row["winner2"] == "B"
    assert row["persists"] is False
    assert row["rank1_in_2"] == 2
    assert row["rank_percentile"] == pytest.approx(0.0)
    assert row["loss_gap"] == pytest.approx(2.0 - 1.0)


# --- main()'s DataFrame-aggregation path: the object-dtype `~inc.persists` bug --------------
# As this file's own module docstring notes, main()'s DataFrame-aggregation path is otherwise
# exercised only by a full producer rerun -- which has never had an excluded row
# (n_excluded_min_T=0 on the real data), leaving this path uncovered. The test below reproduces
# main()'s own row-building + filtering sequence directly (using the real analyze_series(), not a
# reimplementation) with one excluded and one included series, matching the mixed-dtype scenario a
# future data refresh could produce.

def _short_series_row():
    """A series whose first half has fewer than MIN_T_PER_HALF years -- guaranteed excluded,
    forcing `persists=None` for this row (analyze_series()'s own early-return)."""
    T1, T2 = MIN_T_PER_HALF - 1, MIN_T_PER_HALF
    T = T1 + T2
    yrs = list(range(2000, 2000 + T))
    models = ["A", "B"]
    La = np.column_stack([np.arange(T, dtype=float), np.arange(T, dtype=float) + 1.0])
    return analyze_series("short_series", La, models, yrs, "synthetic")


def _normal_series_row():
    T1, T2 = MIN_T_PER_HALF, MIN_T_PER_HALF
    T = T1 + T2
    yrs = list(range(2000, 2000 + T))
    models = ["A", "B"]
    lossA1 = np.zeros(T1); lossB1 = np.ones(T1)          # A wins first half
    lossA2 = np.ones(T2); lossB2 = np.zeros(T2)           # B wins second half -> persists=False
    La = np.column_stack([
        np.concatenate([lossA1, lossA2]), np.concatenate([lossB1, lossB2]),
    ])
    return analyze_series("normal_series", La, models, yrs, "synthetic")


def test_short_series_is_genuinely_excluded_with_persists_none():
    """Sanity check on the fixture itself: confirms the short series really is excluded (so the
    resulting DataFrame really does mix None with real bools, not just look like it should)."""
    row = _short_series_row()
    assert row["excluded"] is True
    assert row["persists"] is None


def test_main_style_filtering_survives_a_mixed_none_bool_persists_column():
    """THE CORE REGRESSION: reproduces main()'s exact row-building + filtering sequence (build R
    from analyze_series() rows -> `inc = R[~R.excluded].copy()` -> an explicit `.astype(bool)` ->
    `nonpersist = inc[~inc.persists]`) with one excluded (persists=None) and one included
    (persists=a real bool) row. Before the fix, `inc.persists` stayed object-dtype even after
    filtering out the None row, and `~` on an object-dtype Series of real bools does BITWISE invert
    (`~True == -2`), not logical negation -- `inc[~inc.persists]` raised `KeyError: "None of
    [Index([-2, -1], ...)] are in the [columns]"`.

    The cast is asserted AGAINST PRODUCTION via inspect.getsource(main), not merely re-applied in
    this test body: re-applying it here would leave this test green with the production line
    deleted, which is precisely the regression it claims to guard."""
    import inspect
    import re

    R = pd.DataFrame([_short_series_row(), _normal_series_row()])
    assert R.persists.dtype == object, "sanity check: the fixture must genuinely produce a mixed-type column"

    inc = R[~R.excluded].copy()
    assert inc.persists.dtype == object, "sanity check: filtering alone must not fix the dtype (that's the bug)"

    # the failure mode itself, on main()'s own data shape: without the cast the filter does not
    # merely mis-select, it raises.
    with pytest.raises(KeyError):
        inc[~inc.persists]

    src = inspect.getsource(KP.main)
    assert re.search(r"""inc\[["']persists["']\]\s*=\s*inc\[["']persists["']\]\.astype\(bool\)""", src), (
        "kstar_persistence.main() must cast inc['persists'] to bool AFTER dropping the excluded "
        "(persists=None) rows and BEFORE `inc[~inc.persists]`. Without that cast the column stays "
        "object-dtype and `~` bitwise-inverts it, raising the KeyError reproduced just above."
    )
    # the cast main() performs must also actually resolve it, on these rows.
    inc["persists"] = inc["persists"].astype(bool)
    assert inc.persists.dtype == bool

    nonpersist = inc[~inc.persists]  # must not raise
    assert list(nonpersist.series) == ["normal_series"], (
        "the one non-persisting series must be correctly selected, not silently mis-selected"
    )
