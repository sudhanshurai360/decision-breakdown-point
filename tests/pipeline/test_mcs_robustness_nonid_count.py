"""mcs_robustness.py's own logic (nonid_count(), the tag/idx selection, counting non-identified
series across a `losses` dict). The rest of the suite covers this statistical family only one layer
below, at mcs_test.py's mcs() itself (test_mcs_test_degenerate_se.py). This pins nonid_count()'s own dispatch and counting logic with hand-constructed,
fully-deterministic panels (no randomness, no seed dependency) rather than the real curated data,
so the test does not depend on breadth_distinct_predictions.csv or drift with it.

Scope note: this covers mcs_robustness.py's nonid_count() only, the function that
mcs.curated_grid_min/_max directly relies on. mcs_power.py's sweep() and
mcs_circular_check.py's three_series_cross_check() are covered separately, in
test_mcs_power.py and test_mcs_circular_check.py respectively.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import mcs_robustness as MR  # noqa: E402


def _panel(abs_cols, sq_cols):
    """Build one series' (La, Ls) pair the way per_series_losses() shapes it: T x K arrays."""
    return np.column_stack(abs_cols), np.column_stack(sq_cols)


def test_genuine_dominance_is_identified_not_counted():
    """One series, model 0 strictly and enormously better every period (real deterministic
    dominance, no noise) -- MCS should collapse to {0}, so this series does NOT count as
    non-identified. Mirrors mcs_test.py's own dominance positive-control, but through
    nonid_count()'s dispatch rather than calling MT.mcs() directly."""
    T = 20
    La, Ls = _panel([np.full(T, 1.0), np.full(T, 5.0)], [np.full(T, 1.0), np.full(T, 25.0)])
    losses = {"DOMINATED": (La, Ls)}
    count = MR.nonid_count(losses, "abs", alpha=0.10, B=200, block=1, seed=0)
    assert count == 0, f"expected 0 non-identified (clear dominance), got {count}"


def test_exact_tie_is_non_identified():
    """One series, two models with EXACTLY identical losses every period (no noise at all) -- the
    bootstrap differential is a constant zero, se=0 by construction, and mcs_test.py's degenerate-
    se guard (test_mcs_test_degenerate_se.py) keeps both models rather than certainly-rejecting on
    zero real evidence. Both models survive -> this series counts as non-identified."""
    T = 20
    La, Ls = _panel([np.full(T, 2.0), np.full(T, 2.0)], [np.full(T, 4.0), np.full(T, 4.0)])
    losses = {"TIED": (La, Ls)}
    count = MR.nonid_count(losses, "abs", alpha=0.10, B=200, block=1, seed=0)
    assert count == 1, f"expected 1 non-identified (exact tie, both survive), got {count}"


def test_counts_sum_across_multiple_series_independently():
    """nonid_count() sums the non-identified indicator ACROSS every series in the dict -- pins
    that dispatch/aggregation logic directly (one dominated + one tied -> exactly 1, not 0 or 2),
    independent of mcs_test.mcs()'s own per-panel correctness (already covered elsewhere)."""
    T = 20
    dom_La, dom_Ls = _panel([np.full(T, 1.0), np.full(T, 5.0)], [np.full(T, 1.0), np.full(T, 25.0)])
    tie_La, tie_Ls = _panel([np.full(T, 2.0), np.full(T, 2.0)], [np.full(T, 4.0), np.full(T, 4.0)])
    losses = {"DOMINATED": (dom_La, dom_Ls), "TIED": (tie_La, tie_Ls)}
    count = MR.nonid_count(losses, "abs", alpha=0.10, B=200, block=1, seed=0)
    assert count == 1, f"expected exactly 1 of 2 series non-identified, got {count}"


def test_tag_selects_abs_vs_sq_independently():
    """`tag` must select La (idx 0) vs Ls (idx 1) independently -- a series identified on abs loss
    but tied on sq loss (or vice versa) must report differently under each tag. Constructs a
    series where the abs-loss arm is a clear dominance (identified) but the sq-loss arm is an
    exact tie (non-identified) to make the two tags disagree on the same series."""
    T = 20
    La = np.column_stack([np.full(T, 1.0), np.full(T, 5.0)])       # abs: dominance -> identified
    Ls = np.column_stack([np.full(T, 3.0), np.full(T, 3.0)])       # sq: exact tie -> non-identified
    losses = {"MIXED": (La, Ls)}
    assert MR.nonid_count(losses, "abs", alpha=0.10, B=200, block=1, seed=0) == 0
    assert MR.nonid_count(losses, "sq", alpha=0.10, B=200, block=1, seed=0) == 1
