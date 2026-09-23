"""social_choice.py's `_cell_stats()`/`permutation_null()` -- the Section 5 within-period
permutation null. This function
computes every canonical `election.permnull_*` value (15 keys) and calibrates Section 5's entire
headline (reversal rate, concentration>1, k*=1, no-Condorcet counts against an arbitrary-winner
null), but had no dedicated unit test of its own -- only indirect coverage via `canonical.py
--check` re-running the whole pipeline. These tests exercise the two functions directly, on small,
hand-constructed panels with known-in-advance answers, independent of the real 51-state/12-series
data.
"""
import numpy as np

import social_choice as SC
import fragility as F
# Shared with test_election_null_ar1.py -- one implementation, imported by both.
from _synthetic_panel import synthetic_panel as _panel


def test_cell_stats_matches_fragility_primitives_directly():
    """_cell_stats() must agree exactly with directly calling fragility()'s own primitives on the
    same panel -- it is a thin wrapper, not a re-derivation, and this locks in that the wrapping
    didn't introduce a discrepancy."""
    L, w = _panel(T=8, K=4, seed=1)
    reversal, conc_gt1, kstar_is_1, no_condorcet, is_cycle = SC._cell_stats(L, w)

    a = F.pooled_winner(L, w)
    pp = F.per_period_winner(L)
    k, b, _ = F.decision_breakdown(L, w, a)
    la = np.asarray(L[a], float); lb = np.asarray(L[b], float) if b is not None else la
    c = w * (lb - la)
    M = float(c.sum())
    conc = np.nan if F._margin_below_floor(la, lb, w, M) else float(np.max(c) / M)
    cstat = F.condorcet_status(L)

    assert reversal == (a != pp)
    assert conc_gt1 == bool(np.isfinite(conc) and conc > 1)
    assert kstar_is_1 == (k == 1)
    assert no_condorcet == (cstat != "winner")
    assert is_cycle == (cstat == "cycle")


def test_cell_stats_on_a_panel_engineered_for_a_known_reversal_and_concentration():
    """A hand-built panel where the answer is known by construction, not derived from the library
    under test: model 'a' wins 5 of 6 periods by a hair (plurality winner) but loses one period by
    a landslide, so its POOLED mean is worse than 'b' -- a textbook score-vs-plurality reversal with
    the single bad period supplying more than the entire net margin (concentration > 1)."""
    L = {
        "a": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 100.0]),
        "b": np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1]),
    }
    w = np.ones(6)
    reversal, conc_gt1, kstar_is_1, no_condorcet, is_cycle = SC._cell_stats(L, w)
    # 'a' wins periods 0-4 (1.0 < 1.1), loses period 5 (100.0 > 1.1) -> plurality winner 'a' (5-1).
    # pooled mean: a=17.5, b=1.1 -> score winner 'b'. Reversal (score != plurality).
    assert reversal is np.True_ or reversal is True
    assert kstar_is_1 is True or kstar_is_1 is np.True_   # deleting the one bad period flips the pooled winner
    assert conc_gt1 is True or conc_gt1 is np.True_       # that single period's margin exceeds the net margin
    assert no_condorcet is False                          # only 2 models -> no cycle/tie possible, 'b' wins head-to-head
    assert is_cycle is False


def test_cell_stats_on_a_robust_landslide_has_no_reversal_no_concentration():
    """Companion negative case: one model dominates every period by a similar amount -> no
    reversal, k* should be large (not 1), no concentration."""
    rng = np.random.default_rng(2)
    L = {"a": np.full(10, 1.0) + rng.normal(0, 0.01, 10),
         "b": np.full(10, 5.0) + rng.normal(0, 0.01, 10)}
    w = np.ones(10)
    reversal, conc_gt1, kstar_is_1, no_condorcet, is_cycle = SC._cell_stats(L, w)
    assert reversal is False or reversal is np.False_
    assert kstar_is_1 is False or kstar_is_1 is np.False_
    assert conc_gt1 is False or conc_gt1 is np.False_


def test_permutation_null_is_deterministic():
    """Same cells + same seed must reproduce byte-identical results -- this is what makes the
    published null table reproducible at all."""
    cells = [_panel(T=6, K=3, seed=i) for i in range(3)]
    obs1, null1 = SC.permutation_null(cells, n_perm=40, seed=777)
    obs2, null2 = SC.permutation_null(cells, n_perm=40, seed=777)
    assert np.array_equal(obs1, obs2)
    assert np.array_equal(null1, null2)


def test_permutation_null_observed_matches_direct_cell_stats_sum():
    """obs must equal independently summing _cell_stats() across the same cells directly -- the
    permutation loop must not silently mutate `cells` or double-count."""
    cells = [_panel(T=7, K=4, seed=i) for i in range(4)]
    obs, _ = SC.permutation_null(cells, n_perm=10, seed=1)
    direct = np.sum([SC._cell_stats(L, w) for L, w in cells], axis=0)
    assert np.array_equal(obs, direct.astype(int))


def test_permutation_null_shuffle_preserves_within_period_value_set(monkeypatch):
    """The null must permute MODEL LABELS within each period, never invent or drop a loss value:
    for every period, the multiset of loss values across models must be identical before and after
    shuffling. This is the exact property that makes it a label-permutation null and not some other
    randomization. Checked on the panels permutation_null() ACTUALLY builds, by spying on every
    panel it hands to _cell_stats() -- a null that shuffled down the columns (across periods)
    instead, or resampled with replacement, would break the per-period multisets below."""
    rng = np.random.default_rng(0)
    L = {m: rng.normal(1.0, 0.3, 5) for m in ("m0", "m1", "m2")}
    w = np.ones(5)
    original_per_period_sets = [sorted(row) for row in np.column_stack([L[m] for m in L])]

    seen = []
    real_cell_stats = SC._cell_stats

    def _spy(Lp, wp):
        seen.append(np.column_stack([Lp[m] for m in Lp]))
        return real_cell_stats(Lp, wp)

    monkeypatch.setattr(SC, "_cell_stats", _spy)
    n_perm = 8
    SC.permutation_null([(L, w)], n_perm=n_perm, seed=0)

    assert len(seen) == n_perm + 1, "expected one observed panel plus one panel per permutation"
    for A in seen:
        assert [sorted(row) for row in A] == original_per_period_sets
    # and the permuted panels must not all be the untouched original -- otherwise the "null" is a
    # no-op and the multiset property above would hold trivially.
    assert any(not np.array_equal(A, seen[0]) for A in seen[1:])


def test_permutation_null_shape_and_valid_range():
    """null array must be (n_perm, 5) with every entry a valid count in [0, n_cells] (each cell
    contributes at most one Boolean 1 per statistic)."""
    n_cells = 5
    cells = [_panel(T=6, K=3, seed=i) for i in range(n_cells)]
    obs, null = SC.permutation_null(cells, n_perm=25, seed=3)
    assert obs.shape == (5,)
    assert null.shape == (25, 5)
    assert (null >= 0).all() and (null <= n_cells).all()
    assert (obs >= 0).all() and (obs <= n_cells).all()


def test_permutation_null_two_model_cell_never_has_intransitive_cycle():
    """A 2-model panel can never produce a Condorcet cycle (cycles require >=3 alternatives) --
    is_cycle (the 5th _cell_stats element) must always be False for every draw, including every
    permuted draw, on a 2-model cell."""
    L, w = _panel(T=6, K=2, seed=5)
    obs, null = SC.permutation_null([(L, w)], n_perm=30, seed=8)
    assert obs[4] == 0
    assert (null[:, 4] == 0).all()
