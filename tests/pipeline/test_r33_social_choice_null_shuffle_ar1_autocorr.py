"""Mutation-killing tests for social_choice.permutation_null / ar1_null and
election_null_ar1._lag1_autocorr.

A blind scan found four number-changing mutations the rest of the suite does not catch, named M1,
M4, M5 and M8 in this file only: (M1) shuffling ACROSS periods within each model instead of model
labels WITHIN each period; (M4) shuffling whole periods with labels intact (a no-op null); (M5)
ar1_null() ignoring ar_hat; (M8) _lag1_autocorr() dividing by sum(x**2) instead of sum(x[:-1]**2).
Nothing elsewhere asserts that ar1_null()'s output depends on ar_hat, and the autocorr recovery
tests run at T=5000, where the (T-1)/T factor is invisible. Each test below names the mutation it
kills ("kills: ...") and was verified to FAIL on a copy carrying that mutation and PASS on the real
module.
"""
import numpy as np
import pytest

import election_null_ar1 as EAR1
import social_choice as SC


# ============================================================================================
# permutation_null(): the shuffle must permute model labels WITHIN each period
# ============================================================================================

def _distinct_panel():
    """T=4 periods x K=3 models with every one of the 12 values distinct, so 'which values sit in
    period t' identifies the period and 'which values sit in model m' identifies the model."""
    A = np.array([[1.0, 2.0, 3.0],
                  [10.0, 20.0, 30.0],
                  [100.0, 200.0, 300.0],
                  [1000.0, 2000.0, 3000.0]])
    L = {"m0": A[:, 0].copy(), "m1": A[:, 1].copy(), "m2": A[:, 2].copy()}
    return A, L, np.ones(4)


def test_permutation_null_shuffles_labels_within_each_period_through_the_real_function(monkeypatch):
    """kills: M1 (`for j: rng.shuffle(A[:, j])`, shuffle across periods within model), M4
    (`rng.shuffle(A)`, shuffle whole periods, a no-op null).

    Calls SC.permutation_null() itself on the distinct-valued panel with _cell_stats monkeypatched
    to record every panel it is handed. For every permuted draw and every period t, the multiset
    of the K values in period t must equal the original period's multiset (a label permutation
    never moves a value between periods -- M1 violates this in essentially every draw; M4 moves
    whole rows, so period t receives another period's values). And across the draws at least one
    model column must differ from the original (the labels really are shuffled -- a no-op would
    reproduce the observed panel in every draw)."""
    A, L, w = _distinct_panel()
    seen = []

    def recording_cell_stats(Lx, wx):
        seen.append(np.column_stack([Lx[m] for m in ("m0", "m1", "m2")]))
        return (False, False, False, False, False)

    monkeypatch.setattr(SC, "_cell_stats", recording_cell_stats)
    n_perm = 25
    SC.permutation_null([(L, w)], n_perm=n_perm, seed=11)   # the panels it builds are what matter
    draws = seen[1:]                       # seen[0] is the observed-statistic call
    assert len(draws) == n_perm
    for D in draws:
        assert D.shape == A.shape
        for t in range(A.shape[0]):
            assert sorted(D[t]) == sorted(A[t]), (
                f"period {t} received values {D[t]} not from its own original row {A[t]}")
    assert any(not np.array_equal(D, A) for D in draws), "labels were never shuffled (no-op null)"
    assert any(not np.array_equal(D[:, 0], A[:, 0]) for D in draws)


def test_permutation_null_does_not_mutate_the_caller_panel():
    """The shuffle must work on a copy (np.column_stack) -- the observed statistics computed first
    and the caller's arrays must be untouched after 2,000-style repeated draws."""
    A, L, w = _distinct_panel()
    before = {m: v.copy() for m, v in L.items()}
    SC.permutation_null([(L, w)], n_perm=10, seed=3)
    for m in L:
        assert np.array_equal(L[m], before[m])


# ============================================================================================
# ar1_null(): the synthetic panels must carry ar_hat
# ============================================================================================

def _cells(n=3, T=12, K=3):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        L = {f"m{j}": 1.0 + 0.1 * rng.normal(size=T) for j in range(K)}
        out.append((L, np.ones(T)))
    return out


def test_ar1_null_forwards_ar_hat_and_each_cells_own_K_T_to_panel_ar1(monkeypatch):
    """kills: M5 (`panel_ar1(rng, K, T, 0.0)` -- ar_hat ignored).

    Spies on election_null_ar1.panel_ar1 (ar1_null imports it at call time, so the module
    attribute is the one it sees) and asserts every synthetic panel is requested at the calling
    cell's own (K, T) and at exactly ar_hat."""
    calls = []
    real = EAR1.panel_ar1

    def spy(g, K, T, ar):
        calls.append((K, T, ar))
        return real(g, K, T, ar)

    monkeypatch.setattr(EAR1, "panel_ar1", spy)
    cells = [_cells(1, T=12, K=3)[0], _cells(1, T=9, K=4)[0]]
    SC.ar1_null(cells, ar_hat=0.37, n_perm=5, seed=1)
    assert len(calls) == 10
    assert calls[0::2] == [(3, 12, 0.37)] * 5
    assert calls[1::2] == [(4, 9, 0.37)] * 5


def test_ar1_null_distribution_depends_on_ar_hat():
    """kills: M5. Same cells, same seed, same n_perm: the null draws at ar_hat=0.0 and ar_hat=0.9
    must differ (the AR(1) recursion changes every synthetic value after the first period). The
    observed statistics, which do not involve the null, must be identical."""
    cells = _cells()
    obs0, null0 = SC.ar1_null(cells, ar_hat=0.0, n_perm=30, seed=5)
    obs9, null9 = SC.ar1_null(cells, ar_hat=0.9, n_perm=30, seed=5)
    assert np.array_equal(obs0, obs9)
    assert null0.shape == null9.shape == (30, 5)
    assert not np.array_equal(null0, null9)


# ============================================================================================
# _lag1_autocorr(): OLS denominator sum(x[:-1]**2), hand-computed
# ============================================================================================

def test_lag1_autocorr_hand_computed_on_a_five_point_spike():
    """kills: M8 (`den = sum(x**2)`).

    x = [0, 0, 0, 0, 5]: mean 1, demeaned [-1, -1, -1, -1, 4].
      numerator   sum(x[1:] * x[:-1]) = (-1)(-1) + (-1)(-1) + (-1)(-1) + (4)(-1) = -1
      denominator sum(x[:-1] ** 2)    = 1 + 1 + 1 + 1                          =  4
      -> -0.25.  With sum(x**2) = 20 the mutant returns -0.05."""
    assert EAR1._lag1_autocorr([0, 0, 0, 0, 5]) == pytest.approx(-0.25)


def test_lag1_autocorr_hand_computed_on_a_linear_ramp():
    """kills: M8. x = 1..6, demeaned [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
      numerator   3.75 + 0.75 - 0.25 + 0.75 + 3.75 = 8.75
      denominator 6.25 + 2.25 + 0.25 + 0.25 + 2.25 = 11.25   (sum(x**2) would be 17.5)
      -> 8.75 / 11.25 = 0.7777...   (the mutant gives exactly 0.5)."""
    assert EAR1._lag1_autocorr([1, 2, 3, 4, 5, 6]) == pytest.approx(8.75 / 11.25)


def test_lag1_autocorr_guards_short_and_constant_input():
    """The documented None branches: fewer than 5 points, or a constant series."""
    assert EAR1._lag1_autocorr([1, 2, 3, 4]) is None
    assert EAR1._lag1_autocorr([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) is None
