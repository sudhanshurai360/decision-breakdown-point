"""Pytest port of code/instrument/test_fragility.py, preserving every original check
exactly -- same RNG seeds, same trial counts, same thresholds -- just as real pytest tests instead of
a standalone script with a hand-rolled PASS/FAIL list, so failures are individually addressable and
this becomes part of the same growing regression suite as every fix from here forward.

The original script (code/instrument/test_fragility.py) is left in place and still runnable directly
-- this port does not replace it, it gives the same checks pytest's real machinery (individual
failures, -k selection, CI integration) on top of the exact same assertions.

NOTE: "preserving every original check exactly" describes the port itself, not the file as it stands
now. The file has grown well past the original script since -- a substantial share of the tests below
are later additions covering behaviour the original never checked. Those additions are held to the
same standard, but they are not part of the ported set and carry no claim of fidelity to it.
"""
import numpy as np
import pytest

import fragility as F


# ---------------------------------------------------------------------------
# 1) greedy k* == brute force, over many random small cases (the exactness lemma)
# ---------------------------------------------------------------------------
def test_breakdown_number_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        T = int(rng.integers(3, 9))
        la, lb, w = rng.random(T), rng.random(T), rng.random(T) + 0.5
        assert F.breakdown_number(la, lb, w)[0] == F._brute_force_k(la, lb, w)


# ---------------------------------------------------------------------------
# 2) ARTIFACT case: deep wins all 13 calm years; AR1 is least-bad in the 2 COVID
#    years -> pooled flips to AR1. Mirrors a real AR1-vs-deep-learning shock artifact.
# ---------------------------------------------------------------------------
@pytest.fixture
def artifact_result():
    T = 15
    w = np.full(T, 52.0)
    covid = [9, 10]
    years = list(range(2011, 2026))
    deep = np.full(T, 0.35)
    ar1 = np.full(T, 0.58)
    for i in covid:
        deep[i], ar1[i] = 8.5, 5.9
    return F.fragility(
        {"ar1": ar1, "deep": deep}, w,
        shock_periods=[years[i] for i in covid], labels=years, n_boot=500,
    )


def test_artifact_pooled_winner_is_ar1(artifact_result):
    assert artifact_result["pooled_winner"] == "ar1"


def test_artifact_per_period_winner_is_deep(artifact_result):
    assert artifact_result["per_period_winner"] == "deep"


def test_artifact_reversal_flagged(artifact_result):
    assert artifact_result["reversal"] is True


def test_artifact_is_fragile_small_kstar(artifact_result):
    assert artifact_result["k_star"] <= 2
    assert artifact_result["k_over_T"] < 0.2


def test_artifact_responsible_periods_subset_of_covid_years(artifact_result):
    assert set(artifact_result["responsible"]) <= {2020, 2021}


def test_artifact_shock_coincidence_true(artifact_result):
    assert artifact_result["shock_coincidence"] is True


def test_artifact_flagged_fragile(artifact_result):
    assert artifact_result["fragile"] is True


def test_artifact_winner_unstable_to_resampling(artifact_result):
    assert artifact_result["winner_stability"] < 0.95


# ---------------------------------------------------------------------------
# 3) TRUE-NEGATIVE: deep beats AR1 in EVERY period (incl COVID) -> robust ranking.
#    The diagnostic must stay silent on a genuinely robust ranking, or it's unfalsifiable.
# ---------------------------------------------------------------------------
@pytest.fixture
def true_negative_result():
    T = 15
    w = np.full(T, 52.0)
    covid = [9, 10]
    deep2 = np.full(T, 0.35)
    ar1b = np.full(T, 0.58)
    for i in covid:
        deep2[i], ar1b[i] = 6.0, 8.0
    return F.fragility({"ar1": ar1b, "deep": deep2}, w, n_boot=500)


def test_true_negative_pooled_winner_is_deep(true_negative_result):
    assert true_negative_result["pooled_winner"] == "deep"


def test_true_negative_no_reversal(true_negative_result):
    assert true_negative_result["reversal"] is False


def test_true_negative_is_robust_high_kstar_over_t(true_negative_result):
    assert true_negative_result["k_over_T"] >= 0.9


def test_true_negative_not_fragile(true_negative_result):
    assert true_negative_result["fragile"] is False


def test_true_negative_winner_stable_to_resampling(true_negative_result):
    assert true_negative_result["winner_stability"] > 0.95


# ---------------------------------------------------------------------------
# 4) SURPRISE-CONCENTRATION: a CLIFF (one spike) -> sci~1, eff_n~1;
#    a GRADUAL ramp -> sci low, eff_n high.
# ---------------------------------------------------------------------------
def test_surprise_concentration_cliff_sci_near_one():
    spike = np.full(15, 0.3)
    spike[9] = 8.0
    sc_cliff = F.surprise_concentration({"a": spike, "b": np.full(15, 0.6)}, labels=list(range(15)))
    assert sc_cliff["sci"] > 0.95


def test_surprise_concentration_cliff_eff_periods_near_one():
    spike = np.full(15, 0.3)
    spike[9] = 8.0
    sc_cliff = F.surprise_concentration({"a": spike, "b": np.full(15, 0.6)}, labels=list(range(15)))
    assert sc_cliff["eff_surprising_periods"] < 1.5


def test_surprise_concentration_cliff_hardest_period_identified():
    spike = np.full(15, 0.3)
    spike[9] = 8.0
    sc_cliff = F.surprise_concentration({"a": spike, "b": np.full(15, 0.6)}, labels=list(range(15)))
    assert sc_cliff["hardest"] == 9


def test_surprise_concentration_gradual_sci_low():
    grad = np.array([.3, .3, .3, .3, .3, .5, .7, .9, 1.1, 1.3, 1.5, .4, .4, .4, .4])
    sc_grad = F.surprise_concentration({"a": grad, "b": grad + 0.2})
    assert sc_grad["sci"] < 0.5


def test_surprise_concentration_gradual_eff_periods_greater_than_cliff():
    spike = np.full(15, 0.3)
    spike[9] = 8.0
    sc_cliff = F.surprise_concentration({"a": spike, "b": np.full(15, 0.6)}, labels=list(range(15)))
    grad = np.array([.3, .3, .3, .3, .3, .5, .7, .9, 1.1, 1.3, 1.5, .4, .4, .4, .4])
    sc_grad = F.surprise_concentration({"a": grad, "b": grad + 0.2})
    assert sc_grad["eff_surprising_periods"] > 3
    assert sc_grad["eff_surprising_periods"] > sc_cliff["eff_surprising_periods"]


# ---------------------------------------------------------------------------
# surprise_concentration's
# `d_max_over_median` has two edge branches (b == 0, and (b < 0) != (d_max < 0)) that neither this
# file's original standalone-script suite nor this pytest port previously exercised at all -- only
# the ordinary positive-baseline path was tested. Pins the current, now-sign-explicit behavior of
# both branches (the zero-baseline case is defined as the ceiling value 1.0, not division by zero).
# ---------------------------------------------------------------------------
def test_d_max_over_median_zero_baseline_zero_max_is_one():
    """b == 0 and d_max == 0 (every model achieves exactly zero loss every period) -> the degenerate
    zero-over-zero case is defined as 1.0 (no surprise above a baseline that IS zero)."""
    L = {"a": np.array([0.0, 0.0, 0.0]), "b": np.array([0.0, 0.0, 0.0])}
    sc = F.surprise_concentration(L)
    assert sc["d_max_over_median"] == 1.0


def test_d_max_over_median_zero_baseline_positive_max_is_positive_inf():
    """b == 0 (median difficulty is exactly zero) but d_max > 0 (the hardest period is not) ->
    d_max/b is +inf, not NaN or a crash. Per-period difficulty d = min-across-models loss = [0, 0, 5]
    (b = median = 0, d_max = max = 5)."""
    L = {"a": np.array([0.0, 0.0, 5.0]), "b": np.array([1.0, 1.0, 6.0])}
    sc = F.surprise_concentration(L)
    assert sc["d_max_over_median"] == float("inf")


def test_d_max_over_median_sign_mismatch_is_nan():
    """(b < 0) != (d_max < 0): baseline and hardest-period difficulty have opposite signs, so the
    ratio is undefined -> NaN, not a signed infinity or a silently wrong finite ratio. Per-period
    difficulty d = [-5, -3, 2] (b = median = -3 < 0, d_max = max = 2 >= 0)."""
    L = {"a": np.array([-5.0, -3.0, 2.0]), "b": np.array([-4.0, -2.0, 3.0])}
    sc = F.surprise_concentration(L)
    assert np.isnan(sc["d_max_over_median"])


# ---------------------------------------------------------------------------
# 5) MULTI-OPPONENT decision-level k* (min over ALL opponents) == brute force,
#    and it's strictly TIGHTER than runner-up-only over enough trials.
# ---------------------------------------------------------------------------
def test_decision_breakdown_matches_brute_force_multi_opponent():
    rng = np.random.default_rng(0)
    for _ in range(400):
        K = int(rng.integers(3, 6))
        T = int(rng.integers(3, 9))
        L = {f"m{j}": rng.random(T) for j in range(K)}
        w = rng.random(T) + 0.5
        kg, _opp, _ = F.decision_breakdown(L, w)
        kb = F._brute_force_k_multi(L, w)
        assert kg == kb


def test_decision_breakdown_multi_opponent_can_be_tighter_than_runner_up():
    """Over enough trials, at least one case has the binding opponent NOT be the pooled runner-up --
    i.e. multi-opponent k* is a strictly tighter bound than the old runner-up-only approximation."""
    rng = np.random.default_rng(0)
    third_binds = False
    for _ in range(400):
        K = int(rng.integers(3, 6))
        T = int(rng.integers(3, 9))
        L = {f"m{j}": rng.random(T) for j in range(K)}
        w = rng.random(T) + 0.5
        _kg, opp, _ = F.decision_breakdown(L, w)
        a = F.pooled_winner(L, w)
        ru = min((m for m in L if m != a), key=lambda m: float(np.average(L[m], weights=w)))
        if opp is not None and opp != ru:
            third_binds = True
            break
    assert third_binds


# ---------------------------------------------------------------------------
# concentration must be NaN-guarded on a
# degenerate margin, the same way winner_stability already is. Ported alongside the identical
# fix and tightened test in the released selection-fragility package (this file's own twin).
# ---------------------------------------------------------------------------
def test_concentration_nan_guarded_on_degenerate_margin():
    """CONFIRMED REGRESSION: concentration used a bare `M > 0` guard, missing the same
    degenerate-margin floor breakdown_number() already applies -- a tiny POSITIVE margin (at or
    below floating-point noise) still divided through to a huge, meaningless ratio. Fixed via
    _margin_below_floor(), shared with breakdown_number's own floor. Same near-tie construction as
    the selection-fragility package's near_tie_float_panel fixture (T=20, margin ~1e-13 relative)."""
    T = 20
    a = np.full(T, 1.0)
    b = a.copy()
    b[0] += 2e-13 * T
    w = np.ones(T)
    r = F.fragility({"a": a, "b": b}, w)
    assert r["k_star"] == 0, "fixture is expected to be a degenerate (near-zero-margin) tie"
    assert np.isnan(r["concentration"]), (
        f"degenerate margin must produce exactly NaN concentration, got {r['concentration']!r}"
    )


def test_concentration_still_finite_on_a_genuine_margin():
    """Negative control for the fix above: a REAL, decisive margin must still produce a normal,
    finite concentration -- the floor must not swallow genuine cases."""
    T = 15
    a = np.full(T, 1.0)
    b = np.full(T, 1.6)
    w = np.ones(T)
    r = F.fragility({"a": a, "b": b}, w)
    assert np.isfinite(r["concentration"])


# ---------------------------------------------------------------------------
# per_period_winner's within-period tie used to be
# resolved by `M.argmin(axis=0)`, which on an exact tie silently picks the lowest dict-insertion-
# order index -- fabricating a "win" out of pure insertion order, falsifying election.reversals_lo/
# hi's own spec claim of bounding the reversal count under EVERY possible plurality tie-break. Fixed
# by porting the tool package's already-reviewed STRICT WINS ONLY convention (a tied period counts
# toward nobody). Confirmed genuinely latent against the real curated panel (0 of its 776
# series-year periods -- 388 series-years x 2 metrics -- have a within-period exact tie today).
# ---------------------------------------------------------------------------
def test_within_period_tie_counts_toward_nobody():
    """CONFIRMED REGRESSION: two models tied at the per-period minimum every single period; a third
    model strictly worse every period. Under the old argmin-based counting, whichever of the tied
    pair sorted first in dict insertion order would be silently awarded the win every period and
    reported as the plurality winner. Under STRICT WINS ONLY, neither tied model wins any period
    (0 strict wins each), so BOTH end up tied at the top win count (0, since the worse model also
    has 0) -- an honest reflection that no model has genuine plurality evidence, not a fabricated
    winner from insertion order."""
    T = 10
    tied_val = np.full(T, 1.0)
    worse = np.full(T, 2.0)
    # insertion order deliberately puts the "loser" name first alphabetically-later model first,
    # so an old insertion-order-driven bug and a name-sort-driven bug would disagree on the winner
    # -- isolates the fix from either accidental convention producing the "right" answer by luck.
    L = {"zzz_tied_a": tied_val, "aaa_tied_b": tied_val.copy(), "mmm_worse": worse}
    winner, tied = F.per_period_winner(L, return_ties=True)
    assert set(tied) == {"zzz_tied_a", "aaa_tied_b", "mmm_worse"}, (
        f"expected all three models tied at 0 strict wins (the within-period tie counts toward "
        f"nobody every period), got tied={tied!r}"
    )
    assert winner == "aaa_tied_b", "outer tie among all-0-win models must still resolve by sorted name"


def test_unique_win_still_counted_normally():
    """Positive control / negative control for the fix above: a model that GENUINELY wins a period
    outright (no tie) must still be credited that win exactly as before -- the fix must not zero out
    legitimate, unambiguous wins."""
    T = 10
    a = np.full(T, 1.0)         # strictly best every period, no ties anywhere
    b = np.full(T, 2.0)
    c = np.full(T, 3.0)
    winner, tied = F.per_period_winner({"a": a, "b": b, "c": c}, return_ties=True)
    assert winner == "a"
    assert tied == ["a"]


def test_mixed_ties_and_unique_wins():
    """A model that wins some periods outright and ties others: only the outright wins count.
    Model 'a' uniquely wins periods 0-4 (5 strict wins); 'a' and 'b' tie at periods 5-9 (0 strict
    wins each from those periods). 'a' must have exactly 5 strict wins, 'b' exactly 0."""
    a = np.array([1.0] * 5 + [1.0] * 5)
    b = np.array([2.0] * 5 + [1.0] * 5)   # loses periods 0-4, ties periods 5-9
    c = np.array([3.0] * 10)              # never wins or ties
    winner, tied = F.per_period_winner({"a": a, "b": b, "c": c}, return_ties=True)
    assert winner == "a"
    assert tied == ["a"], "'a' has 5 strict wins, strictly more than b's 0 and c's 0 -- must win outright"


# ---------------------------------------------------------------------------
# breakdown_number's `order = np.argsort(c)[::-1]`
# is not guaranteed stable for equal values, and the trailing [::-1] reverses the whole result -- so
# two periods with the EXACT same margin contribution could be removed/named in an order that is an
# accidental byproduct of the sort algorithm. k* (the removal count) is unaffected, but WHICH periods
# land in `removed` when the greedy loop stops PARTWAY through a tied group is not. Fixed via
# `np.argsort(-c, kind="stable")`: identical whenever no two contributions tie, and for a genuine
# tie, deterministically prefers the earlier period. Present, unfixed, in BOTH copies of this file
# until now.
# ---------------------------------------------------------------------------
def test_breakdown_number_tied_margin_prefers_earlier_period():
    """CONFIRMED REGRESSION: 5 periods with contributions [20, 5, 5, 5, -7] (period 0 removed first
    unambiguously; periods 1-3 tied at 5; period 4 is negative/never removed). The greedy loop stops
    strictly INSIDE the tied group (after removing 2 of the 3 tied periods -- k*=3 either way, only
    the SET differs). Under the old np.argsort(c)[::-1], the tied group was processed as [3, 2, 1]
    (reversed original order), removing {0, 3, 2} and leaving period 1 un-removed. The fix must
    instead remove {0, 1, 2}, leaving period 3 (the LATER tied period) un-removed."""
    la = np.zeros(5)
    lb = np.array([20.0, 5.0, 5.0, 5.0, -7.0])
    w = np.ones(5)
    k, removed = F.breakdown_number(la, lb, w)
    assert k == 3, f"k* must be 3 regardless of tie order (removal COUNT is tie-order-invariant), got {k}"
    assert removed == [0, 1, 2], (
        f"expected the EARLIER two tied periods (1, 2) removed and the LATER (3) left, got {removed!r}"
    )


def test_breakdown_number_unaffected_when_no_ties():
    """Positive control: with no exact ties in the margin contributions, the fix must produce the
    exact same removed set as the original descending-sort convention."""
    rng = np.random.default_rng(2)
    la = rng.normal(1.0, 0.05, 15)
    lb = la + rng.normal(0.3, 0.05, 15)
    w = np.ones(15)
    k, removed = F.breakdown_number(la, lb, w)
    c = w * (lb - la)
    expected_order = np.argsort(c)[::-1]
    s, expected_removed = float(c.sum()), []
    for i in expected_order:
        if c[i] <= 0:
            break
        s -= float(c[i])
        expected_removed.append(int(i))
        if s <= 0:
            break
    assert removed == expected_removed, "no ties present -- fix must not change the removed set"


# ---------------------------------------------------------------------------
# breakdown_number() had no `w >= 0` guard.
# Verified directly (not merely asserted): a negative weight on a period UNRELATED to the margin
# (la==lb there, contributing exactly 0 to c regardless of weight sign) can still defeat the
# degenerate-margin floor's protection, because the floor is scaled by wscale=mean(w) -- a single
# large-magnitude negative weight can drag wscale negative, flipping the floor's sign so a genuine
# floating-point-noise tie is no longer recognized as degenerate. Fixed by raising ValueError on
# any negative weight.
# ---------------------------------------------------------------------------
def test_breakdown_number_rejects_negative_weight():
    """CONFIRMED REGRESSION construction: T=8, la constant, lb differs from la by ~2e-13*T at one
    period (the same formula pattern test_concentration_nan_guarded_on_degenerate_margin uses at
    T=20 to construct a degenerate tie, at a smaller T here) -- a genuine floating-point-noise tie,
    k*=0 with all-positive weights. Injecting a
    large negative weight on an UNRELATED period (index 1, where la==lb, contributing 0 to c
    regardless of that period's weight) flipped this to a reported k*=1 before this fix. Now must
    raise instead."""
    T = 8
    la = np.full(T, 1.0)
    lb = la.copy()
    lb[0] += 2e-13 * T
    w = np.ones(T)
    w[1] = -50.0
    with pytest.raises(ValueError, match="negative"):
        F.breakdown_number(la, lb, w)


def test_breakdown_number_all_positive_weights_still_works():
    """Negative control / positive control: the same degenerate-tie construction with all-positive
    weights must still work exactly as before (k*=0, no exception)."""
    T = 8
    la = np.full(T, 1.0)
    lb = la.copy()
    lb[0] += 2e-13 * T
    w = np.ones(T)
    k, removed = F.breakdown_number(la, lb, w)
    assert k == 0
    assert removed == []


def test_pooled_winner_rejects_negative_weight():
    """CONFIRMED REGRESSION (found alongside breakdown_number's own negative-weight
    guard): np.average(x, weights=w) is directly sign-sensitive to sum(w) -- with a
    negative weight large enough to push sum(w) toward zero, this silently computes a wildly wrong
    pooled mean instead of raising. pooled_winner is also the FIRST function decision_breakdown/
    fragility() call, so breakdown_number's own guard alone does not protect this entry point.

    Model 'a' has an OUTLIER, and the outlier is load-bearing: a CONSTANT array would not
    demonstrate the corruption at all, because a constant array's weighted mean is sign-invariant
    whenever sum(w)!=0. With this w, np.average([1,1,1,10], weights=[1,1,1,-2.9]) independently
    verified to compute -260 (vs. an honest weighted mean of ~3.25) -- confirming this isn't a
    hypothetical risk for this specific construction, not merely a generic sign-sensitivity claim."""
    L = {"a": np.array([1.0, 1.0, 1.0, 10.0]), "b": np.array([5.0, 5.0, 5.0, 5.0])}
    w = np.array([1.0, 1.0, 1.0, -2.9])   # sum(w) = 0.1, near-zero denominator
    with pytest.raises(ValueError, match="negative"):
        F.pooled_winner(L, w)


def test_pooled_winner_all_positive_weights_still_works():
    """Negative control / positive control: an ordinary all-positive-weight case must still work
    exactly as before."""
    L = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0, 6.0])}
    w = np.ones(3)
    assert F.pooled_winner(L, w) == "a"


# ---------------------------------------------------------------------------
# the `w < 0` guard
# above (and its regression test) rejects negative weights, but an ALL-ZERO (still non-negative)
# weight vector slipped past it and reached `np.average`'s internal `sum(w)` division unguarded --
# reproduced directly: F.pooled_winner({'a':[1,2,3],'b':[2,1,3]}, np.array([0.,0.,0.])) raised a bare
# ZeroDivisionError from inside numpy, not this function's own named ValueError. Fixed by extending
# the guard to also reject an all-zero weight vector explicitly.
# ---------------------------------------------------------------------------
def test_pooled_winner_rejects_all_zero_weights():
    """CONFIRMED REGRESSION: an all-zero (non-negative, so it passes the `w < 0` check) weight
    vector must raise this function's own clear ValueError, not an opaque ZeroDivisionError from
    inside np.average."""
    L = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([2.0, 1.0, 3.0])}
    w = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="strictly positive"):
        F.pooled_winner(L, w)


def test_pooled_winner_uniform_nonzero_weights_still_works():
    """Positive control / negative control for the fix above: every real caller in this codebase
    passes uniform w=ones(T) -- confirm the new all-zero guard does not disturb that ordinary case."""
    L = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0, 6.0])}
    w = np.ones(3)
    assert F.pooled_winner(L, w) == "a"
    # a non-uniform but still all-positive weight vector must also be unaffected
    w2 = np.array([1.0, 2.0, 5.0])
    assert F.pooled_winner(L, w2) == "a"
