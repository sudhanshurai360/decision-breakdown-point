"""Value-pinning tests for code/instrument/fragility.py.

A mutation scan of fragility.py found that the mutations below SURVIVED the whole
fragility-touching test subset as it stood at the time although each changes a number the paper
prints:

  M12  pooled_winner ignores `w` (np.mean instead of np.average)   -> election.weight_* keys
  M11  concentration = max|c|/M instead of max(c)/M                 -> Table 1 Concentration, flagship
  M7   winner_stability resamples losses but not weights (w vs w[idx])
  M8   winner_stability non-circular blocks                          -> every stability value
  M6   breakdown_number: an exact running-margin tie no longer counts as a flip
  M15  Fragile rule k/T < 0.25 -> k/T <= 0.25

Every existing pooled-winner / concentration / stability test used either w=ones or asserted
only NaN-vs-finite / bounds, so none of these had a pinned value. Each test below is a
synthetic fixture with a HAND-COMPUTED answer (derivation in the docstring), and each
docstring names the mutant it was verified to kill.
"""
import numpy as np
import pytest

import fragility as F


# ---------------------------------------------------------------------------------------------
# pooled_winner / fragility(): the weights must actually enter the pooled mean
# ---------------------------------------------------------------------------------------------
def test_pooled_winner_weighted_and_unweighted_winners_differ_on_this_panel():
    """kills: M12 (pooled_winner ignores w).
    L = {a: [0, 0, 2.5], b: [1, 1, 1]}.
      w = [1, 1, 5]: mean_a = (0 + 0 + 5*2.5)/7 = 12.5/7 = 1.786 > mean_b = 1.0  -> 'b'
      w = [1, 1, 1]: mean_a = 2.5/3 = 0.833 < 1.0                                  -> 'a'
    A pooled winner that ignores the weights returns 'a' in both cases."""
    L = {"a": np.array([0.0, 0.0, 2.5]), "b": np.array([1.0, 1.0, 1.0])}
    assert F.pooled_winner(L, [1.0, 1.0, 5.0]) == "b"
    assert F.pooled_winner(L, [1.0, 1.0, 1.0]) == "a"
    # insertion order must not matter for the weighted verdict either
    assert F.pooled_winner({"b": L["b"], "a": L["a"]}, [1.0, 1.0, 5.0]) == "b"


def test_fragility_champion_kstar_and_concentration_follow_the_weights():
    """kills: M12 (pooled_winner ignores w) at the fragility() level -- under M12 the
    month-weighted arm hands decision_breakdown the WRONG champion, so k* collapses to 0 and
    concentration to NaN (exactly the failure mode the scan measured on the real 24 cells:
    election.weight_winner_changes 3 -> 0, weight_kstar_changes 6 -> 9).
    Same panel as above, w = [1, 1, 5]:
      champion b (1.0 < 1.786); rival a; c_t = w_t*(l_a - l_b) = [-1, -1, 7.5]; M = 5.5 > 0
      greedy deletes period 2 (c=7.5): 5.5 - 7.5 = -2 <= 0 -> k* = 1, responsible = [2]
      concentration = max(c)/M = 7.5/5.5;  k/T = 1/3
    w = [1, 1, 1]:
      champion a (0.833 < 1.0); c_t = (l_b - l_a) = [1, 1, -1.5]; M = 0.5
      greedy deletes period 0 (first of the two c=1 ties, stable argsort): 0.5 - 1 <= 0 -> k*=1,
      responsible = [0]; concentration = 1/0.5 = 2.0."""
    L = {"a": np.array([0.0, 0.0, 2.5]), "b": np.array([1.0, 1.0, 1.0])}
    r = F.fragility(L, np.array([1.0, 1.0, 5.0]), n_boot=200, seed=0)
    assert r["pooled_winner"] == "b"
    assert r["runner_up"] == "a" and r["pooled_runner_up"] == "a"
    assert r["k_star"] == 1 and r["responsible"] == [2]
    assert r["k_over_T"] == pytest.approx(1 / 3)
    assert r["concentration"] == pytest.approx(7.5 / 5.5)

    r1 = F.fragility(L, np.ones(3), n_boot=200, seed=0)
    assert r1["pooled_winner"] == "a"
    assert r1["k_star"] == 1 and r1["responsible"] == [0]
    assert r1["concentration"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------------------------
# concentration: max(c)/M, NOT max|c|/M -- pinned to hand values
# ---------------------------------------------------------------------------------------------
def test_concentration_is_largest_positive_contribution_over_margin_hand_values():
    """kills: M11 (concentration = max|c|/M).
    la = [0, 0, 0], lb = [2, 1, -2.5], w = 1: c = [2, 1, -2.5], M = 0.5,
      concentration = max(c)/M = 2/0.5 = 4.0   (max|c|/M would be 2.5/0.5 = 5.0)
    la = [0, 0, 0], lb = [3, 1, -1]:            c = [3, 1, -1],  M = 3,  concentration = 1.0
    Both panels have a genuine margin (M/T >> the 1e-12 floor), a single rival, and k*=1, so
    concentration, concentration_by_opponent[b], concentration_lo and concentration_hi must all
    equal the same hand value."""
    w = np.ones(3)
    r = F.fragility({"a": np.zeros(3), "b": np.array([2.0, 1.0, -2.5])}, w, n_boot=200, seed=0)
    assert r["pooled_winner"] == "a" and r["runner_up"] == "b" and r["k_star"] == 1
    assert r["concentration"] == pytest.approx(4.0)
    assert r["concentration_by_opponent"] == {"b": pytest.approx(4.0)}
    assert r["concentration_lo"] == pytest.approx(4.0)
    assert r["concentration_hi"] == pytest.approx(4.0)

    r2 = F.fragility({"a": np.zeros(3), "b": np.array([3.0, 1.0, -1.0])}, w, n_boot=200, seed=0)
    assert r2["concentration"] == pytest.approx(1.0)


def test_concentration_by_opponent_uses_each_rivals_own_margin():
    """kills: M11 (second, three-model shape: the |c| mutant moves the binding rival's value
    but not the non-binding one, so a test on ONE rival could be dodged by a partial revert).
    la = 0; lb = [2, 1, -2.5] (k*=1, conc 4.0); lc = [1, 1, 1] (M=3, k*=3, conc 1/3).
    Binding rival is b (k*=1 < 3), so n_binding_tied == 1 and concentration == 4.0; the
    per-opponent dict only carries the binding ties, i.e. {b: 4.0}."""
    L = {"a": np.zeros(3), "b": np.array([2.0, 1.0, -2.5]), "c": np.array([1.0, 1.0, 1.0])}
    r = F.fragility(L, np.ones(3), n_boot=200, seed=0)
    assert r["runner_up"] == "b" and r["n_binding_tied"] == 1
    assert r["concentration"] == pytest.approx(4.0)
    assert set(r["concentration_by_opponent"]) == {"b"}
    # the non-binding rival's own pairwise breakdown is the documented k*=T "beats every period" case
    assert F.breakdown_number(L["a"], L["c"], np.ones(3))[0] == 3


# ---------------------------------------------------------------------------------------------
# breakdown_number: an EXACT zero running margin is a flip (tie == 'a lost', the documented
# convention shared with _brute_force_k and _brute_force_k_multi)
# ---------------------------------------------------------------------------------------------
def test_breakdown_number_exact_zero_running_margin_counts_as_flip():
    """kills: M6 (`s <= 0` -> `s < 0`, tie no longer a flip).
    la = [0, 0, 0], lb = [1, 1, -1], w = 1: c = [1, 1, -1], M = 1.
    Deleting period 0 leaves means la = 0, lb = 0 -- an exact tie, which by the module's own
    convention means 'a' no longer strictly wins, so k* = 1 with responsible = [0]. A strict
    `s < 0` would keep deleting (period 1 too) and report k* = 2."""
    la, lb, w = np.zeros(3), np.array([1.0, 1.0, -1.0]), np.ones(3)
    k, removed = F.breakdown_number(la, lb, w)
    assert (k, removed) == (1, [0])
    assert F._brute_force_k(la, lb, w) == 1
    # and the decision-level wrapper agrees (single rival)
    assert F.decision_breakdown({"a": la, "b": lb}, w) == (1, "b", [0])


# ---------------------------------------------------------------------------------------------
# winner_stability: resamples PERIODS (losses AND weights together), circular blocks
# ---------------------------------------------------------------------------------------------
_L5 = {"a": np.array([0.0, 1.0, 1.0, 1.0, 1.0]), "b": np.array([1.0, 0.0, 0.0, 0.0, 0.0])}
_W5 = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
# On this panel `a` wins a resample iff period 0 (weight 100) is present at least once:
#   with n0 >= 1 copies of period 0 and 5-n0 other slots, weighted mean_a = (5-n0)/(100 n0 + 5-n0)
#   < mean_b = 100 n0/(100 n0 + 5 - n0)  <=>  5 - n0 < 100 n0, always true for n0 >= 1;
#   with n0 = 0: mean_a = 1 > mean_b = 0. So winner_stability = P(period 0 in the resample).


def test_winner_stability_iid_resample_carries_weights_with_periods():
    """kills: M7 (bootstrap uses w instead of w[idx]).
    block=1 (iid): P(period 0 appears in 5 draws) = 1 - (4/5)^5 = 0.67232.
    Under M7 the weight 100 stays glued to SLOT 0, so `a` wins only if slot 0 draws period 0:
    P = 1/5 = 0.20 (the other slots' weight-1 mass, at most 4, can never outweigh 100 - 4).
    n_boot = 4000 -> binomial SE 0.0074; the band below is +/- 4 SE."""
    stab, freq = F.winner_stability(_L5, _W5, n_boot=4000, block=1, seed=0)
    assert F.pooled_winner(_L5, _W5) == "a"
    assert abs(stab - (1 - 0.8 ** 5)) < 0.03, stab
    assert freq["a"] == pytest.approx(stab) and freq["a"] + freq["b"] == pytest.approx(1.0)


def test_winner_stability_circular_block_resample_hand_computed():
    """kills: M8 (non-circular blocks in winner_stability).
    block=2, T=5, circular: ceil(5/2)=3 block starts uniform on {0..4}; blocks {s, s+1 mod 5},
    {s, s+1 mod 5}, {s} (third block truncated to one slot). Period 0 is absent iff
    s1 not in {0,4}, s2 not in {0,4}, s3 != 0: (3/5)(3/5)(4/5) = 0.288 -> stability 0.712.
    Non-circular (starts uniform on {0..3}, block {s, s+1}) gives absent = (3/4)^3 = 0.4219 ->
    0.578. n_boot = 4000 -> SE 0.007; band +/- 0.03 separates the two by > 4 SE each side."""
    stab, _ = F.winner_stability(_L5, _W5, n_boot=4000, block=2, seed=0)
    assert abs(stab - 0.712) < 0.03, stab


# ---------------------------------------------------------------------------------------------
# Fragile rule: reversal AND k/T strictly below 0.25 (not <=)
# ---------------------------------------------------------------------------------------------
def test_fragile_rule_k_over_T_exactly_one_quarter_is_not_fragile():
    """kills: M15 (k/T < 0.25 -> k/T <= 0.25).
    T = 4: a = [1.01, 1.01, 1.01, 0.10], b = [1, 1, 1, 5], w = 1.
      pooled: a = 3.13/4 = 0.7825 < b = 2.0 -> champion a;  per-period: b wins 3 of 4 -> reversal
      c = lb - la = [-0.01, -0.01, -0.01, 4.9], M = 4.87; delete period 3 -> -0.03 <= 0 -> k*=1
      k/T = 1/4 = 0.25 EXACTLY (float-exact).
      stability (default block = round(4**(1/3)) = 2, circular, 2 starts): `a` wins iff period 3
      is in the resample (one 0.10-vs-5 period outweighs three 0.01 deficits); absent iff both
      starts are in {0, 1}: (1/2)^2 = 0.25 -> stability = 0.75 (>= 0.60, so clause 1 is off).
    fragile = (0.75 < 0.60) or (reversal and 0.25 < 0.25) = False. With `<=` it would be True."""
    L = {"a": np.array([1.01, 1.01, 1.01, 0.10]), "b": np.array([1.0, 1.0, 1.0, 5.0])}
    r = F.fragility(L, np.ones(4), n_boot=2000, seed=0)
    assert r["pooled_winner"] == "a" and r["per_period_winner"] == "b" and r["reversal"] is True
    assert r["k_star"] == 1 and r["responsible"] == [3]
    assert r["k_over_T"] == 0.25
    assert abs(r["winner_stability"] - 0.75) < 0.04, r["winner_stability"]
    assert r["fragile"] is False
