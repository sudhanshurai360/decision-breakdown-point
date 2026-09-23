"""Covers fragility.py functions NOT covered by its sibling tests/instrument/test_fragility.py
(the pytest port of code/instrument/test_fragility.py): per_period_winner,
condorcet_winner/status, winner_stability, exchangeable_benchmark, and fragility()'s
per-opponent-tie fields.

Two of the behaviours pinned here started life as documentation of then-unfixed bugs:
pooled_winner()'s missing degenerate-tie floor, and fragility()'s single-model crash. Both are
fixed (the released package's _MARGIN_REL_FLOOR floor was ported to pooled_winner; an explicit
len(L) < 2 guard was added to fragility()), so those tests are now real regression locks, not bug
documentation.
"""
import numpy as np
import pytest

import fragility as F


# ---------------------------------------------------------------------------
# per_period_winner -- plurality winner over STRICT per-period wins, with a sorted-name tie-break
# (see the `tied = sorted(...)  # sorted -> caller-independent` line in fragility.py); verify the
# behaviour directly, don't just trust the comment.
# ---------------------------------------------------------------------------
def test_per_period_winner_majority_vote():
    # m0 wins periods 0,1; m1 wins period 2 -> m0 is the plurality winner
    L = {"m0": np.array([1.0, 1.0, 5.0]), "m1": np.array([2.0, 2.0, 1.0])}
    assert F.per_period_winner(L) == "m0"


def test_per_period_winner_tie_breaks_by_sorted_name_not_insertion_order():
    # exact tie in per-period win count -> resolves by sorted name, deterministic regardless of dict order
    L1 = {"zzz": np.array([1.0, 5.0]), "aaa": np.array([5.0, 1.0])}
    L2 = {"aaa": np.array([5.0, 1.0]), "zzz": np.array([1.0, 5.0])}
    assert F.per_period_winner(L1) == "aaa"
    assert F.per_period_winner(L2) == "aaa"
    winner, tied = F.per_period_winner(L1, return_ties=True)
    assert winner == "aaa"
    assert set(tied) == {"aaa", "zzz"}


# ---------------------------------------------------------------------------
# condorcet_winner / condorcet_status -- the three canonical voting rules; Kahn's-algorithm
# cycle detection.
# ---------------------------------------------------------------------------
def test_condorcet_winner_exists_when_one_model_beats_all():
    L = {"a": np.array([1.0, 1.0, 1.0]), "b": np.array([2.0, 2.0, 2.0]), "c": np.array([3.0, 3.0, 3.0])}
    assert F.condorcet_winner(L) == "a"
    assert F.condorcet_status(L) == "winner"


def test_condorcet_status_genuine_cycle_rock_paper_scissors():
    """A genuine intransitive 3-cycle: a beats b, b beats c, c beats a by pairwise majority, so no
    Condorcet winner exists AND the beats-graph has a real directed cycle. Asserts 'cycle'
    exactly -- `status in ("cycle", "tie")` would be vacuous, since those are the only two values
    condorcet_status() can return once condorcet_winner() is None."""
    # Rock-paper-scissors ranks, rotated one position per period (lower loss = better):
    #   period 0: a < b < c   period 1: b < c < a   period 2: c < a < b
    # so each pair splits 2-1 in favour of the "beating" model.
    L = {
        "a": np.array([1.0, 3.0, 2.0]),
        "b": np.array([2.0, 1.0, 3.0]),
        "c": np.array([3.0, 2.0, 1.0]),
    }
    assert F.condorcet_winner(L) is None
    assert F.condorcet_status(L) == "cycle"


def test_condorcet_status_tie_not_cycle_when_pairwise_ties_cause_no_winner():
    # two models tied in every period (never one strictly beats the other) -> no winner, but no
    # directed cycle either (an undirected non-edge, not a violated transitivity) -> 'tie'
    L = {"a": np.array([1.0, 1.0]), "b": np.array([1.0, 1.0])}
    assert F.condorcet_winner(L) is None
    assert F.condorcet_status(L) == "tie"


def test_condorcet_status_four_cycle_not_misclassified_as_tie():
    """Regression lock for the documented UMCSENT/rmsse fix (a genuine 4-cycle was
    previously mislabeled 'tie' by a 3-permutation-only cycle search, since fixed). Checks the
    general invariant across 50 random panels: condorcet_status()'s 'cycle' classification must
    agree with an independent Kahn's-algorithm topological-sort check of the beats-relation (a
    directed graph can be topologically sorted iff it has no cycle, so 'cycle' <=> the topological
    sort fails to cover every model)."""
    from collections import deque
    for trial_seed in range(50):
        rng = np.random.default_rng(trial_seed)
        K = int(rng.integers(3, 6))
        T = int(rng.integers(4, 10))
        Lr = {f"m{j}": rng.normal(0, 1, T) for j in range(K)}
        status = F.condorcet_status(Lr)
        cw = F.condorcet_winner(Lr)
        assert (cw is not None) == (status == "winner")
        if status in ("cycle", "tie"):
            # independently recompute Kahn's algorithm result and confirm it matches the function's claim
            models = list(Lr)
            M = {m: np.asarray(Lr[m], float) for m in models}
            def beats(x, y):
                return int((M[x] < M[y]).sum()) > int((M[y] < M[x]).sum())
            indeg = {m: 0 for m in models}
            adj = {m: [] for m in models}
            for x in models:
                for y in models:
                    if x != y and beats(x, y):
                        adj[x].append(y); indeg[y] += 1
            q = deque([m for m in models if indeg[m] == 0]); ordered = 0
            while q:
                u = q.popleft(); ordered += 1
                for v in adj[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
            expected = "cycle" if ordered < len(models) else "tie"
            assert status == expected, f"seed={trial_seed}: status={status} expected={expected}"


# ---------------------------------------------------------------------------
# winner_stability / exchangeable_benchmark -- block-bootstrap calibration.
# ---------------------------------------------------------------------------
def test_winner_stability_is_high_for_a_dominant_model():
    L = {"good": np.full(30, 1.0), "bad": np.full(30, 5.0)}
    stability, freq = F.winner_stability(L, np.ones(30), n_boot=300, seed=0)
    assert stability > 0.95
    assert freq["good"] > 0.95


def test_winner_stability_bounded_in_unit_interval():
    rng = np.random.default_rng(1)
    L = {"a": rng.normal(1, 0.3, 20), "b": rng.normal(1.05, 0.3, 20)}
    stability, freq = F.winner_stability(L, np.ones(20), n_boot=200, seed=0)
    assert 0.0 <= stability <= 1.0
    assert all(0.0 <= v <= 1.0 for v in freq.values())
    assert abs(sum(freq.values()) - 1.0) < 1e-9


def test_exchangeable_benchmark_null_median_not_at_one_over_k():
    """Locks in the documented finding that winner_stability's null does NOT go to
    1/K for K>2 -- it's closer to ~0.5 regardless of K, which is why this benchmark function exists
    at all (asserting 1/K as a threshold would be wrong)."""
    median, p05, p95 = F.exchangeable_benchmark(K=6, T=30, n_rep=60, n_boot=150, seed=0)
    assert not (0.15 < median < 0.20)  # NOT near 1/6 ~ 0.167
    assert 0.3 < median < 0.75          # roughly the documented ~0.5-ish region
    assert p05 < median < p95


# ---------------------------------------------------------------------------
# fragility() -- the per-opponent-tie fields (concentration_by_opponent, concentration_lo/hi,
# n_binding_tied, reversal_ambiguous), not covered by tests/instrument/test_fragility.py.
# ---------------------------------------------------------------------------
def test_fragility_concentration_range_contains_named_concentration():
    T = 10
    w = np.ones(T)
    L = {"a": np.full(T, 1.0), "b": np.full(T, 1.5), "c": np.full(T, 1.5)}  # b,c tied as opponents
    r = F.fragility(L, w, n_boot=100)
    assert r["n_binding_tied"] == 2, (
        "the fixture exists to produce exactly two binding opponents: b and c carry identical "
        "losses (1.5 everywhere), so both overtake a at the same k* and decision_breakdown() must "
        "return BOTH as binding ties. `>= 1` would be structurally true for any input, since "
        "decision_breakdown() always seeds ties with the binding model itself and fragility() "
        "refuses fewer than 2 models."
    )
    # Assert finiteness rather than branching on it: on this fixture concentration is a real number
    # (0.1), so a future NaN is a regression that must fail here, not silently skip the range check.
    assert np.isfinite(r["concentration"])
    assert r["concentration_lo"] <= r["concentration"] <= r["concentration_hi"] + 1e-9


def test_fragility_reversal_ambiguous_false_when_plurality_winner_is_unique():
    T = 10
    w = np.ones(T)
    # a is the clear pooled winner AND the clear per-period winner too -> no ambiguity possible
    L = {"a": np.full(T, 1.0), "b": np.full(T, 2.0)}
    r = F.fragility(L, w, n_boot=100)
    # `reversal_ambiguous = bool(len(pp_tied) > 1 and a in pp_tied)`: here per_period_tied == ['a'],
    # so `a` IS in the tied set and it is the LENGTH clause that makes this False. Both clauses
    # matter, so pin the mechanism rather than only the outcome.
    assert r["per_period_tied"] == ["a"]
    assert r["reversal_ambiguous"] is False


def test_fragility_single_model_raises_clear_valueerror_not_opaque_crash():
    """This previously crashed with the opaque 'ValueError: min() iterable argument is empty' inside
    pooled_runner_up's computation (fragility.py, the `min(sorted(m for m in L if m != a), ...)`
    a few lines into fragility()) for single-model L. It now fails fast, at the top, with a message
    naming the actual precondition."""
    T = 5
    with pytest.raises(ValueError, match="requires at least 2 models"):
        F.fragility({"only": np.arange(1.0, 6.0)}, np.ones(T), n_boot=50)


# ---------------------------------------------------------------------------
# pooled_winner() degenerate-tie floor, ported from the released selection-fragility package's
# _MARGIN_REL_FLOOR floor. Before the fix: winner
# flipped under extreme weight rescaling on the repro panel below, and disagreed with an exact
# (Fraction) reference on 76/200,000 (0.038%) random decimal-quantized uniform-weight panels via
# ordinary float64 summation non-associativity -- no weight rescaling needed to trigger it. This is
# the winner pooled_winner() feeds as `a` into decision_breakdown/fragility() for every real family
# using this file (election.*, flagship.*, comb.*).
# ---------------------------------------------------------------------------
def test_pooled_winner_weight_scale_invariant():
    L = {"m0": np.array([0, 0, 2, 2.125]), "m1": np.array([0, 0, 0, 4.125])}
    winners = {scale: F.pooled_winner(L, np.ones(4) * scale) for scale in (1, 1e-3, 1e3, 1e-6, 1e6, 1e12)}
    assert len(set(winners.values())) == 1, f"winner changes with weight scale: {winners}"


def test_pooled_winner_raises_clear_error_on_nan_mean_not_opaque_indexerror():
    """A NaN in any model's
    losses can make NaN itself become `best` (Python's min() semantics: nothing compares < NaN, so
    once NaN leads it never loses), which then makes `tied` empty (NaN <= tol is always False) and
    crashed with an opaque 'IndexError: list index out of range'. Now raises a clear, named error."""
    L = {"a": np.array([1.0, np.nan, 3.0]), "b": np.array([1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match="non-finite pooled mean"):
        F.pooled_winner(L, [1.0, 1.0, 1.0])


def test_fragility_pooled_runner_up_tie_breaks_by_sorted_name_not_insertion_order():
    """pooled_runner_up (fragility()'s internal computation of the pooled SECOND-best model) uses the
    same sorted-name tie-break convention as pooled_winner/per_period_winner/decision_breakdown --
    verify directly, not just incidentally via another test's docstring."""
    T = 10
    w = np.ones(T)
    # b and c exactly tied for runner-up behind a; sorted-name tie-break -> b (not whichever the
    # caller happened to insert first)
    L1 = {"a": np.full(T, 1.0), "c": np.full(T, 2.0), "b": np.full(T, 2.0)}
    L2 = {"a": np.full(T, 1.0), "b": np.full(T, 2.0), "c": np.full(T, 2.0)}
    assert F.fragility(L1, w, n_boot=50)["pooled_runner_up"] == "b"
    assert F.fragility(L2, w, n_boot=50)["pooled_runner_up"] == "b"


def test_pooled_winner_matches_exact_reference_at_uniform_weight():
    from fractions import Fraction

    def exact_pooled_winner(L, w):
        means = {}
        for m, arr in L.items():
            total = sum(Fraction(x).limit_denominator(10**12) * Fraction(ww) for x, ww in zip(arr, w))
            wtotal = sum(Fraction(ww) for ww in w)
            means[m] = total / wtotal
        return min(means, key=lambda m: means[m])

    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(20000):
        T = rng.integers(4, 9)
        K = rng.integers(2, 4)
        L = {f"m{j}": np.round(rng.uniform(0, 2, T), 2) for j in range(K)}
        w = np.ones(T)
        if F.pooled_winner(L, w) != exact_pooled_winner(L, w):
            mismatches += 1
    assert mismatches == 0, f"{mismatches}/20000 mismatches against exact reference at uniform weight"


# ---------------------------------------------------------------------------
# fragility(stability_block=...) -- ported from the released package (which already had
# it) so callers whose T is an EXCHANGEABLE battery (e.g. ml_witness.py's 6 Open-LLM Leaderboard
# tasks), not a genuinely time-ordered sequence, can bypass the default T**(1/3) block-bootstrap
# heuristic, which otherwise silently imposes an arbitrary adjacency structure on unordered data.
# ---------------------------------------------------------------------------
def test_fragility_stability_block_is_threaded_to_winner_stability():
    """THE CONFIRMED REGRESSION this fixes (ml_witness.py): fragility() used to hardcode
    winner_stability's block length via its own T**(1/3) default, with no way for a caller to
    override it. Verify the wiring directly, not just "some value changed": fragility()'s reported
    winner_stability for a given stability_block must equal a direct winner_stability(..., block=...)
    call on the identical data/seed -- a kwarg silently swallowed or misrouted could still produce
    SOME number without actually reaching the bootstrap."""
    rng = np.random.default_rng(3)
    T = 6
    L = {"a": rng.normal(1.0, 0.3, T), "b": rng.normal(1.05, 0.3, T), "c": rng.normal(1.1, 0.3, T)}
    w = np.ones(T)
    for block in (1, 2, None):
        r = F.fragility(L, w, n_boot=500, seed=0, stability_block=block)
        direct_stability, _ = F.winner_stability(L, w, n_boot=500, block=block, seed=0)
        assert r["winner_stability"] == pytest.approx(direct_stability), (
            f"stability_block={block}: fragility()'s winner_stability ({r['winner_stability']}) must "
            f"match a direct winner_stability(..., block={block}) call ({direct_stability})")


def test_fragility_stability_block_1_changes_result_on_short_exchangeable_panel():
    """Confirms stability_block has real effect at T=6 (the Open LLM Leaderboard's actual size,
    where the unfixed default resolves to block=round(6**(1/3))=2): a non-degenerate, contested
    panel gives a numerically different winner_stability under stability_block=1 (plain iid) than
    under the untouched default -- not a no-op parameter."""
    rng = np.random.default_rng(4)
    T = 6
    L = {"a": rng.normal(1.0, 0.4, T), "b": rng.normal(1.02, 0.4, T), "c": rng.normal(1.15, 0.4, T)}
    w = np.ones(T)
    r_default = F.fragility(L, w, n_boot=2000, seed=0)
    r_block1 = F.fragility(L, w, n_boot=2000, seed=0, stability_block=1)
    assert r_default["winner_stability"] != pytest.approx(r_block1["winner_stability"], abs=1e-9), (
        "stability_block=1 produced a numerically identical winner_stability to the untouched "
        "default -- the parameter should be changing the resample pattern at T=6 (default block=2)")


def test_fragility_stability_block_default_unchanged_for_existing_callers():
    """Negative control the other direction: omitting stability_block (every EXISTING caller of
    fragility() -- run_multiseries.py, score_multiseries.py, social_choice.py,
    flagship_fragility.py, all genuinely time-ordered annual panels; combination_benchmark.py and
    null_sensitivity.py call winner_stability()/exchangeable_benchmark() directly, bypassing
    fragility(), with their own independent block= params, so they are not affected by this port
    either way) must reproduce exactly the pre-port behavior, since the new parameter is
    keyword-only with default None. Locks that this port is purely additive."""
    rng = np.random.default_rng(5)
    T = 30
    L = {"a": rng.normal(1.0, 0.3, T), "b": rng.normal(1.03, 0.3, T)}
    w = np.ones(T)
    r = F.fragility(L, w, n_boot=300, seed=0)
    stability, _ = F.winner_stability(L, w, n_boot=300, seed=0)  # block=None -> the old T**(1/3) default
    assert r["winner_stability"] == pytest.approx(stability)


def test_winner_stability_raises_on_degenerate_block_at_least_T():
    """THE CONFIRMED REGRESSION (independently reproduced
    before fixing): at block>=T, _block_resample_idx's circular scheme draws exactly one start and
    wraps a full length-T block, so every resample is a rotation of the SAME T periods -- pooled_winner
    is invariant to period order, so winner_stability silently returned exactly 1.0 regardless of the
    data. Ported the release package's already-fixed guard into this file."""
    L = {"a": np.array([1., 2, 3, 4, 5, 6]), "b": np.array([2., 1, 4, 3, 6, 5]),
         "c": np.array([3., 3, 3, 3, 3, 3])}  # c is a constant -- deliberately NOT a genuine winner
    w = np.ones(6)
    with pytest.raises(ValueError, match="block=6 >= T=6"):
        F.winner_stability(L, w, n_boot=50, block=6, seed=0)
    with pytest.raises(ValueError, match="block=100 >= T=6"):
        F.winner_stability(L, w, n_boot=50, block=100, seed=0)


def test_winner_stability_still_works_at_block_just_under_T():
    """Negative control: the guard must not be off-by-one -- block=T-1 (still a genuine, if large,
    subsample) must NOT raise. NOTE: the original fixture ({"a":[1..6],
    "b":[2,1,4,3,6,5]}) turned out to be a genuine exact pooled tie (both means are exactly 3.5) --
    winner_stability() now correctly returns nan for a degenerate margin (a guard added alongside
    this port, from the released package), which made this test's [0,1]-range assertion
    spuriously fail even though nan is the CORRECT answer here, not a bug. Swapped to a panel with
    a real, non-degenerate winner so this test still checks what it's actually meant to (the
    block-vs-T boundary), not degeneracy handling."""
    L = {"a": np.array([1., 2, 3, 4, 5, 6]), "b": np.array([2., 1, 4, 3, 6, 10])}
    w = np.ones(6)
    stability, freq = F.winner_stability(L, w, n_boot=50, block=5, seed=0)
    assert 0.0 <= stability <= 1.0


# ---------------------------------------------------------------------------
# Three defensive guards from the documented fix
# history (winner_stability's n_boot<=0 raise, block=0 honored via `is None` not falsy,
# decision_breakdown's a=0/""/False honored via `is None` not falsy) had no dedicated
# regression test, unlike every sibling guard in this file. All three are latent (no real
# caller hits them) but worth locking in given how carefully every other guard here is tested.
# ---------------------------------------------------------------------------
def test_winner_stability_raises_on_nonpositive_n_boot():
    L = {"a": np.array([1., 2, 3, 4, 5, 6]), "b": np.array([2., 1, 4, 3, 6, 10])}
    w = np.ones(6)
    with pytest.raises(ValueError, match="n_boot must be a positive integer"):
        F.winner_stability(L, w, n_boot=0, seed=0)
    with pytest.raises(ValueError, match="n_boot must be a positive integer"):
        F.winner_stability(L, w, n_boot=-5, seed=0)


def test_winner_stability_honors_explicit_block_zero_not_falsy_default():
    """block=0 is falsy but a valid-looking explicit value; the guard must test identity against
    None, not truthiness -- an explicit block=0
    must NOT be silently replaced by the T**(1/3) default.

    NOTE: an earlier version of this
    test only asserted `0.0 <= stability <= 1.0`, which is vacuous -- at the original T=6 fixture,
    block=0 (the `block<=1` iid branch) and the T**(1/3) default (round(6**(1/3))=1, the SAME
    branch) happen to coincide, so the exact regression this test claims to guard against (an
    explicit block=0 silently replaced by the default) would still pass. Fixed with a T=27 panel
    where the default block is 3 (a genuinely different resampling scheme from block<=1), and a
    real discriminating assertion: block=0 must match block=1 (same iid code path, so identical
    given a fixed seed) and must NOT match the block=None default."""
    rng = np.random.default_rng(42)
    T = 27
    L = {"a": rng.normal(1.0, 1.0, T), "b": rng.normal(1.05, 1.0, T), "c": rng.normal(1.1, 1.0, T)}
    w = np.ones(T)
    stab_block0, _ = F.winner_stability(L, w, n_boot=500, block=0, seed=0)
    stab_block1, _ = F.winner_stability(L, w, n_boot=500, block=1, seed=0)
    stab_default, _ = F.winner_stability(L, w, n_boot=500, block=None, seed=0)
    assert stab_block0 == pytest.approx(stab_block1), (
        "block=0 and block=1 both take the block<=1 iid branch of _block_resample_idx and must "
        "produce IDENTICAL results under the same seed if block=0 is honored as a literal value"
    )
    assert stab_block0 != pytest.approx(stab_default), (
        "block=0 must not silently fall back to the T**(1/3) default (3, here) -- if it did, "
        "stab_block0 would equal stab_default instead of stab_block1"
    )


def test_decision_breakdown_honors_falsy_but_valid_a():
    """a=0/""/False are falsy but would be valid explicit model names; decision_breakdown must test
    identity against None, not truthiness -- passing a real
    model literally named "" must be honored as the champion, not silently replaced by
    pooled_winner()'s own pick.

    NOTE: an earlier fixture used
    a="0" (a non-empty string, which is TRUTHY in Python -- it never exercised the falsy path the
    `is None` guard exists for) AND made "0" the pooled winner anyway, so `a or pooled_winner(...)`
    would have yielded "0" either way -- the test passed regardless of whether the guard was
    present. Fixed with a="" (genuinely falsy) on a panel where "" is NOT the pooled winner, so a
    regression (silently replacing a="" with pooled_winner()'s own pick, "b") is distinguishable:
    reverting the `is None` guard to `a or pooled_winner(...)` changes both k* and the opponent."""
    L = {"": np.array([5., 5., 5.]), "b": np.array([1., 1., 1.])}
    w = np.ones(3)
    k, opponent, removed = F.decision_breakdown(L, w, a="")
    assert opponent == "b"  # the only other model -- confirms a="" was honored as champion, not overridden
    assert k == 0  # "" (loss=5) already loses to "b" (loss=1) with no deletions needed
