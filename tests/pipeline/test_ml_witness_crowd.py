"""ml_witness.py's crowd(). It lives at module level with `rng` as an explicit parameter -- as a
closure nested inside main() it would not be importable and could not be unit-tested at all, unlike
m4_perseries_kstar.py's analogous logic in the same file group. It is a thin 2-line delegator to
arena_crowding.crowd(), where the actual flip-rate logic lives. These tests exercise it directly on
small synthetic score matrices, not the real ~3000-model Open LLM Leaderboard.
"""
import numpy as np

import ml_witness as MLW


def test_crowd_flip_rate_is_zero_when_mean_and_borda_winners_always_agree():
    """4 models, 3 tasks, model 0 uniformly best on every task (dominates row-wise) -- its mean loss
    AND its per-task rank are both lowest on every draw, so the flip rate must be exactly 0 at every
    pool size."""
    mat = np.array([
        [1.0, 1.0, 1.0],
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
        [7.0, 7.0, 7.0],
    ])
    rng = np.random.default_rng(0)
    out = MLW.crowd(mat, rng)
    assert set(out.keys()) == {2, 4}, "k=8..64 must all be skipped: k > len(mat)=4"
    assert out[2] == 0.0
    assert out[4] == 0.0


class _FixedChoice:
    """Stand-in for np.random.Generator that always returns the same idx array from .choice() --
    removes draw randomness so a single hand-verified (mat, idx) pair determines crowd()'s output
    exactly, across all 500 internal iterations."""
    def __init__(self, idx):
        self.idx = np.asarray(idx)

    def choice(self, n, k, replace=False):
        assert not replace and k == len(self.idx)
        return self.idx


def test_crowd_flip_rate_is_one_when_mean_and_borda_winners_disagree_on_a_fixed_draw():
    """2 models x 2 tasks: model 0 has mean 5.0, model 1 has mean 3.5 (the lower one) ->
    mean-winner=model1 (argmin). Per-task ranks (0=lowest loss), in mat's OWN row order
    (idx=[0,1], the identity draw):
    task0 ranks [1,0] (model1 lower), task1 ranks [0,1] (model0 lower) -> rank-matrix
    [[1,0],[0,1]] -> row means [0.5, 0.5], an exact tie, and np.argmin breaks ties toward the FIRST
    index -> rank-winner=model0 (index 0). mean-winner(1) != rank-winner(0): a flip. Pinning the
    draw's row order via _FixedChoice (rather than relying on natural rng.choice draws) matters
    here specifically because this fixture has a rank-mean TIE -- argmin's first-index tie-break
    means a DIFFERENT row order for the same 2-row subset can flip the tie the other way, silently
    changing the answer (caught empirically before writing this test: the untied natural-rng version
    of this fixture gave 0.554, not 1.0, for exactly this reason)."""
    mat = np.array([
        [10.0, 0.0],   # model 0: mean 5.0
        [1.0, 6.0],    # model 1: mean 3.5
    ])
    rng = _FixedChoice([0, 1])
    out = MLW.crowd(mat, rng)
    assert out[2] == 1.0


def test_crowd_stops_at_the_largest_k_not_exceeding_the_pool():
    mat = np.zeros((5, 2))
    rng = np.random.default_rng(0)
    out = MLW.crowd(mat, rng)
    assert set(out.keys()) == {2, 4}, "k=5 itself is never tried (only 2,4,8,16,32,64); k=8 exceeds len(mat)=5"


def test_crowd_is_deterministic_given_a_seeded_rng():
    rng_a = np.random.default_rng(0)
    mat = np.random.default_rng(1).normal(size=(10, 4))
    out_a = MLW.crowd(mat, rng_a)
    rng_b = np.random.default_rng(0)
    out_b = MLW.crowd(mat, rng_b)
    assert out_a == out_b
